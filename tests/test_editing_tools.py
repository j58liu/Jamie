import json
from pathlib import Path

import pytest

from assistant.agent import run_agent
from assistant.llm.base import LLMResponse
from assistant.security import CallbackApprovalProvider, PermissionLevel
from assistant.tools import create_workspace_registry


class NoModel:
    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        raise AssertionError("Slash commands must not call the model")


def approving_registry(root: Path):
    requests = []

    def approve(request):
        requests.append(request)
        return True

    registry = create_workspace_registry(
        root,
        approvals=CallbackApprovalProvider(approve),
    )
    return registry, requests


def test_workspace_registry_exposes_read_and_editing_tools(tmp_path: Path):
    registry, _ = approving_registry(tmp_path)
    definitions = registry.definitions()
    assert [definition["name"] for definition in definitions] == [
        "read_file",
        "list_directory",
        "find_files",
        "search_files",
        "write_file",
        "edit_file",
        "move_file",
        "create_directory",
        "trash_file",
    ]
    risks = {definition["name"]: definition["risk"] for definition in definitions}
    assert risks["write_file"] == PermissionLevel.SAFE_WRITE.value
    assert risks["trash_file"] == PermissionLevel.SAFE_WRITE.value


@pytest.mark.usefixtures("confirm_every_tool")
def test_write_is_denied_by_default_before_snapshot_or_mutation(tmp_path: Path):
    registry = create_workspace_registry(tmp_path)
    result = registry.execute("write_file", {"path": "new.txt", "content": "new"})
    assert not result.success
    assert not (tmp_path / "new.txt").exists()
    assert registry.handle_command("/undo") == "Nothing to undo."


@pytest.mark.usefixtures("confirm_every_tool")
def test_edit_diff_verification_and_undo(tmp_path: Path):
    source = tmp_path / "math.py"
    source.write_text("def add(a, b):\n    return a - b\n")
    registry, requests = approving_registry(tmp_path)
    registry.begin_turn()

    result = registry.execute(
        "edit_file",
        {
            "path": "math.py",
            "old_text": "return a - b",
            "new_text": "return a + b",
        },
    )

    assert result.success
    assert "-    return a - b" in result.output
    assert "+    return a + b" in result.output
    assert len(requests) == 1
    assert requests[0].arguments["path"] == "math.py"
    assert registry.verification_message() is not None
    diff = registry.handle_command("/diff")
    assert "-    return a - b" in diff
    assert "+    return a + b" in diff

    assert registry.execute("read_file", {"path": "math.py"}).success
    assert registry.verification_message() is None
    registry.end_turn()
    assert run_agent(NoModel(), registry, "/diff") == diff
    assert run_agent(NoModel(), registry, "/undo") == "Undid the most recent turn."
    assert source.read_text() == "def add(a, b):\n    return a - b\n"


def test_snapshot_is_original_state_across_multiple_edits(tmp_path: Path):
    file = tmp_path / "value.txt"
    file.write_text("one\n")
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    assert registry.execute("write_file", {"path": "value.txt", "content": "two\n"}).success
    assert registry.execute("write_file", {"path": "value.txt", "content": "three\n"}).success
    registry.end_turn()

    assert file.read_text() == "three\n"
    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert file.read_text() == "one\n"


def test_undo_restores_only_the_most_recent_turn(tmp_path: Path):
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    assert registry.execute("write_file", {"path": "first.txt", "content": "first"}).success
    registry.end_turn()
    registry.begin_turn()
    assert registry.execute("write_file", {"path": "second.txt", "content": "second"}).success
    registry.end_turn()

    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert (tmp_path / "first.txt").read_text() == "first"
    assert not (tmp_path / "second.txt").exists()
    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert not (tmp_path / "first.txt").exists()


def test_empty_file_creation_has_diff_and_can_be_undone(tmp_path: Path):
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    result = registry.execute("write_file", {"path": "empty.txt", "content": ""})
    registry.end_turn()

    assert result.success
    assert "+empty.txt" in result.output
    assert (tmp_path / "empty.txt").is_file()
    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert not (tmp_path / "empty.txt").exists()


@pytest.mark.usefixtures("confirm_every_tool")
def test_move_create_and_trash_are_undone_as_one_turn(tmp_path: Path):
    (tmp_path / "source.txt").write_text("source")
    (tmp_path / "trash.txt").write_text("trash")
    registry, requests = approving_registry(tmp_path)
    registry.begin_turn()

    assert registry.execute(
        "move_file", {"source": "source.txt", "destination": "moved.txt"}
    ).success
    assert registry.execute("create_directory", {"path": "nested/empty", "parents": True}).success
    assert registry.execute("trash_file", {"path": "trash.txt"}).success
    registry.end_turn()

    assert not (tmp_path / "source.txt").exists()
    assert (tmp_path / "moved.txt").read_text() == "source"
    assert (tmp_path / "nested/empty").is_dir()
    assert not (tmp_path / "trash.txt").exists()
    assert [request.permission for request in requests] == [
        PermissionLevel.SAFE_WRITE,
        PermissionLevel.SAFE_WRITE,
        PermissionLevel.SAFE_WRITE,
    ]

    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert (tmp_path / "source.txt").read_text() == "source"
    assert not (tmp_path / "moved.txt").exists()
    assert not (tmp_path / "nested").exists()
    assert (tmp_path / "trash.txt").read_text() == "trash"


def test_undo_restores_original_nested_content_after_parent_is_trashed(tmp_path: Path):
    nested = tmp_path / "parent/child.txt"
    nested.parent.mkdir()
    nested.write_text("original")
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    assert registry.execute(
        "edit_file",
        {"path": "parent/child.txt", "old_text": "original", "new_text": "edited"},
    ).success
    assert registry.execute("trash_file", {"path": "parent"}).success
    registry.end_turn()

    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert nested.read_text() == "original"


def test_undo_removes_new_child_when_parent_snapshot_was_taken_later(tmp_path: Path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "original.txt").write_text("original")
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    assert registry.execute("write_file", {"path": "parent/new.txt", "content": "new"}).success
    assert registry.execute("trash_file", {"path": "parent"}).success
    registry.end_turn()

    assert registry.handle_command("/undo") == "Undid the most recent turn."
    assert (parent / "original.txt").read_text() == "original"
    assert not (parent / "new.txt").exists()


def test_edit_requires_unique_match_unless_replace_all(tmp_path: Path):
    file = tmp_path / "repeat.txt"
    file.write_text("old old")
    registry, _ = approving_registry(tmp_path)
    ambiguous = registry.execute(
        "edit_file",
        {"path": "repeat.txt", "old_text": "old", "new_text": "new"},
    )
    assert not ambiguous.success
    assert "occurs 2 times" in ambiguous.output
    assert file.read_text() == "old old"

    replaced = registry.execute(
        "edit_file",
        {
            "path": "repeat.txt",
            "old_text": "old",
            "new_text": "new",
            "replace_all": True,
        },
    )
    assert replaced.success
    assert file.read_text() == "new new"


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt"])
def test_editing_paths_cannot_leave_workspace(tmp_path: Path, path: str):
    registry, _ = approving_registry(tmp_path)
    result = registry.execute("write_file", {"path": path, "content": "blocked"})
    assert not result.success


def test_editing_through_symlink_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / "outside-edit.txt"
    outside.write_text("original")
    (tmp_path / "link.txt").symlink_to(outside)
    registry, _ = approving_registry(tmp_path)

    result = registry.execute("write_file", {"path": "link.txt", "content": "changed"})

    assert not result.success
    assert "outside" in result.output or "symbolic links" in result.output
    assert outside.read_text() == "original"


def test_agent_must_verify_changed_file_before_finish(tmp_path: Path):
    file = tmp_path / "answer.txt"
    file.write_text("wrong\n")
    responses = iter(
        [
            json.dumps(
                {
                    "action": "tool",
                    "tool": "edit_file",
                    "args": {
                        "path": "answer.txt",
                        "old_text": "wrong",
                        "new_text": "right",
                    },
                }
            ),
            json.dumps({"action": "finish", "message": "Changed."}),
            json.dumps(
                {
                    "action": "tool",
                    "tool": "read_file",
                    "args": {"path": "answer.txt"},
                }
            ),
            json.dumps({"action": "finish", "message": "Changed and verified."}),
        ]
    )

    class Model:
        def __init__(self):
            self.requests = []

        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            self.requests.append(list(messages))
            return LLMResponse(next(responses))

    model = Model()
    registry, _ = approving_registry(tmp_path)
    answer = run_agent(model, registry, "Fix answer.txt.")

    assert answer == "Changed and verified."
    assert file.read_text() == "right\n"
    assert "have not verified" in model.requests[2][-1].content
    assert 'read_file with path="answer.txt"' in model.requests[2][-1].content
    assert run_agent(NoModel(), registry, "/undo") == "Undid the most recent turn."
    assert file.read_text() == "wrong\n"


def test_unrelated_read_does_not_verify_changed_file(tmp_path: Path):
    (tmp_path / "other.txt").write_text("other")
    registry, _ = approving_registry(tmp_path)
    registry.begin_turn()
    assert registry.execute("write_file", {"path": "changed.txt", "content": "new"}).success
    assert registry.execute("read_file", {"path": "other.txt"}).success

    message = registry.verification_message()
    assert message is not None
    assert 'read_file with path="changed.txt"' in message
