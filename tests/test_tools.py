from pathlib import Path

import pytest

from assistant.agent.loop import run_agent
from assistant.llm.base import LLMResponse
from assistant.tools import ReadFileTool, Risk, ToolRegistry, create_read_only_registry


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "README.md").write_text("first line\nhello world\nlast line\n")
    (root / "empty.txt").write_text("")
    source = root / "src"
    source.mkdir()
    (source / "alpha.py").write_text("print('orchid')\n")
    nested = source / "nested"
    nested.mkdir()
    (nested / "beta.py").write_text("# ORCHID\nprint('beta')\n")
    (nested / "notes.txt").write_text("orchid notes\n")
    (root / "binary.bin").write_bytes(b"\xff\xfe\x00")
    return root


def test_default_registry_contains_only_four_read_only_tools(sandbox: Path):
    registry = create_read_only_registry(sandbox)
    definitions = registry.definitions()
    assert [definition["name"] for definition in definitions] == [
        "read_file",
        "list_directory",
        "find_files",
        "search_files",
    ]
    assert {definition["risk"] for definition in definitions} == {Risk.READ.value}
    assert all(definition["description"] for definition in definitions)
    assert all(definition["input_schema"]["type"] == "object" for definition in definitions)
    assert "shell" not in registry.instructions()


def test_read_file_reads_requested_line_range(sandbox: Path):
    registry = create_read_only_registry(sandbox)
    result = registry.execute(
        "read_file",
        {"path": "README.md", "offset": 1, "limit": 1},
    )
    assert result.success
    assert result.output == "hello world\n"


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside.txt",
        "src",
        "missing.txt",
    ],
)
def test_read_file_rejects_unsafe_or_non_file_paths(sandbox: Path, path: str):
    result = create_read_only_registry(sandbox).execute("read_file", {"path": path})
    assert not result.success


def test_read_file_rejects_symlink_that_leaves_workspace(sandbox: Path, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (sandbox / "escape.txt").symlink_to(outside)

    result = create_read_only_registry(sandbox).execute("read_file", {"path": "escape.txt"})
    assert not result.success
    assert "outside" in result.output


def test_list_directory_is_sorted_and_marks_entry_types(sandbox: Path):
    (sandbox / "link").symlink_to(sandbox / "README.md")
    result = create_read_only_registry(sandbox).execute("list_directory", {"path": "."})
    assert result.success
    assert result.output.splitlines() == [
        "README.md",
        "binary.bin",
        "empty.txt",
        "link@",
        "src/",
    ]


def test_list_directory_limits_output(sandbox: Path):
    result = create_read_only_registry(sandbox).execute("list_directory", {"path": ".", "limit": 2})
    assert result.success
    assert result.output.splitlines() == ["README.md", "binary.bin", "[truncated]"]


def test_find_files_recurses_and_returns_workspace_relative_paths(sandbox: Path):
    result = create_read_only_registry(sandbox).execute(
        "find_files", {"path": "src", "pattern": "*.py"}
    )
    assert result.success
    assert result.output.splitlines() == ["src/alpha.py", "src/nested/beta.py"]


@pytest.mark.parametrize("pattern", ["", "../*", "/tmp/*"])
def test_find_files_rejects_unsafe_patterns(sandbox: Path, pattern: str):
    result = create_read_only_registry(sandbox).execute(
        "find_files", {"path": ".", "pattern": pattern}
    )
    assert not result.success
    assert "Invalid arguments" in result.output


def test_search_files_returns_literal_matches_with_line_numbers(sandbox: Path):
    result = create_read_only_registry(sandbox).execute(
        "search_files",
        {
            "path": "src",
            "pattern": "*.py",
            "query": "orchid",
            "case_sensitive": False,
        },
    )
    assert result.success
    assert result.output.splitlines() == [
        "src/alpha.py:1:print('orchid')",
        "src/nested/beta.py:1:# ORCHID",
    ]


def test_search_files_skips_non_utf8_files(sandbox: Path):
    result = create_read_only_registry(sandbox).execute(
        "search_files", {"path": ".", "pattern": "*", "query": "orchid"}
    )
    assert result.success
    assert "binary.bin" not in result.output
    assert "src/alpha.py" in result.output


def test_registry_returns_failures_for_unknown_tools_and_invalid_arguments(sandbox: Path):
    registry = create_read_only_registry(sandbox)
    unknown = registry.execute("run_shell", {"command": "pwd"})
    invalid = registry.execute("read_file", {"path": "README.md", "unexpected": True})
    wrong_type = registry.execute("read_file", {"path": "README.md", "offset": "1"})
    assert not unknown.success
    assert "Unknown tool" in unknown.output
    assert not invalid.success
    assert "Invalid arguments" in invalid.output
    assert not wrong_type.success


def test_duplicate_tool_registration_is_rejected(sandbox: Path):
    first = create_read_only_registry(sandbox)
    with pytest.raises(ValueError, match="already registered"):
        first.register(ReadFileTool(sandbox))


def test_registry_instructions_are_sent_to_model(sandbox: Path):
    class Model:
        def __init__(self):
            self.messages = None

        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            self.messages = list(messages)
            return LLMResponse('{"action":"finish","message":"Done."}')

    model = Model()
    assert run_agent(model, create_read_only_registry(sandbox), "Finish.") == "Done."
    instructions = model.messages[0].content
    assert "Available tools" in instructions
    assert '"name":"read_file"' in instructions
    assert '"risk":"READ"' in instructions


def test_agent_reads_real_file_before_answering(sandbox: Path):
    class Model:
        def __init__(self):
            self.call = 0

        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            self.call += 1
            if self.call == 1:
                return LLMResponse(
                    '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}'
                )
            assert "hello world" in messages[-1].content
            return LLMResponse('{"action":"finish","message":"README contains hello world."}')

    answer = run_agent(
        Model(),
        create_read_only_registry(sandbox),
        "What does README.md contain?",
    )
    assert "hello world" in answer


def test_registry_requires_existing_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="Workspace root"):
        create_read_only_registry(tmp_path / "missing")

    file = tmp_path / "file.txt"
    file.write_text("text")
    with pytest.raises(ValueError, match="must be a directory"):
        create_read_only_registry(file)


def test_registry_rejects_invalid_tool_metadata():
    class Input:
        pass

    class InvalidTool:
        name = "Run Shell"
        description = "bad"
        input_schema = Input
        risk = Risk.READ

        def execute(self, arguments):
            return ""

    with pytest.raises(ValueError, match="Tool names"):
        ToolRegistry().register(InvalidTool())
