import subprocess
from pathlib import Path

import pytest

from assistant.security import CallbackApprovalProvider, PermissionLevel
from assistant.tools import (
    ClipboardReadTool,
    CommandResult,
    OpenApplicationTool,
    SubprocessCommandRunner,
    create_macos_registry,
)


class FakeRunner:
    def __init__(self, outputs: list[str] | None = None):
        self.outputs = iter(outputs or [])
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append((arguments, input_text, timeout))
        return CommandResult(stdout=next(self.outputs, ""), stderr="")


def approved_registry(root: Path, runner: FakeRunner):
    requests = []
    registry = create_macos_registry(
        root,
        approvals=CallbackApprovalProvider(lambda request: requests.append(request) or True),
        runner=runner,
    )
    return registry, requests


def test_macos_registry_exposes_six_classified_tools(tmp_path: Path):
    registry, _ = approved_registry(tmp_path, FakeRunner())
    risks = {definition["name"]: definition["risk"] for definition in registry.definitions()}
    assert risks["open_application"] == PermissionLevel.SAFE_WRITE.value
    assert risks["reveal_in_finder"] == PermissionLevel.SAFE_WRITE.value
    assert risks["clipboard_read"] == PermissionLevel.SENSITIVE.value
    assert risks["clipboard_write"] == PermissionLevel.SAFE_WRITE.value
    assert risks["run_shortcut"] == PermissionLevel.SENSITIVE.value
    assert risks["run_applescript"] == PermissionLevel.SENSITIVE.value


def test_open_application_uses_fixed_argument_array_after_approval(tmp_path: Path):
    runner = FakeRunner()
    registry, requests = approved_registry(tmp_path, runner)
    result = registry.execute("open_application", {"name": "Safari"})
    assert result.success
    assert result.output == "Opened application Safari."
    assert runner.calls == [(["open", "-a", "Safari"], None, 30)]
    assert requests[0].tool == "open_application"
    assert requests[0].arguments == {"name": "Safari"}


def test_reveal_in_finder_resolves_only_workspace_paths(tmp_path: Path):
    file = tmp_path / "README.md"
    file.write_text("read me")
    runner = FakeRunner()
    registry, _ = approved_registry(tmp_path, runner)
    result = registry.execute("reveal_in_finder", {"path": "README.md"})
    escaped = registry.execute("reveal_in_finder", {"path": "../outside.txt"})
    assert result.success
    assert runner.calls == [(["open", "-R", str(file)], None, 30)]
    assert not escaped.success


def test_clipboard_read_and_write_use_pbpaste_and_pbcopy(tmp_path: Path):
    runner = FakeRunner(["copied text", ""])
    registry, requests = approved_registry(tmp_path, runner)
    read = registry.execute("clipboard_read", {})
    write = registry.execute("clipboard_write", {"text": "replacement"})
    assert read.success
    assert read.output == "copied text"
    assert write.success
    assert runner.calls == [
        (["pbpaste"], None, 10),
        (["pbcopy"], "replacement", 10),
    ]
    assert [request.permission for request in requests] == [
        PermissionLevel.SENSITIVE,
        PermissionLevel.SAFE_WRITE,
    ]


def test_shortcut_and_applescript_use_fixed_commands(tmp_path: Path):
    runner = FakeRunner(["shortcut output\n", "42\n"])
    registry, _ = approved_registry(tmp_path, runner)
    shortcut = registry.execute("run_shortcut", {"name": "Morning Routine"})
    script = registry.execute("run_applescript", {"script": "return 42"})
    assert shortcut.success
    assert "shortcut output" in shortcut.output
    assert script.success
    assert "42" in script.output
    assert runner.calls == [
        (["shortcuts", "run", "Morning Routine"], None, 300),
        (["osascript", "-e", "return 42"], None, 60),
    ]


def test_default_denial_prevents_macos_command(tmp_path: Path):
    runner = FakeRunner()
    registry = create_macos_registry(tmp_path, runner=runner)
    result = registry.execute("open_application", {"name": "Safari"})
    assert not result.success
    assert runner.calls == []


@pytest.mark.parametrize("name", ["", "  ", "-option", "bad\nname", "bad\x00name"])
def test_application_and_shortcut_names_are_validated(name: str):
    runner = FakeRunner()
    application = OpenApplicationTool(runner)
    with pytest.raises(ValueError):
        application.input_schema.model_validate({"name": name})
    assert runner.calls == []


def test_empty_clipboard_has_explicit_result():
    tool = ClipboardReadTool(FakeRunner([""]))
    result = tool.execute(tool.Input())
    assert result == "Clipboard is empty."


def test_subprocess_runner_never_uses_shell(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout="ok", stderr="")

    monkeypatch.setattr("assistant.tools.macos.sys.platform", "darwin")
    monkeypatch.setattr("assistant.tools.macos.subprocess.run", fake_run)
    result = SubprocessCommandRunner().run(["open", "-a", "Safari"], timeout=30)
    assert result.stdout == "ok"
    assert calls[0][0] == ["open", "-a", "Safari"]
    assert "shell" not in calls[0][1]


def test_subprocess_failure_is_safe_tool_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(arguments, **kwargs):
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr="not found")

    monkeypatch.setattr("assistant.tools.macos.sys.platform", "darwin")
    monkeypatch.setattr("assistant.tools.macos.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="macOS command failed"):
        SubprocessCommandRunner().run(["open", "-a", "Missing"], timeout=30)
