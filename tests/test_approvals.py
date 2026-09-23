import io
from pathlib import Path

from rich.console import Console

from assistant.security.approvals import ApprovalRequest, ApprovalStatus
from assistant.security.audit import AuditPhase, InMemoryAuditLog
from assistant.security.policy import PermissionLevel
from assistant.tools.registry import create_workspace_registry
from assistant.ui.approvals import TerminalApprovalProvider, visible


def _provider(*answers: str | BaseException) -> tuple[TerminalApprovalProvider, Console, list]:
    console = Console(file=io.StringIO(), width=100, no_color=True)
    queue = list(answers)
    asked = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return TerminalApprovalProvider(console, ask), console, asked


def _request(tool="write_file", permission=PermissionLevel.SAFE_WRITE, **arguments):
    return ApprovalRequest(tool=tool, permission=permission, arguments=arguments or {"path": "a"})


def test_panel_shows_action_every_argument_and_policy_risk():
    provider, console, _ = _provider("y")

    assert provider.request(_request(path="notes.md", content="# Notes\nline two"))

    output = console.file.getvalue()
    assert "Approval required" in output
    assert "Jamie wants to write:" in output
    assert "notes.md" in output
    assert "content: # Notes" in output and "line two" in output
    assert "Risk: SAFE_WRITE — changes local files or apps" in output
    assert "[y] allow once  [a] allow session  [n] deny" in output


def test_unknown_tools_fall_back_to_the_tool_name():
    provider, console, _ = _provider("n")
    provider.request(_request("send_email", PermissionLevel.EXTERNAL, to="x@example.com"))
    assert "Jamie wants to use send_email." in console.file.getvalue()
    assert "to: x@example.com" in console.file.getvalue()


def test_control_and_bidi_characters_are_escaped():
    assert visible("a\rb\x1b[2Kc‮exe.txt\nnext\tcol") == ("a\\rb\\x1b[2Kc\\u202eexe.txt\nnext\tcol")
    provider, console, _ = _provider("n")
    provider.request(_request(path="safe\r\x1b[2Kevil"))
    assert "safe\\r\\x1b[2Kevil" in console.file.getvalue()


def test_blank_no_eof_and_interrupt_deny():
    for answer in ("", "n", "NO", EOFError(), KeyboardInterrupt()):
        provider, _, _ = _provider(answer)
        assert provider.request(_request()) is False


def test_invalid_answer_asks_again():
    provider, console, asked = _provider("maybe", "y")
    assert provider.request(_request())
    assert len(asked) == 2
    assert "Choose [y] allow once" in console.file.getvalue()


def test_allow_session_covers_later_requests_for_the_same_tool_only():
    provider, console, asked = _provider("a", "n")

    assert provider.request(_request(path="one.md"))
    assert provider.request(_request(path="two.md"))
    assert provider.request(_request("edit_file", path="one.md")) is False

    assert len(asked) == 2
    assert "write_file allowed for this session" in console.file.getvalue()


def test_sensitive_actions_cannot_be_allowed_for_the_session():
    request = _request("run_applescript", PermissionLevel.SENSITIVE, script='display dialog "hi"')
    provider, console, asked = _provider("a", "y", "y")

    assert provider.request(request)
    assert provider.request(request)

    output = console.file.getvalue()
    assert "[y] allow once  [n] deny" in output
    assert "allow session" not in output
    assert len(asked) == 3  # "a" rejected, then asked again for the second request


def test_policy_decides_and_denial_blocks_execution(tmp_path: Path):
    (tmp_path / "keep.txt").write_text("original")
    provider, console, asked = _provider("n", "a")
    audit = InMemoryAuditLog()
    registry = create_workspace_registry(tmp_path, approvals=provider, audit=audit)

    read = registry.execute("read_file", {"path": "keep.txt"})
    denied = registry.execute("write_file", {"path": "keep.txt", "content": "changed"})
    first = registry.execute("write_file", {"path": "new.txt", "content": "1"})
    second = registry.execute("write_file", {"path": "new2.txt", "content": "2"})

    assert read.success and len(asked) == 2  # READ never prompts
    assert not denied.success and (tmp_path / "keep.txt").read_text() == "original"
    assert first.success and second.success
    outcomes = [
        (e.phase, e.approval)
        for e in audit.events
        if e.tool == "write_file" and e.phase is not AuditPhase.REQUESTED
    ]
    assert outcomes == [
        (AuditPhase.DENIED, ApprovalStatus.DENIED),
        (AuditPhase.COMPLETED, ApprovalStatus.APPROVED),
        (AuditPhase.COMPLETED, ApprovalStatus.APPROVED),
    ]
