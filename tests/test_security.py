import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from assistant.agent import run_agent
from assistant.llm.base import LLMResponse
from assistant.security import (
    DEFAULT_TOOL_PERMISSIONS,
    ApprovalRequest,
    AuditError,
    AuditPhase,
    CallbackApprovalProvider,
    InMemoryAuditLog,
    JsonlAuditLog,
    PermissionLevel,
    PermissionPolicy,
    PolicyAction,
    PolicyError,
    redact_arguments,
)
from assistant.security.audit import MAX_ARGUMENT_CHARACTERS
from assistant.tools import ReadFileTool, ToolRegistry


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RecordingTool:
    description = "Record whether execution was reached."
    input_schema = EmptyInput

    def __init__(self, name: str, risk: PermissionLevel) -> None:
        self.name = name
        self.risk = risk
        self.executions = 0

    def execute(self, arguments: EmptyInput) -> str:
        self.executions += 1
        return "executed"


def test_default_policy_classifies_named_tools():
    assert DEFAULT_TOOL_PERMISSIONS == {
        "read_file": PermissionLevel.READ,
        "list_directory": PermissionLevel.READ,
        "find_files": PermissionLevel.READ,
        "search_files": PermissionLevel.READ,
        "extract_page": PermissionLevel.READ,
        "open_application": PermissionLevel.SAFE_WRITE,
        "reveal_in_finder": PermissionLevel.SAFE_WRITE,
        "clipboard_write": PermissionLevel.SAFE_WRITE,
        "move_file": PermissionLevel.SAFE_WRITE,
        "write_file": PermissionLevel.SAFE_WRITE,
        "edit_file": PermissionLevel.SAFE_WRITE,
        "create_directory": PermissionLevel.SAFE_WRITE,
        "shell": PermissionLevel.SENSITIVE,
        "clipboard_read": PermissionLevel.SENSITIVE,
        "run_shortcut": PermissionLevel.SENSITIVE,
        "run_applescript": PermissionLevel.SENSITIVE,
        "send_email": PermissionLevel.EXTERNAL,
        "web_search": PermissionLevel.EXTERNAL,
        "fetch_url": PermissionLevel.EXTERNAL,
        "research": PermissionLevel.EXTERNAL,
        "delete_file": PermissionLevel.DESTRUCTIVE,
        "trash_file": PermissionLevel.SAFE_WRITE,
    }


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("read_file", PolicyAction.ALLOW),
        ("write_file", PolicyAction.CONFIRM),
        ("shell", PolicyAction.CONFIRM),
        ("send_email", PolicyAction.CONFIRM),
        ("delete_file", PolicyAction.CONFIRM),
    ],
)
def test_policy_allows_only_read_actions_without_confirmation(tool: str, expected: PolicyAction):
    assert PermissionPolicy().decide(tool).action is expected


def test_policy_rejects_missing_or_conflicting_tool_declarations():
    policy = PermissionPolicy()
    with pytest.raises(PolicyError, match="no permission policy"):
        policy.permission_for("unclassified")
    with pytest.raises(PolicyError, match="policy requires SENSITIVE"):
        policy.validate_declaration("shell", PermissionLevel.READ)


def test_read_action_executes_without_calling_approval_provider(tmp_path: Path):
    file = tmp_path / "README.md"
    file.write_text("hello")

    class UnexpectedApproval:
        def request(self, approval: ApprovalRequest) -> bool:
            raise AssertionError("READ actions must not ask for approval")

    audit = InMemoryAuditLog()
    registry = ToolRegistry(approvals=UnexpectedApproval(), audit=audit)
    registry.register(ReadFileTool(tmp_path))
    result = registry.execute("read_file", {"path": "README.md"})

    assert result.success
    assert result.output == "hello"
    assert [event.phase for event in audit.events] == [
        AuditPhase.REQUESTED,
        AuditPhase.COMPLETED,
    ]
    assert audit.events[-1].permission is PermissionLevel.READ
    assert audit.events[-1].approval.value == "not_required"


def test_sensitive_action_fails_closed_without_approval():
    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    audit = InMemoryAuditLog()
    registry = ToolRegistry(audit=audit)
    registry.register(tool)

    result = registry.execute("shell", {})

    assert not result.success
    assert "not granted" in result.output
    assert tool.executions == 0
    assert [event.phase for event in audit.events] == [
        AuditPhase.REQUESTED,
        AuditPhase.DENIED,
    ]


def test_sensitive_action_executes_only_after_bound_approval():
    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        return True

    audit = InMemoryAuditLog()
    registry = ToolRegistry(
        approvals=CallbackApprovalProvider(approve),
        audit=audit,
    )
    registry.register(tool)

    result = registry.execute("shell", {})

    assert result.success
    assert tool.executions == 1
    assert len(requests) == 1
    assert requests[0].tool == "shell"
    assert requests[0].permission is PermissionLevel.SENSITIVE
    assert requests[0].arguments == {}
    assert audit.events[-1].phase is AuditPhase.COMPLETED
    assert audit.events[-1].approval.value == "approved"


def test_model_arguments_cannot_claim_approval():
    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    approvals = []
    registry = ToolRegistry(
        approvals=CallbackApprovalProvider(lambda request: approvals.append(request) or True)
    )
    registry.register(tool)

    result = registry.execute("shell", {"approved": True})

    assert not result.success
    assert "Invalid arguments" in result.output
    assert approvals == []
    assert tool.executions == 0


def test_agent_action_cannot_bypass_permission_engine():
    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            self.calls += 1
            return LLMResponse('{"action":"tool","tool":"shell","args":{}}')

    model = Model()
    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    registry = ToolRegistry()
    registry.register(tool)

    answer = run_agent(model, registry, "Run a command.")

    assert answer == "Stopped: shell was not approved."
    assert tool.executions == 0
    assert model.calls == 1


def test_invalid_approval_response_fails_closed():
    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    registry = ToolRegistry(
        approvals=CallbackApprovalProvider(lambda request: "yes")  # type: ignore[arg-type]
    )
    registry.register(tool)

    result = registry.execute("shell", {})

    assert not result.success
    assert "could not be obtained" in result.output
    assert tool.executions == 0


def test_registry_rejects_tool_without_central_policy():
    tool = RecordingTool("custom_tool", PermissionLevel.READ)
    with pytest.raises(PolicyError, match="no permission policy"):
        ToolRegistry().register(tool)


def test_jsonl_audit_records_redacted_arguments(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    registry = ToolRegistry(audit=JsonlAuditLog(path))
    registry.register(ReadFileTool(tmp_path))
    (tmp_path / "private.txt").write_text("private")

    assert registry.execute("read_file", {"path": "private.txt"}).success

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert [event["phase"] for event in events] == ["requested", "completed"]
    assert all(event["arguments"]["path"] == "private.txt" for event in events)


def test_audit_failure_blocks_execution():
    class BrokenAudit:
        def record(self, event):
            raise AuditError("unavailable")

    tool = RecordingTool("shell", PermissionLevel.SENSITIVE)
    registry = ToolRegistry(
        approvals=CallbackApprovalProvider(lambda request: True),
        audit=BrokenAudit(),
    )
    registry.register(tool)

    with pytest.raises(AuditError, match="unavailable"):
        registry.execute("shell", {})
    assert tool.executions == 0


def test_audit_truncates_long_argument_values():
    recorded = redact_arguments({"path": "notes.txt", "content": "x" * 5_000, "limit": 4})

    assert recorded["path"] == "notes.txt"
    assert recorded["limit"] == 4
    assert recorded["content"] == f"{'x' * MAX_ARGUMENT_CHARACTERS}[truncated]"
