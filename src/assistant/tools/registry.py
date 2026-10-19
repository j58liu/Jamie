"""Registration, schemas, and execution for agent tools."""

import json
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from assistant.agent.loop import ToolResult
from assistant.llm.base import LLMBackend
from assistant.security.approvals import (
    ApprovalError,
    ApprovalProvider,
    ApprovalStatus,
    DenyAllApprovals,
    authorize,
)
from assistant.security.audit import (
    AuditEvent,
    AuditPhase,
    AuditSink,
    InMemoryAuditLog,
    redact_arguments,
)
from assistant.security.policy import (
    PermissionLevel,
    PermissionPolicy,
    PolicyAction,
    PolicyDecision,
)
from assistant.tools.base import Tool, ToolError
from assistant.tools.changes import WorkspaceChanges
from assistant.tools.editing import (
    CreateDirectoryTool,
    EditFileTool,
    MoveFileTool,
    TrashFileTool,
    WriteFileTool,
)
from assistant.tools.filesystem import (
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
)
from assistant.tools.macos import (
    ClipboardReadTool,
    ClipboardWriteTool,
    CommandRunner,
    OpenApplicationTool,
    RevealInFinderTool,
    RunAppleScriptTool,
    RunShortcutTool,
)
from assistant.tools.web import ExtractPageTool, FetchUrlTool, WebClient, WebSearchTool


def _compact_schema(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _compact_schema(item)
            for key, item in value.items()
            if key not in {"title", "$schema"}
        }
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value


class ToolRegistry:
    """Validate and dispatch named tools for the agent loop."""

    def __init__(
        self,
        *,
        policy: PermissionPolicy | None = None,
        approvals: ApprovalProvider | None = None,
        audit: AuditSink | None = None,
        changes: WorkspaceChanges | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._policy = policy if policy is not None else PermissionPolicy()
        self._approvals = approvals if approvals is not None else DenyAllApprovals()
        self._audit = audit if audit is not None else InMemoryAuditLog()
        self._changes = changes

    @property
    def audit(self) -> AuditSink:
        return self._audit

    def close(self) -> None:
        """Release workspace snapshots; shared web clients belong to the caller."""
        if self._changes is not None:
            self._changes.close()

    def register(self, tool: Tool) -> None:
        self._policy.validate_declaration(tool.name, tool.risk)
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, object]]:
        """Return the registered tool metadata included in the model prompt."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
                "risk": tool.risk.value,
            }
            for tool in self._tools.values()
        ]

    def instructions(self) -> str:
        """Describe the registered tools for the model."""
        prompt_definitions = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": _compact_schema(tool.input_schema.model_json_schema()),
                "risk": tool.risk.value,
            }
            for tool in self._tools.values()
        ]
        definitions = json.dumps(prompt_definitions, ensure_ascii=False, separators=(",", ":"))
        examples = {
            "read_file": '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}',
            "list_directory": '{"action":"tool","tool":"list_directory","args":{}}',
            "find_files": ('{"action":"tool","tool":"find_files","args":{"pattern":"*.md"}}'),
            "search_files": ('{"action":"tool","tool":"search_files","args":{"query":"text"}}'),
            "write_file": (
                '{"action":"tool","tool":"write_file","args":{"path":"file.txt","content":"text"}}'
            ),
            "edit_file": (
                '{"action":"tool","tool":"edit_file","args":{"path":"file.txt",'
                '"old_text":"before","new_text":"after"}}'
            ),
            "move_file": (
                '{"action":"tool","tool":"move_file","args":{"source":"old.txt",'
                '"destination":"new.txt"}}'
            ),
            "create_directory": (
                '{"action":"tool","tool":"create_directory","args":{"path":"new"}}'
            ),
            "trash_file": ('{"action":"tool","tool":"trash_file","args":{"path":"old.txt"}}'),
            "open_application": (
                '{"action":"tool","tool":"open_application","args":{"name":"Safari"}}'
            ),
            "reveal_in_finder": (
                '{"action":"tool","tool":"reveal_in_finder","args":{"path":"README.md"}}'
            ),
            "clipboard_read": ('{"action":"tool","tool":"clipboard_read","args":{}}'),
            "clipboard_write": (
                '{"action":"tool","tool":"clipboard_write","args":{"text":"hello"}}'
            ),
            "run_shortcut": (
                '{"action":"tool","tool":"run_shortcut","args":{"name":"Morning Routine"}}'
            ),
            "run_applescript": (
                '{"action":"tool","tool":"run_applescript","args":{"script":"return 1"}}'
            ),
            "web_search": (
                '{"action":"tool","tool":"web_search","args":{"query":"current topic"}}'
            ),
            "fetch_url": (
                '{"action":"tool","tool":"fetch_url","args":{"url":"https://example.com/"}}'
            ),
            "extract_page": (
                '{"action":"tool","tool":"extract_page","args":{"url":"https://example.com/"}}'
            ),
            "research": (
                '{"action":"tool","tool":"research","args":{"question":"What changed in X?"}}'
            ),
        }
        registered_examples = "\n".join(
            examples.get(
                name,
                json.dumps(
                    {"action": "tool", "tool": name, "args": {}},
                    separators=(",", ":"),
                ),
            )
            for name in self._tools
        )
        return (
            f"Available tools and their argument schemas:\n{definitions}\n"
            "The schema properties belong inside the action's args object. Examples:\n"
            f"{registered_examples}"
        )

    def begin_turn(self) -> None:
        if self._changes is not None:
            self._changes.begin_turn()

    def end_turn(self) -> None:
        if self._changes is not None:
            self._changes.end_turn()

    def verification_message(self) -> str | None:
        return self._changes.verification_message() if self._changes is not None else None

    def handle_command(self, command: str) -> str | None:
        if self._changes is None or command not in {"/diff", "/undo"}:
            return None
        if command == "/diff":
            decision = PolicyDecision(command, PermissionLevel.READ, PolicyAction.ALLOW)
            approval = ApprovalStatus.NOT_REQUIRED
        else:
            decision = PolicyDecision(command, PermissionLevel.SAFE_WRITE, PolicyAction.CONFIRM)
            approval = ApprovalStatus.APPROVED
        self._record(command, {}, AuditPhase.REQUESTED, decision=decision)
        try:
            result = self._changes.handle_command(command)
        except Exception:
            self._record(
                command,
                {},
                AuditPhase.ERROR,
                decision=decision,
                approval=approval,
                success=False,
            )
            raise
        self._record(
            command,
            {},
            AuditPhase.COMPLETED,
            decision=decision,
            approval=approval,
            success=result not in {"Nothing to undo.", "No changes to show."},
        )
        return result

    def _record(
        self,
        tool: str,
        args: dict[str, object],
        phase: AuditPhase,
        *,
        decision: PolicyDecision | None = None,
        approval: ApprovalStatus | None = None,
        success: bool | None = None,
    ) -> None:
        self._audit.record(
            AuditEvent(
                tool=tool,
                permission=decision.permission if decision is not None else None,
                policy_action=decision.action if decision is not None else None,
                approval=approval,
                phase=phase,
                arguments=redact_arguments(args),
                success=success,
            )
        )

    def execute(self, tool: str, args: dict[str, object]) -> ToolResult:
        """Apply policy and approval before dispatching validated arguments."""
        registered = self._tools.get(tool)
        if registered is None:
            self._record(tool, args, AuditPhase.REJECTED, success=False)
            available = ", ".join(self._tools) or "none"
            return ToolResult(
                success=False,
                output=f"Unknown tool {tool!r}. Available tools: {available}.",
            )
        try:
            inputs = registered.input_schema.model_validate(args)
        except ValidationError as exc:
            decision = self._policy.decide(tool)
            self._record(tool, args, AuditPhase.REJECTED, decision=decision, success=False)
            details = exc.errors(
                include_url=False,
                include_input=False,
                include_context=False,
            )
            serialized = json.dumps(details, separators=(",", ":"))
            return ToolResult(
                success=False,
                output=f"Invalid arguments for {tool!r}: {serialized}",
            )

        normalized = cast("dict[str, JsonValue]", inputs.model_dump(mode="json"))
        decision = self._policy.decide(tool)
        self._record(tool, normalized, AuditPhase.REQUESTED, decision=decision)
        try:
            approval = authorize(decision, normalized, self._approvals)
        except ApprovalError:
            self._record(tool, normalized, AuditPhase.ERROR, decision=decision, success=False)
            return ToolResult(success=False, output="User approval could not be obtained.")
        if approval is ApprovalStatus.DENIED:
            self._record(
                tool,
                normalized,
                AuditPhase.DENIED,
                decision=decision,
                approval=approval,
                success=False,
            )
            return ToolResult(success=False, output="User approval was not granted.", denied=True)

        try:
            output = registered.execute(inputs)
        except ToolError as exc:
            result = ToolResult(success=False, output=str(exc))
        except Exception:
            self._record(
                tool,
                normalized,
                AuditPhase.ERROR,
                decision=decision,
                approval=approval,
                success=False,
            )
            raise
        else:
            final = getattr(registered, "ends_turn", False)
            result = ToolResult(success=True, output=output, final=final)
        self._record(
            tool,
            normalized,
            AuditPhase.COMPLETED,
            decision=decision,
            approval=approval,
            success=result.success,
        )
        if result.success and self._changes is not None:
            self._changes.observe(tool, normalized, result.output)
        return result


def create_read_only_registry(root: Path) -> ToolRegistry:
    """Register the four filesystem tools for one workspace root."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root))
    registry.register(ListDirectoryTool(root))
    registry.register(FindFilesTool(root))
    registry.register(SearchFilesTool(root))
    return registry


def create_workspace_registry(
    root: Path,
    *,
    approvals: ApprovalProvider | None = None,
    audit: AuditSink | None = None,
) -> ToolRegistry:
    """Register read and approved editing tools for one tracked workspace."""
    changes = WorkspaceChanges(root)
    registry = ToolRegistry(approvals=approvals, audit=audit, changes=changes)
    registry.register(ReadFileTool(root))
    registry.register(ListDirectoryTool(root))
    registry.register(FindFilesTool(root))
    registry.register(SearchFilesTool(root))
    registry.register(WriteFileTool(changes))
    registry.register(EditFileTool(changes))
    registry.register(MoveFileTool(changes))
    registry.register(CreateDirectoryTool(changes))
    registry.register(TrashFileTool(changes))
    return registry


def create_macos_registry(
    root: Path,
    *,
    approvals: ApprovalProvider | None = None,
    audit: AuditSink | None = None,
    runner: CommandRunner | None = None,
) -> ToolRegistry:
    """Register workspace and approved macOS tools for one agent session."""
    registry = create_workspace_registry(root, approvals=approvals, audit=audit)
    registry.register(OpenApplicationTool(runner))
    registry.register(RevealInFinderTool(root, runner))
    registry.register(ClipboardReadTool(runner))
    registry.register(ClipboardWriteTool(runner))
    registry.register(RunShortcutTool(runner))
    registry.register(RunAppleScriptTool(runner))
    return registry


def register_web_tools(
    registry: ToolRegistry, client: WebClient, backend: LLMBackend | None = None
) -> ToolRegistry:
    """Add web search and page reading; with a model backend, also cited research."""
    if backend is not None:
        from assistant.research import ResearchTool  # research imports the web tools

        # Listed before web_search so a small model reads it first.
        registry.register(ResearchTool(backend, client))
    registry.register(WebSearchTool(client))
    registry.register(FetchUrlTool(client))
    registry.register(ExtractPageTool(client))
    return registry


def create_web_registry(
    *,
    approvals: ApprovalProvider | None = None,
    audit: AuditSink | None = None,
    client: WebClient | None = None,
) -> ToolRegistry:
    """Register only the three web research tools."""
    registry = ToolRegistry(approvals=approvals, audit=audit)
    return register_web_tools(registry, client if client is not None else WebClient())


def create_agent_registry(
    root: Path,
    *,
    approvals: ApprovalProvider | None = None,
    audit: AuditSink | None = None,
    runner: CommandRunner | None = None,
    web: WebClient | None = None,
    backend: LLMBackend | None = None,
) -> ToolRegistry:
    """Register workspace, macOS, and web tools; ``backend`` enables the research tool."""
    registry = create_macos_registry(root, approvals=approvals, audit=audit, runner=runner)
    return register_web_tools(registry, web if web is not None else WebClient(), backend)
