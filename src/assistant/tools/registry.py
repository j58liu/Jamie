"""Registration, schemas, and execution for agent tools."""

import json
import re
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue, ValidationError

from assistant.agent.loop import ToolResult
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
    argument_digest,
)
from assistant.security.policy import PermissionPolicy, PolicyDecision
from assistant.tools.base import Risk, Tool, ToolError
from assistant.tools.filesystem import (
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
)

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ToolRegistry:
    """Validate and dispatch named tools for the agent loop."""

    def __init__(
        self,
        *,
        policy: PermissionPolicy | None = None,
        approvals: ApprovalProvider | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._policy = policy if policy is not None else PermissionPolicy()
        self._approvals = approvals if approvals is not None else DenyAllApprovals()
        self._audit = audit if audit is not None else InMemoryAuditLog()

    @property
    def audit(self) -> AuditSink:
        return self._audit

    def register(self, tool: Tool) -> None:
        if not _TOOL_NAME.fullmatch(tool.name):
            raise ValueError("Tool names must use lowercase letters, numbers, and underscores.")
        if not tool.description.strip():
            raise ValueError("Tool descriptions must not be blank.")
        if not isinstance(tool.risk, Risk):
            raise ValueError("Tool risk must be a Risk value.")
        if not isinstance(tool.input_schema, type) or not issubclass(tool.input_schema, BaseModel):
            raise ValueError("Tool input_schema must be a Pydantic model class.")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
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
        definitions = json.dumps(self.definitions(), ensure_ascii=False, separators=(",", ":"))
        return (
            f"Available tools and their argument schemas:\n{definitions}\n"
            "The schema properties belong inside the action's args object. Examples:\n"
            '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}\n'
            '{"action":"tool","tool":"list_directory","args":{}}\n'
            '{"action":"tool","tool":"find_files","args":{"pattern":"*.md"}}\n'
            '{"action":"tool","tool":"search_files","args":{"query":"text"}}'
        )

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
                arguments_sha256=argument_digest(args),
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
            return ToolResult(success=False, output="User approval was not granted.")

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
            if not isinstance(output, str):
                self._record(
                    tool,
                    normalized,
                    AuditPhase.ERROR,
                    decision=decision,
                    approval=approval,
                    success=False,
                )
                raise TypeError(f"Tool {tool!r} must return a string.")
            result = ToolResult(success=True, output=output)
        self._record(
            tool,
            normalized,
            AuditPhase.COMPLETED,
            decision=decision,
            approval=approval,
            success=result.success,
        )
        return result


def create_read_only_registry(root: Path) -> ToolRegistry:
    """Register the four filesystem tools for one workspace root."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root))
    registry.register(ListDirectoryTool(root))
    registry.register(FindFilesTool(root))
    registry.register(SearchFilesTool(root))
    return registry
