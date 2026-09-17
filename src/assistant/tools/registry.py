"""Registration, schemas, and execution for agent tools."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from assistant.agent.loop import ToolResult
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

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

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

    def execute(self, tool: str, args: dict[str, object]) -> ToolResult:
        """Validate arguments and convert expected tool errors into observations."""
        registered = self._tools.get(tool)
        if registered is None:
            available = ", ".join(self._tools) or "none"
            return ToolResult(
                success=False,
                output=f"Unknown tool {tool!r}. Available tools: {available}.",
            )
        try:
            inputs = registered.input_schema.model_validate(args)
        except ValidationError as exc:
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
        try:
            output = registered.execute(inputs)
        except ToolError as exc:
            return ToolResult(success=False, output=str(exc))
        if not isinstance(output, str):
            raise TypeError(f"Tool {tool!r} must return a string.")
        return ToolResult(success=True, output=output)


def create_read_only_registry(root: Path) -> ToolRegistry:
    """Register the four filesystem tools for one workspace root."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root))
    registry.register(ListDirectoryTool(root))
    registry.register(FindFilesTool(root))
    registry.register(SearchFilesTool(root))
    return registry
