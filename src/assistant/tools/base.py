"""Shared contracts for tools exposed to the agent."""

from typing import Protocol

from pydantic import BaseModel

from assistant.security.policy import PermissionLevel

Risk = PermissionLevel


class ToolError(ValueError):
    """A tool could not safely complete the requested operation."""


class Tool(Protocol):
    """A validated operation that can be registered for agent use."""

    name: str
    description: str
    input_schema: type[BaseModel]
    risk: PermissionLevel

    def execute(self, arguments: BaseModel) -> str: ...
