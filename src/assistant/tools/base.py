"""Shared contracts for tools exposed to the agent."""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class Risk(StrEnum):
    """The effect a tool may have outside the agent process."""

    READ_ONLY = "read_only"


class ToolError(ValueError):
    """A tool could not safely complete the requested operation."""


class Tool(Protocol):
    """A validated operation that can be registered for agent use."""

    name: str
    description: str
    input_schema: type[BaseModel]
    risk: Risk

    def execute(self, arguments: BaseModel) -> str: ...
