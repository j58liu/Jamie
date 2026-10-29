"""Connector contracts shared by every provider integration.

A connector makes provider API calls and returns small typed models. Connector tools
adapt one operation to Jamie's Tool protocol; the registry validates arguments and the
security policy decides whether the call may run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from assistant.security.policy import PermissionLevel


class ConnectionStatus(StrEnum):
    CONNECTED = "connected"
    NOT_CONNECTED = "not_connected"
    EXPIRED = "expired"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    UNAVAILABLE = "unavailable"


class Capability(StrEnum):
    READ = "read"
    DRAFT = "draft"
    SEND = "send"
    WRITE = "write"
    FREE_BUSY = "free_busy"


@dataclass(frozen=True)
class ConnectorInfo:
    """What a connector is, whether it can run now, and which operations it offers."""

    name: str
    status: ConnectionStatus
    capabilities: frozenset[Capability] = frozenset()
    operations: tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "capabilities": sorted(c.value for c in self.capabilities),
            "operations": list(self.operations),
            **({"detail": self.detail} if self.detail else {}),
        }


class Connector(Protocol):
    name: str

    def info(self) -> ConnectorInfo: ...


class ConnectorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def to_json(value: BaseModel | Sequence[BaseModel] | dict | list) -> str:
    """Serialize results compactly; empty fields are dropped to save model context."""
    if isinstance(value, BaseModel):
        data: object = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, (list, tuple)):
        data = [
            item.model_dump(mode="json", exclude_none=True) if isinstance(item, BaseModel) else item
            for item in value
        ]
    else:
        data = value
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class ConnectorTool:
    """Base for tools that expose one connector operation.

    Subclasses set ``name``, ``description``, ``input_schema`` and ``risk`` and implement
    ``run``. Connector exceptions are ToolErrors, so the registry already reports them.
    """

    name: str
    description: str
    input_schema: type[BaseModel]
    risk: PermissionLevel

    def run(self, arguments): ...  # pragma: no cover - overridden

    def execute(self, arguments: BaseModel) -> str:
        return to_json(self.run(arguments))
