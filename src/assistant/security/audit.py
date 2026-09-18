"""Structured audit records for permission decisions and tool execution."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from assistant.security.approvals import ApprovalStatus
from assistant.security.policy import PermissionLevel, PolicyAction

MAX_ARGUMENT_CHARACTERS = 500


class AuditPhase(StrEnum):
    REQUESTED = "requested"
    COMPLETED = "completed"
    DENIED = "denied"
    REJECTED = "rejected"
    ERROR = "error"


class AuditError(RuntimeError):
    """An audit record could not be persisted."""


class AuditEvent(BaseModel):
    """One immutable security event with redacted arguments and no tool output."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool: str
    permission: PermissionLevel | None
    policy_action: PolicyAction | None
    approval: ApprovalStatus | None
    phase: AuditPhase
    arguments: dict[str, JsonValue]
    success: bool | None = None


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


def _redact(value: Any) -> JsonValue:
    if not isinstance(value, str | int | float | bool | None):
        value = str(value)
    if isinstance(value, str) and len(value) > MAX_ARGUMENT_CHARACTERS:
        return f"{value[:MAX_ARGUMENT_CHARACTERS]}[truncated]"
    return value


def redact_arguments(arguments: dict[str, Any]) -> dict[str, JsonValue]:
    """Keep argument values readable but bound how much text one event retains."""
    return {key: _redact(value) for key, value in arguments.items()}


class InMemoryAuditLog:
    """Retain audit events for the lifetime of one registry."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)


class JsonlAuditLog:
    """Append audit events to a local JSON Lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, event: AuditEvent) -> None:
        line = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with self._path.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
        except OSError as exc:
            raise AuditError(f"Cannot append audit record to {self._path}.") from exc
