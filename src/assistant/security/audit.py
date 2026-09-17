"""Structured audit records for permission decisions and tool execution."""

import hashlib
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from assistant.security.approvals import ApprovalStatus
from assistant.security.policy import PermissionLevel, PolicyAction


class AuditPhase(StrEnum):
    REQUESTED = "requested"
    COMPLETED = "completed"
    DENIED = "denied"
    REJECTED = "rejected"
    ERROR = "error"


class AuditError(RuntimeError):
    """An audit record could not be persisted."""


class AuditEvent(BaseModel):
    """One immutable security event without raw tool arguments or output."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool: str
    permission: PermissionLevel | None
    policy_action: PolicyAction | None
    approval: ApprovalStatus | None
    phase: AuditPhase
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    success: bool | None = None


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


def argument_digest(arguments: dict[str, Any]) -> str:
    """Hash canonical arguments so audit logs do not retain their contents."""
    canonical = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class InMemoryAuditLog:
    """Retain audit events for the lifetime of one registry."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = Lock()

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)


class JsonlAuditLog:
    """Append audit events to a local JSON Lines file and flush each record."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    def record(self, event: AuditEvent) -> None:
        line = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            raise AuditError(f"Cannot append audit record to {self._path}.") from exc
