"""Typed progress events independent of terminal rendering."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from assistant.agent.loop import ToolResult
    from assistant.security.approvals import ApprovalRequest


@dataclass(frozen=True)
class TurnStarted:
    prompt: str


@dataclass(frozen=True)
class AssistantDelta:
    """User-visible answer text only; never reasoning or action JSON."""

    text: str


@dataclass(frozen=True)
class AssistantFinished:
    text: str
    model: str | None = None
    finish_reason: str | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ToolStarted:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolFinished:
    tool: str
    result: ToolResult
    duration: float


@dataclass(frozen=True)
class ToolFailed:
    tool: str
    error: str
    duration: float = 0.0


@dataclass(frozen=True)
class ApprovalRequested:
    request: ApprovalRequest


@dataclass(frozen=True)
class TurnFailed:
    error: str


type AgentEvent = (
    TurnStarted
    | AssistantDelta
    | AssistantFinished
    | ToolStarted
    | ToolFinished
    | ToolFailed
    | ApprovalRequested
    | TurnFailed
)
type EventSink = Callable[[AgentEvent], None]


def discard_event(event: AgentEvent) -> None:
    """Default sink for callers that only need the return value."""


_sink: ContextVar[EventSink] = ContextVar("jamie_event_sink", default=discard_event)


def emit_event(event: AgentEvent) -> None:
    """Report an event to the current agent turn, if one is active."""
    _sink.get()(event)


@contextmanager
def event_scope(sink: EventSink) -> Iterator[None]:
    """Bind nested tool and approval events to one turn without changing policy."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
