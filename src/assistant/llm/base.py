"""Shared types for text chat backends."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Message:
    """One text message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Generated text and the server's completion metadata."""

    content: str
    model: str | None = None
    finish_reason: str | None = None
    output_tokens: int | None = None
    first_token_seconds: float | None = None  # known only for streamed requests
    input_tokens: int | None = None


@dataclass(frozen=True)
class ServerInfo:
    """What the running inference server reports about its loaded model."""

    model: str
    model_path: str | None
    context: int


@dataclass(frozen=True)
class ContextUsage:
    """Tokens the conversation occupies in the server's context window."""

    used: int
    maximum: int
    output_reserved: int = 0

    @property
    def available(self) -> int:
        return max(self.maximum - self.used - self.output_reserved, 0)

    @property
    def percent(self) -> int:
        return round(100 * self.used / self.maximum)


class LLMError(RuntimeError):
    """A backend request failed or returned an invalid response."""

    def __init__(self, message: str, *, response: LLMResponse | None = None) -> None:
        super().__init__(message)
        self.response = response


class LLMBackend(Protocol):
    """A backend that produces a single text response to chat messages."""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        thinking: bool | None = None,
    ) -> LLMResponse: ...


_response_schema: ContextVar[dict[str, Any] | None] = ContextVar("jamie_schema", default=None)


def current_response_schema() -> dict[str, Any] | None:
    """The JSON schema replies must follow, if the caller set one; see ``constrained_to``."""
    return _response_schema.get()


@contextmanager
def constrained_to(schema: dict[str, Any] | None) -> Iterator[None]:
    """Ask backends that can constrain generation to reply with JSON matching ``schema``.

    Like the cancellation token, this reaches the backend through a context variable, so
    callers and test backends keep the plain ``chat`` signature. Keep the scope around
    one request: model calls made inside tools must not inherit it.
    """
    token = _response_schema.set(schema)
    try:
        yield
    finally:
        _response_schema.reset(token)


@contextmanager
def managed_backend(backend: LLMBackend) -> Iterator[LLMBackend]:
    """Close an owned backend, including adapters that do not require cleanup."""
    try:
        yield backend
    finally:
        closer = getattr(backend, "close", None)
        if closer is not None:
            closer()
