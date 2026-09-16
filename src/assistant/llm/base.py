"""Shared types for text chat backends."""

from dataclasses import dataclass
from typing import Literal, Protocol


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


class LLMError(RuntimeError):
    """A backend request failed or returned an invalid response."""


class LLMBackend(Protocol):
    """A backend that produces a single text response to chat messages."""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
