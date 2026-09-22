"""Reusable lifecycle for Jamie's direct-chat application runtime."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from assistant.agent.events import (
    AssistantFinished,
    EventSink,
    TurnFailed,
    TurnStarted,
    discard_event,
)
from assistant.config import ProjectConfig
from assistant.llm.answer import answer_text
from assistant.llm.base import LLMBackend, LLMResponse, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import MemoryStore, MemoryStoreError, format_memory_context

BackendFactory = Callable[[], LLMBackend]
MemoryStoreFactory = Callable[[Path], MemoryStore]


class JamieApplicationError(RuntimeError):
    """The shared application lifecycle was used in an invalid state."""


@dataclass(frozen=True)
class TurnResult:
    """The complete result of one direct-chat turn."""

    message: str
    model: str | None = None
    finish_reason: str | None = None
    output_tokens: int | None = None

    @classmethod
    def from_response(cls, response: LLMResponse) -> TurnResult:
        return cls(
            message=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
            output_tokens=response.output_tokens,
        )


async def _close_backend(backend: LLMBackend) -> None:
    closer = getattr(backend, "aclose", None)
    if closer is None:
        closer = getattr(backend, "close", None)
    if closer is None:
        return
    result = closer()
    if inspect.isawaitable(result):
        await result


class JamieApplication:
    """Own one reusable backend and the services needed for direct chat turns."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        backend_factory: BackendFactory | None = None,
        memory_store_factory: MemoryStoreFactory = MemoryStore,
    ) -> None:
        self._config = config
        self._backend_factory = backend_factory or LlamaCppBackend
        self._memory_store_factory = memory_store_factory
        self._backend: LLMBackend | None = None
        self._memory: MemoryStore | None = None
        self._closed = False

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize this application's backend and memory store once."""
        if self._closed:
            raise JamieApplicationError("The Jamie application is closed.")
        if self._backend is not None:
            return

        backend = self._backend_factory()
        try:
            memory = self._memory_store_factory(self._config.resolved_memory_path())
        except BaseException:
            await _close_backend(backend)
            raise
        self._backend = backend
        self._memory = memory

    async def run_turn(
        self,
        prompt: str,
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        event_sink: EventSink = discard_event,
    ) -> TurnResult:
        """Run one direct-chat turn with the initialized application services."""
        if self._closed:
            raise JamieApplicationError("The Jamie application is closed.")
        if self._backend is None or self._memory is None:
            raise JamieApplicationError("Start the Jamie application before running a turn.")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        event_sink(TurnStarted(prompt))
        session_id = None
        try:
            session_id = self._memory.start_session("ask")
            self._memory.add_message(session_id, "user", prompt)
            memory_context = format_memory_context(self._memory.search_memories(prompt))
            messages = []
            if memory_context is not None:
                messages.append(Message(role="system", content=memory_context))
            messages.append(Message(role="user", content=prompt))
            response = await asyncio.to_thread(
                self._backend.chat,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = answer_text(response.content)
            self._memory.add_message(session_id, "assistant", text)
            self._memory.remember_exchange(session_id, prompt, text)
            self._memory.finish_session(session_id, success=True)
        except BaseException as exc:
            if session_id is not None:
                self._finish_failed_session(session_id)
            event_sink(TurnFailed(str(exc) or type(exc).__name__))
            raise
        result = TurnResult(text, response.model, response.finish_reason, response.output_tokens)
        event_sink(
            AssistantFinished(
                result.message, result.model, result.finish_reason, result.output_tokens
            )
        )
        return result

    async def close(self) -> None:
        """Close owned resources once and reject future work."""
        if self._closed:
            return
        self._closed = True
        backend = self._backend
        self._backend = None
        self._memory = None
        if backend is not None:
            await _close_backend(backend)

    def _finish_failed_session(self, session_id: str) -> None:
        if self._memory is None:
            return
        try:
            self._memory.finish_session(session_id, success=False)
        except MemoryStoreError:
            pass
