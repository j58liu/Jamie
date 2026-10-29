"""Reusable lifecycle for Jamie's direct chat and agent turns."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter
from typing import Self

from assistant.agent.events import (
    AssistantDelta,
    AssistantFinished,
    EventSink,
    RequestContext,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnStarted,
    discard_event,
)
from assistant.agent.loop import (
    MAX_STEPS,
    TIMEOUT_SECONDS,
    ToolExecutor,
    _system_instructions,
    run_agent,
)
from assistant.cancellation import CancellationToken, cancellation_scope
from assistant.config import ProjectConfig
from assistant.llm.answer import answer_text
from assistant.llm.base import ContextUsage, LLMBackend, LLMResponse, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import (
    ConversationSummary,
    MemoryStore,
    MemoryStoreError,
    RememberingTools,
    format_memory_context,
)
from assistant.runtime.context import RequestBudget
from assistant.tools.scopes import READ_TOOLS

BackendFactory = Callable[[], LLMBackend]
MemoryStoreFactory = Callable[[Path], MemoryStore]

COMPACT_REQUEST = (
    "Summarize our conversation so far so we can continue from the summary alone. Keep "
    "facts, names, numbers, file paths, decisions, and open questions. Write plain "
    "sentences, at most 200 words."
)
COMPACT_MAX_TOKENS = 512


class JamieApplicationError(RuntimeError):
    """The shared application lifecycle was used in an invalid state."""


@dataclass(frozen=True)
class TurnResult:
    """The complete result of one application turn."""

    message: str
    model: str | None = None
    finish_reason: str | None = None
    output_tokens: int | None = None
    first_token_seconds: float | None = None
    compacted_history: tuple[Message, ...] | None = None
    context: ContextUsage | None = None

    @classmethod
    def from_response(cls, response: LLMResponse) -> TurnResult:
        return cls(
            message=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
            output_tokens=response.output_tokens,
            first_token_seconds=response.first_token_seconds,
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


class _AgentTurnBackend:
    """Apply interactive generation settings and collect usage across agent steps."""

    def __init__(
        self,
        backend: LLMBackend,
        application: JamieApplication,
        temperature: float,
        max_tokens: int,
        thinking: bool | None,
        history_count: int = 0,
    ) -> None:
        self._backend = backend
        self._application = application
        self._options = {"temperature": temperature, "max_tokens": max_tokens}
        if thinking is not None:
            self._options["thinking"] = thinking
        self.last_response: LLMResponse | None = None
        self.output_tokens = 0
        self.first_token_seconds: float | None = None
        self.history_count = history_count
        self.compacted_history: tuple[Message, ...] | None = None
        self.context: ContextUsage | None = None

    def chat(self, messages: list[Message]) -> LLMResponse:
        prepared = self._application._budget.prepare(
            messages,
            reserve=self._options["max_tokens"],
            history_count=self.history_count,
            thinking=self._options.get("thinking"),
        )
        if prepared.history is not None:
            self.compacted_history = tuple(prepared.history)
            self.history_count = len(prepared.history)
        self.context = prepared.usage
        self._application.model_calls += 1
        response = self._backend.chat(messages, **self._options)
        self.last_response = response
        self.output_tokens += response.output_tokens or 0
        if self.first_token_seconds is None:
            self.first_token_seconds = response.first_token_seconds
        return response


class JamieApplication:
    """Own one reusable backend and the services needed for turns."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        backend_factory: BackendFactory | None = None,
        memory_store_factory: MemoryStoreFactory = MemoryStore,
        extract_memories: bool = False,
    ) -> None:
        self._config = config
        self._backend_factory = backend_factory or LlamaCppBackend
        self._memory_store_factory = memory_store_factory
        self._backend: LLMBackend | None = None
        self._memory: MemoryStore | None = None
        self._closed = False
        self.model_calls = 0  # requests sent to the model, including failed ones
        self._budget: RequestBudget | None = None
        self._last_context: ContextUsage | None = None
        self._measured_history: list[Message] | None = None
        self._last_tool_names: tuple[str, ...] = ()
        self._connectors = None
        self._extract_memories = extract_memories
        self._personal = None
        self.last_extraction = None  # the most recent memory extraction outcome
        self.last_plan = None  # the most recent plan outcome, when the planner ran

    @property
    def personal(self):
        """Typed personal memory beside the conversation store, when that store is SQLite."""
        if self._personal is None and isinstance(self._memory, MemoryStore):
            from assistant.knowledge import PersonalMemory, PersonalMemoryError

            try:
                self._personal = PersonalMemory(self._memory.path)
            except PersonalMemoryError:
                return None
        return self._personal

    def _memory_context(self, memory, prompt: str, conversation_id: str | None) -> str | None:
        """Relevant personal facts plus excerpts from earlier conversations."""
        parts = []
        personal = self.personal
        if personal is not None:
            from assistant.knowledge.retrieval import format_personal_context, recall

            budget = self._config.memory.personal_context_tokens
            facts = format_personal_context(recall(personal, prompt), budget)
            if facts:
                parts.append(facts)
        recalled = memory.search_memories(prompt, exclude_session=conversation_id)
        excerpts = format_memory_context(recalled)
        if excerpts:
            parts.append(excerpts)
        return "\n\n".join(parts) or None

    def _extract(self, prompt: str, evidence_ref: str | None) -> None:
        """Learn durable facts from the user's own words; never fails the turn."""
        personal = self.personal
        if not self._extract_memories or personal is None or self._backend is None:
            return
        from assistant.knowledge.extraction import MemoryExtractor, likely_personal

        if not likely_personal(prompt):
            return

        def commitment(text: str, evidence: str) -> None:
            from assistant.tasks.models import TaskState
            from assistant.tasks.store import TaskStore

            store = TaskStore(self._config.resolved_tasks_path())
            store.create_task(f"Commitment: {text[:80]}", text, state=TaskState.SCHEDULED)

        try:
            self.model_calls += 1
            self.last_extraction = MemoryExtractor(
                self._backend, personal, on_commitment=commitment
            ).process(prompt, evidence_ref=evidence_ref, force=True)
        except Exception:  # memory is best effort; the answer already succeeded
            self.last_extraction = None

    @property
    def config(self) -> ProjectConfig:
        return self._config

    @property
    def connectors(self):
        """Personal-application connectors from configuration, created once on first use.

        Creating them reads configuration only; no provider is contacted.
        """
        if self._connectors is None:
            from assistant.connectors import build_connectors

            self._connectors = build_connectors(self._config)
        return self._connectors

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
        self._budget = RequestBudget(backend, self._summary_started)

    def _summary_started(self) -> None:
        self.model_calls += 1

    async def run_turn(
        self,
        prompt: str,
        *,
        history: list[Message] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        cancel: CancellationToken | None = None,
        conversation_id: str | None = None,
        event_sink: EventSink = discard_event,
        tools: ToolExecutor | None = None,
        max_steps: int = MAX_STEPS,
        timeout_seconds: float = TIMEOUT_SECONDS,
        plan: str = "never",
    ) -> TurnResult:
        """Run one chat or tool-using turn with the initialized services.

        ``thinking`` is forwarded only when set, so backends without it keep working.
        Passing ``tools`` enables the agent action protocol and Python permission policy;
        ``max_steps`` and ``timeout_seconds`` then bound the agent. ``plan`` is "never",
        "auto" (multi-step requests use the planner), or "always".
        A cancelled turn raises TurnCancelled once the model request has stopped, and
        its answer is never stored. With ``conversation_id`` the exchange is appended
        to that resumable conversation only after it succeeds; otherwise each turn is
        its own session, as for ``jamie ask``.
        """
        memory = self._require_memory()
        if self._backend is None:
            raise JamieApplicationError("Start the Jamie application before running a turn.")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if tools is not None:
            return await self._run_agent_turn(
                prompt,
                tools=tools,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
                cancel=cancel,
                conversation_id=conversation_id,
                event_sink=event_sink,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                plan=plan,
            )

        event_sink(TurnStarted(prompt))
        session_id = None
        try:
            if conversation_id is None:
                session_id = memory.start_session("ask")
                memory.add_message(session_id, "user", prompt)
            # The conversation's own exchanges are already in the history.
            memory_context = self._memory_context(memory, prompt, conversation_id)
            messages = []
            if memory_context is not None:
                messages.append(Message(role="system", content=memory_context))
            messages.extend(history or [])
            messages.append(Message(role="user", content=prompt))
            options = {} if thinking is None else {"thinking": thinking}
            with cancellation_scope(cancel):
                prepared = await asyncio.to_thread(
                    self._budget.prepare,
                    messages,
                    reserve=max_tokens,
                    history_count=len(history or []),
                    thinking=thinking,
                )
                self.model_calls += 1
                response = await self._chat(
                    messages, event_sink, temperature=temperature, max_tokens=max_tokens, **options
                )
            if cancel is not None:
                cancel.raise_if_cancelled()
            text = answer_text(response.content)
            if conversation_id is None:
                memory.add_message(session_id, "assistant", text)
                memory.remember_exchange(session_id, prompt, text)
                memory.finish_session(session_id, success=True)
            else:
                memory.save_exchange(conversation_id, prompt, text, str(Path.cwd()))
                memory.remember_exchange(conversation_id, prompt, text)
            await asyncio.to_thread(self._extract, prompt, session_id or conversation_id)
        except BaseException as exc:
            if session_id is not None:
                self._finish_failed_session(session_id)
            event_sink(TurnFailed(str(exc) or type(exc).__name__))
            raise
        compacted = tuple(prepared.history) if prepared.history is not None else None
        result = replace(
            TurnResult.from_response(response),
            message=text,
            compacted_history=compacted,
            context=prepared.usage,
        )
        self._remember_context(result, prompt, history)
        event_sink(
            AssistantFinished(
                result.message, result.model, result.finish_reason, result.output_tokens
            )
        )
        return result

    async def _chat(self, messages: list[Message], sink: EventSink, **options) -> LLMResponse:
        """Drain worker deltas on the caller's event loop, preserving event order."""
        stream = getattr(self._backend, "stream_chat", None)
        if stream is None:
            return await asyncio.to_thread(self._backend.chat, messages, **options)
        pending: Queue[str] = Queue()
        task = asyncio.create_task(
            asyncio.to_thread(stream, messages, on_delta=pending.put, **options)
        )

        def drain():
            while True:
                try:
                    text = pending.get_nowait()
                except Empty:
                    return
                sink(AssistantDelta(text))

        try:
            while not task.done():
                drain()
                await asyncio.wait({task}, timeout=0.01)
        finally:
            drain()
        return await task

    async def _run_agent_turn(
        self,
        prompt: str,
        *,
        tools: ToolExecutor,
        history: list[Message] | None,
        temperature: float,
        max_tokens: int,
        thinking: bool | None,
        cancel: CancellationToken | None,
        conversation_id: str | None,
        event_sink: EventSink,
        max_steps: int,
        timeout_seconds: float,
        plan: str = "never",
    ) -> TurnResult:
        memory = self._require_memory()
        session_id = conversation_id
        standalone = conversation_id is None
        request_context = None

        def observe(event):
            nonlocal request_context
            if isinstance(event, RequestContext):
                request_context = event.usage
            event_sink(event)

        try:
            if standalone:
                session_id = memory.start_session("agent")
                memory.add_message(session_id, "user", prompt)
            context = self._memory_context(memory, prompt, conversation_id)
            backend = _AgentTurnBackend(
                self._backend, self, temperature, max_tokens, thinking, len(history or [])
            )
            recorded = RememberingTools(tools, memory, session_id)
            answer = None
            self.last_plan = None
            from assistant.planning import PlanningError, needs_plan, run_plan

            if plan == "always" or (plan == "auto" and needs_plan(prompt)):
                try:
                    outcome = await asyncio.to_thread(
                        run_plan, backend, recorded, prompt, memory_context=context,
                        event_sink=observe, cancel=cancel,
                    )  # fmt: skip
                except PlanningError:
                    outcome = None  # no usable plan; the agent loop handles the request
                if outcome is not None:
                    self.last_plan = outcome
                    answer = outcome.answer
            if answer is None:
                answer = await asyncio.to_thread(
                    run_agent,
                    backend,
                    recorded,
                    prompt,
                    history=history,
                    memory_context=context,
                    event_sink=observe,
                    cancel=cancel,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                )
            if cancel is not None:
                cancel.raise_if_cancelled()
            if standalone:
                memory.add_message(session_id, "assistant", answer)
                memory.remember_exchange(session_id, prompt, answer)
                memory.finish_session(session_id, success=True)
            else:
                memory.save_exchange(session_id, prompt, answer, str(Path.cwd()))
                memory.remember_exchange(session_id, prompt, answer)
            await asyncio.to_thread(self._extract, prompt, session_id)
        except BaseException:
            if standalone and session_id is not None:
                self._finish_failed_session(session_id)
            raise
        last = backend.last_response
        result = TurnResult(
            answer,
            model=last.model if last else None,
            finish_reason=last.finish_reason if last else None,
            output_tokens=backend.output_tokens,
            first_token_seconds=backend.first_token_seconds,
            compacted_history=backend.compacted_history,
            context=request_context or backend.context,
        )
        self._remember_context(result, prompt, history, tools)
        return result

    def _remember_context(self, result: TurnResult, prompt: str, history, tools=None) -> None:
        self._last_context = result.context
        self._last_tool_names = self._tool_names(tools)
        self._measured_history = [
            *(result.compacted_history if result.compacted_history is not None else history or []),
            Message("user", prompt),
            Message("assistant", result.message),
        ]

    @staticmethod
    def _tool_names(tools) -> tuple[str, ...]:
        definitions = getattr(tools, "definitions", None)
        return tuple(d["name"] for d in definitions()) if definitions is not None else ()

    async def run_local_tool(
        self,
        tool: str,
        args: dict,
        *,
        tools: ToolExecutor,
        cancel: CancellationToken | None = None,
        event_sink: EventSink = discard_event,
    ):
        """Run an explicit local read through the same validation, policy, and audit."""
        self._require_memory()
        if tool not in READ_TOOLS:
            raise ValueError("Local commands support workspace reads only.")
        event_sink(TurnStarted(tool))
        event_sink(ToolStarted(tool, deepcopy(args)))
        started = perf_counter()
        try:
            with cancellation_scope(cancel):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                result = await asyncio.to_thread(tools.execute, tool, args)
                if cancel is not None:
                    cancel.raise_if_cancelled()
            duration = perf_counter() - started
            if result.success:
                event_sink(ToolFinished(tool, result, duration))
            else:
                event_sink(ToolFailed(tool, result.output, duration))
            event_sink(AssistantFinished(result.output))
            return result
        except BaseException as exc:
            event_sink(TurnFailed(str(exc)))
            raise

    @property
    def memory(self) -> MemoryStore:
        """The started application's store, for commands and scheduled tasks."""
        return self._require_memory()

    def start_conversation(self) -> str:
        return self._require_memory().start_conversation(str(Path.cwd()))

    def recent_conversations(self, limit: int = 10) -> list[ConversationSummary]:
        return self._require_memory().recent_conversations(limit=limit)

    def find_conversation(self, id_prefix: str) -> ConversationSummary | None:
        return self._require_memory().find_conversation(id_prefix)

    def conversation_messages(self, conversation_id: str) -> list[Message]:
        return self._require_memory().conversation_messages(conversation_id)

    async def compact(
        self, history: list[Message], *, cancel: CancellationToken | None = None
    ) -> list[Message]:
        """Summarize the history into one exchange that later turns continue from.

        Only the history sent to the model shrinks; the saved conversation keeps every
        exchange. A cancelled request raises TurnCancelled and leaves the history unchanged.
        """
        if self._backend is None:
            raise JamieApplicationError("Start the Jamie application before compacting.")
        if not history:
            raise ValueError("There is no conversation to compact.")
        self.model_calls += 1
        with cancellation_scope(cancel):
            messages = [*history, Message(role="user", content=COMPACT_REQUEST)]
            await asyncio.to_thread(
                self._budget.prepare,
                messages,
                reserve=COMPACT_MAX_TOKENS,
                history_count=len(history),
            )
            response = await asyncio.to_thread(
                self._backend.chat,
                messages,
                temperature=0.2,
                max_tokens=COMPACT_MAX_TOKENS,
            )
        if cancel is not None:
            cancel.raise_if_cancelled()
        summary = answer_text(response.content)
        return [
            Message(role="user", content=COMPACT_REQUEST),
            Message(role="assistant", content=summary),
        ]

    def context_usage(
        self,
        history: list[Message],
        *,
        tools: ToolExecutor | None = None,
        max_tokens: int | None = None,
    ) -> ContextUsage | None:
        """Last complete request, or the conversation's base cost before the next prompt."""
        if (
            history == self._measured_history
            and self._last_context is not None
            and self._tool_names(tools) == self._last_tool_names
            and (max_tokens is None or max_tokens == self._last_context.output_reserved)
        ):
            return self._last_context
        messages = list(history)
        if tools is not None:
            messages.insert(0, Message("system", _system_instructions(tools)))
        return (
            self._budget.measure(messages, max_tokens or 1024) if self._budget is not None else None
        )

    def _require_memory(self) -> MemoryStore:
        if self._closed:
            raise JamieApplicationError("The Jamie application is closed.")
        if self._memory is None:
            raise JamieApplicationError("Start the Jamie application before running a turn.")
        return self._memory

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
