"""Run queued background tasks a few at a time while the user keeps working."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from assistant.agent.events import AgentEvent, ToolFailed, ToolStarted
from assistant.background.approvals import DeferredApprovalProvider, Grant
from assistant.background.store import (
    CANCELLED,
    NEEDS_INPUT,
    BackgroundStore,
    BackgroundTask,
    PendingApproval,
)
from assistant.cancellation import CancellationToken, TurnCancelled

POLL_SECONDS = 2.0
DEFAULT_PARALLEL = 2  # the launcher starts llama-server with --parallel 2
# A background task may wait a long time for the user's approval.
BACKGROUND_TIMEOUT_SECONDS = 25 * 60 * 60
BACKGROUND_MAX_STEPS = 24


class Notifier(Protocol):
    def send(self, title: str, body: str) -> None: ...


class PrintNotifier:
    """For `jamie serve` in a terminal and for logs."""

    def send(self, title: str, body: str) -> None:
        print(f"[Jamie] {title}: {body}", file=sys.stderr, flush=True)


class FanOutNotifier:
    def __init__(self, *notifiers: Notifier | None) -> None:
        self._notifiers = [n for n in notifiers if n is not None]

    def send(self, title: str, body: str) -> None:
        for notifier in self._notifiers:
            with contextlib.suppress(Exception):
                notifier.send(title, body)


def short(text: str, limit: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else f"{line[: limit - 1]}…"


@dataclass
class TaskContext:
    """What a runner needs: approvals, cancellation, progress, and notifications."""

    task: BackgroundTask
    store: BackgroundStore
    cancel: CancellationToken
    notifier: Notifier
    grants: tuple[Grant, ...] = ()
    attachment: Callable[[], str | None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def approvals(self, *, attachment: Callable[[], str | None] | None = None):
        return DeferredApprovalProvider(
            self.store,
            self.task.id,
            notify=self.notify_approval,
            grants=self.grants,
            cancel=self.cancel,
            attachment=attachment or self.attachment,
        )

    def notify_approval(self, approval: PendingApproval) -> None:
        self.notifier.send(
            f"Jamie needs your OK (task {self.task.id})",
            f"{short(approval.summary, 160)} Approve with: jamie bg approve {approval.id}",
        )

    def progress(self, text: str) -> None:
        self.store.set_progress(self.task.id, text)

    def event_sink(self, event: AgentEvent) -> None:
        if isinstance(event, ToolStarted):
            self.progress(f"Running {event.tool}")
        elif isinstance(event, ToolFailed):
            self.progress(f"{event.tool} failed; continuing")


type Runner = Callable[[BackgroundTask, TaskContext], Awaitable[str]]


class TaskNeedsInput(Exception):
    """A task stopped because only the user can answer; retrying continues it."""

    def __init__(self, message: str, questions: list[str] | None = None) -> None:
        super().__init__(message)
        self.questions = list(questions or [])


def acquire_worker_lock(path: Path) -> IO[str] | None:
    """Only one process runs background tasks; the others just queue and approve."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


class BackgroundWorker:
    def __init__(
        self,
        store: BackgroundStore,
        runners: dict[str, Runner],
        *,
        notifier: Notifier | None = None,
        max_parallel: int = DEFAULT_PARALLEL,
        poll_seconds: float = POLL_SECONDS,
        grants_for: Callable[[BackgroundTask], tuple[Grant, ...]] | None = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("Run at least one task at a time.")
        self.store = store
        self._runners = runners
        self._notifier = notifier or PrintNotifier()
        self._max_parallel = max_parallel
        self._poll = poll_seconds
        self._grants_for = grants_for
        self._running: dict[int, tuple[asyncio.Task, CancellationToken]] = {}

    @property
    def running(self) -> list[int]:
        return sorted(self._running)

    def start_ready(self) -> list[asyncio.Task]:
        """Claim queued tasks up to the free slots and start them."""
        started = []
        while len(self._running) < self._max_parallel:
            task = self.store.claim_next(frozenset(self._runners))
            if task is None:
                break
            token = CancellationToken()
            job = asyncio.create_task(self._run(task, token), name=f"jamie-bg-{task.id}")
            self._running[task.id] = (job, token)
            started.append(job)
        return started

    def check_cancellations(self) -> None:
        for task_id, (_, token) in list(self._running.items()):
            if not token.cancelled and self.store.cancel_requested(task_id):
                token.cancel()

    async def run_forever(self) -> None:
        self.store.recover_interrupted()
        try:
            while True:
                try:
                    self.check_cancellations()
                    self.start_ready()
                except Exception as exc:  # noqa: BLE001 - keep serving; the store may recover
                    print(f"Jamie: background tasks unavailable: {exc}", file=sys.stderr)
                await asyncio.sleep(self._poll)
        finally:
            await self.stop()

    async def stop(self) -> None:
        for _, token in self._running.values():
            token.cancel()
        jobs = [job for job, _ in self._running.values()]
        if jobs:
            await asyncio.wait(jobs, timeout=10)

    async def drain(self) -> None:
        """Run until the queue is empty and nothing is running (tests, `jamie bg run`)."""
        while True:
            self.check_cancellations()
            self.start_ready()
            if not self._running:
                return
            jobs = [job for job, _ in self._running.values()]
            await asyncio.wait(jobs, timeout=self._poll, return_when=asyncio.FIRST_COMPLETED)

    async def _run(self, task: BackgroundTask, token: CancellationToken) -> None:
        grants = self._grants_for(task) if self._grants_for is not None else ()
        context = TaskContext(task, self.store, token, self._notifier, grants)
        title = f"Task {task.id}: {short(task.prompt, 40)}"
        try:
            result = await self._runners[task.kind](task, context)
        except (TurnCancelled, asyncio.CancelledError):
            self.store.fail(task.id, "Cancelled.", status=CANCELLED)
            self._notifier.send(title, "Cancelled.")
        except TaskNeedsInput as exc:
            questions = "".join(f"\n- {q}" for q in exc.questions)
            self.store.update_payload(task.id, questions=exc.questions)
            self.store.fail(task.id, f"{exc}{questions}", status=NEEDS_INPUT)
            self._notifier.send(title, f"Needs your answers: {short(str(exc), 120)}")
        except Exception as exc:  # noqa: BLE001 - one task's failure never stops the rest
            message = str(exc) or type(exc).__name__
            self.store.fail(task.id, message)
            self._notifier.send(title, f"Could not finish: {short(message, 200)}")
        else:
            self.store.finish(task.id, result)
            self._notifier.send(title, short(result, 240))
        finally:
            self._running.pop(task.id, None)


def agent_runner(
    application,
    registry_factory: Callable[[Any], Any],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
) -> Runner:
    """Run a prompt with the full agent, asking for approvals without blocking the user."""

    async def run(task: BackgroundTask, context: TaskContext) -> str:
        from assistant.tools.scopes import max_steps_for, select_scopes

        tools = registry_factory(context.approvals())
        scopes = task.payload.get("scopes") or ["all"]
        selected = select_scopes(tools, scopes)
        steps = max(BACKGROUND_MAX_STEPS, max_steps_for(scopes) or 0)
        try:
            result = await application.run_turn(
                task.prompt,
                tools=selected,
                conversation_id=application.start_conversation(),
                cancel=context.cancel,
                event_sink=context.event_sink,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=False,
                max_steps=steps,
                timeout_seconds=BACKGROUND_TIMEOUT_SECONDS,
                plan="auto",
            )
        finally:
            close = getattr(tools, "close", None)
            if close is not None:
                close()
        return result.message

    return run
