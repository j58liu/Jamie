"""Durable task execution: claim, rebuild, validate, approve, execute, persist.

Each tool call is recorded as a step *before* it runs, with a stable operation key,
and its outcome is stored afterwards. A task is rebuilt from these records after a
restart or an approval, so no in-memory agent state has to survive.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from assistant.agent.loop import AgentError, ToolResult, run_agent, tool_observation
from assistant.agent.protocol import ToolAction
from assistant.llm.base import LLMBackend, LLMError, Message
from assistant.tasks.approvals import DeferredApprovalProvider
from assistant.tasks.context import current_operation_key
from assistant.tasks.models import ApprovalKind, ApprovalStatus, StepStatus, Task, TaskState
from assistant.tasks.notifications import Notifier, NullNotifier
from assistant.tasks.store import TaskStore

MAX_ATTEMPTS = 3
MAX_STEPS = 16
TIMEOUT_SECONDS = 600
LEASE_SECONDS = 120
RETRY_DELAY = timedelta(minutes=5)

# (task, approval provider) -> a tool registry scoped for that task
RegistryFactory = Callable[[Task, DeferredApprovalProvider], object]


class StepRecorder:
    """Wrap a registry so every tool call becomes a durable step."""

    def __init__(self, registry, store: TaskStore, task: Task,
                 approvals: DeferredApprovalProvider) -> None:  # fmt: skip
        self._registry = registry
        self._store = store
        self._task = task
        self._approvals = approvals

    def __getattr__(self, name: str):
        return getattr(self._registry, name)

    def execute(self, tool: str, args: dict) -> ToolResult:
        step = self._store.start_step(self._task, tool, args)
        return self.execute_step(step.id, step.idempotency_key, tool, args)

    def execute_step(self, step_id: str, key: str, tool: str, args: dict) -> ToolResult:
        self._approvals.current_step_id = step_id
        token = current_operation_key.set(key)
        try:
            result = self._registry.execute(tool, args)
        except Exception as exc:
            self._store.finish_step(
                step_id, StepStatus.FAILED, {"success": False, "output": str(exc)}
            )
            raise
        finally:
            current_operation_key.reset(token)
            self._approvals.current_step_id = None
        if result.denied:
            self._store.finish_step(step_id, StepStatus.WAITING)
        else:
            status = StepStatus.SUCCEEDED if result.success else StepStatus.FAILED
            self._store.finish_step(
                step_id, status, {"success": result.success, "output": result.output}
            )
        return result


def rebuild_messages(store: TaskStore, task: Task) -> list[Message]:
    """The action/observation pairs of every finished step, in order."""
    messages: list[Message] = []
    for step in store.steps(task.id):
        if step.status not in {StepStatus.SUCCEEDED, StepStatus.FAILED}:
            continue
        args = store.load_payload(step.args_payload_id) or {}
        result = store.load_payload(step.result_payload_id) or {}
        action = ToolAction(action="tool", tool=step.tool_name, args=args)
        messages.append(Message(role="assistant", content=action.model_dump_json()))
        messages.append(
            tool_observation(
                action,
                ToolResult(success=bool(result.get("success")), output=str(result.get("output"))),
            )
        )
    return messages


@dataclass
class _Heartbeat:
    store: TaskStore
    task_id: str
    worker_id: str
    lease_seconds: float

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True, name="jamie-lease")

    def _beat(self) -> None:
        while not self._stop.wait(self.lease_seconds / 3):
            self.store.heartbeat(self.task_id, self.worker_id, self.lease_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


class TaskEngine:
    def __init__(
        self,
        store: TaskStore,
        *,
        backend_factory: Callable[[], LLMBackend],
        registry_factory: RegistryFactory,
        notifier: Notifier | None = None,
        worker_id: str | None = None,
        lease_seconds: float = LEASE_SECONDS,
        max_steps: int = MAX_STEPS,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> None:
        self._store = store
        self._backend_factory = backend_factory
        self._registry_factory = registry_factory
        self._notifier = notifier if notifier is not None else NullNotifier()
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self._lease = lease_seconds
        self._max_steps = max_steps
        self._timeout = timeout_seconds
        self._backend: LLMBackend | None = None

    def run_all(self, limit: int = 20) -> list[Task]:
        finished = []
        for _ in range(limit):
            task = self.run_next()
            if task is None:
                break
            finished.append(task)
        return finished

    def run_next(self) -> Task | None:
        task = self._store.claim(self.worker_id, self._lease)
        if task is None:
            return None
        with _Heartbeat(self._store, task.id, self.worker_id, self._lease):
            return self._run(task)

    def _backend_once(self) -> LLMBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _run(self, task: Task) -> Task:
        approvals = DeferredApprovalProvider(self._store, task)
        recorder = StepRecorder(self._registry_factory(task, approvals), self._store, task,
                                approvals)  # fmt: skip
        try:
            stop = self._resume_approved(task, recorder)
            if stop is not None:
                return stop
            answer = run_agent(
                self._backend_once(),
                recorder,
                task.instruction,
                max_steps=self._max_steps,
                timeout_seconds=self._timeout,
                prior_messages=rebuild_messages(self._store, task),
            )
        except (AgentError, LLMError) as exc:
            return self._retry_or_fail(task, str(exc))
        except Exception as exc:  # a tool crashed; keep the record, do not loop forever
            return self._retry_or_fail(task, f"{type(exc).__name__}: {exc}")

        if approvals.deferred is not None:
            return self._wait(task, approvals.deferred.tool)
        updated = self._store.transition(
            task.id, TaskState.COMPLETED, expected=TaskState.RUNNING, result_summary=answer
        )
        self._notifier.notify("Jamie finished a task", f"{task.title}: {answer}")
        return updated

    def _resume_approved(self, task: Task, recorder: StepRecorder) -> Task | None:
        """Run the operation the user approved, exactly as it was stored."""
        for approval in self._store.approvals(task_id=task.id, status=ApprovalStatus.APPROVED):
            if approval.kind is ApprovalKind.RUN_MISSED:
                self._store.set_approval_status(
                    approval.id, ApprovalStatus.USED, expected=ApprovalStatus.APPROVED
                )
                continue
            if approval.step_id is None:
                continue
            step = self._store.get_step(approval.step_id)
            if step.status not in {StepStatus.WAITING, StepStatus.AMBIGUOUS}:
                continue
            args = self._store.load_payload(step.args_payload_id) or {}
            result = recorder.execute_step(step.id, step.idempotency_key, step.tool_name, args)
            if result.denied:  # the approval no longer matched; ask again
                return self._wait(task, step.tool_name)
        return None

    def _wait(self, task: Task, tool: str) -> Task:
        updated = self._store.transition(
            task.id, TaskState.WAITING_APPROVAL, expected=TaskState.RUNNING
        )
        self._notifier.notify(
            "Jamie needs approval", f"{task.title}: {tool.replace('_', ' ')}. Run: jamie tasks"
        )
        return updated

    def _retry_or_fail(self, task: Task, error: str) -> Task:
        attempts = task.attempt_count + 1
        if attempts < MAX_ATTEMPTS:
            return self._store.transition(
                task.id, TaskState.READY, expected=TaskState.RUNNING,
                attempt_count=attempts, last_error=error,
                scheduled_at=self._store.now() + RETRY_DELAY * attempts,
            )  # fmt: skip
        updated = self._store.transition(
            task.id, TaskState.FAILED, expected=TaskState.RUNNING,
            attempt_count=attempts, last_error=error,
        )  # fmt: skip
        self._notifier.notify("Jamie task failed", f"{task.title}: {error}")
        return updated
