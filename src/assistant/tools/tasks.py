"""Tools for Jamie's own work: what is running, what is scheduled, and stopping either.

They let the user manage background tasks, job applications, and schedules in plain
words ("what are you working on?", "stop the morning summary") instead of commands.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from assistant import schedule
from assistant.background.store import ACTIVE, BackgroundStore
from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError

TASK_TOOLS = frozenset(
    {"jamie_work_status", "jamie_start_background", "jamie_cancel_work", "jamie_schedule"}
)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _short(text: str, limit: int = 80) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else f"{line[: limit - 1]}…"


class WorkStatusTool:
    name = "jamie_work_status"
    description = (
        "List Jamie's own background tasks, job applications, approvals waiting for the "
        "user, and scheduled tasks. Use for 'what are you working on' or 'how are my "
        "applications going'."
    )
    risk = PermissionLevel.READ

    class Input(_Input):
        include_finished: bool = False

    input_schema = Input

    def __init__(self, store: BackgroundStore, memory: Callable[[], object | None]) -> None:
        self._store = store
        self._memory = memory

    def execute(self, arguments: Input) -> str:
        tasks = self._store.tasks(active_only=not arguments.include_finished, limit=25)
        data: dict[str, object] = {
            "background": [
                {
                    "id": task.id,
                    "kind": task.kind,
                    "status": task.status,
                    "task": _short(task.prompt),
                    **({"progress": task.progress} if task.progress else {}),
                    **({"result": _short(task.result, 200)} if task.result else {}),
                    **({"error": _short(task.error, 200)} if task.error else {}),
                }
                for task in tasks
            ],
            "waiting_for_user": [
                {"approval": p.id, "task": p.task_id, "action": _short(p.summary, 120)}
                for p in self._store.pending_approvals()
            ],
        }
        memory = self._memory()
        if memory is not None:
            data["scheduled"] = [
                {
                    "id": task.id,
                    "when": schedule.describe(task.repeat, task.time),
                    "task": _short(task.prompt),
                }
                for task in memory.schedules()
            ]
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class StartBackgroundTool:
    name = "jamie_start_background"
    description = (
        "Start a task that runs in the background while the user keeps working; Jamie "
        "notifies them when it finishes. Use when the user wants something done without "
        "waiting, or for long work."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        task: str = Field(min_length=1, max_length=4000)

    input_schema = Input

    def __init__(self, store: BackgroundStore) -> None:
        self._store = store

    def execute(self, arguments: Input) -> str:
        task_id = self._store.add_task(arguments.task, source="agent")
        return f"Started background task {task_id}. Tell the user they will be notified."


class CancelWorkTool:
    name = "jamie_cancel_work"
    description = (
        "Stop a background task or job application (kind 'task') or remove a scheduled "
        "task (kind 'schedule') by the id from jamie_work_status."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        kind: str = Field(pattern="^(task|schedule)$")
        id: int = Field(ge=1)

    input_schema = Input

    def __init__(self, store: BackgroundStore, memory: Callable[[], object | None]) -> None:
        self._store = store
        self._memory = memory

    def execute(self, arguments: Input) -> str:
        if arguments.kind == "schedule":
            memory = self._memory()
            if memory is None or not memory.remove_schedule(arguments.id):
                raise ToolError(f"No scheduled task {arguments.id}.")
            return f"Removed scheduled task {arguments.id}."
        task = self._store.task(arguments.id)
        if task is None or task.status not in ACTIVE:
            raise ToolError(f"Background task {arguments.id} is not running or queued.")
        self._store.request_cancel(arguments.id)
        return f"Stopping background task {arguments.id}."


class ScheduleTool:
    name = "jamie_schedule"
    description = (
        "Schedule a task Jamie runs by itself later. when: 'daily HH:MM', 'weekly mon HH:MM', "
        "'every 2h', 'at HH:MM', or 'on YYYY-MM-DD HH:MM'. For a simple reminder to the user, "
        "use reminders_create instead."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        when: str = Field(min_length=3, max_length=40)
        task: str = Field(min_length=1, max_length=2000)

    input_schema = Input

    def __init__(self, memory: Callable[[], object | None]) -> None:
        self._memory = memory

    def execute(self, arguments: Input) -> str:
        memory = self._memory()
        if memory is None:
            raise ToolError("Scheduling is not available here.")
        try:
            repeat, time, prompt = schedule.parse(f"{arguments.when} {arguments.task}")
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        now = datetime.now(UTC)
        first = schedule.next_run(repeat, time, now)
        if first <= now:
            raise ToolError("That time has already passed.")
        task_id = memory.add_schedule(prompt, repeat, time, first)
        when = first.astimezone().strftime("%a %d %b %H:%M")
        return f"Scheduled task {task_id}: {schedule.describe(repeat, time)}, first {when}."


def register_task_tools(registry, store: BackgroundStore | None, memory=None):
    """``memory`` returns the conversation store (which holds schedules), or None."""
    if store is None:
        return registry
    lookup = memory if callable(memory) else (lambda: memory)
    registry.register(WorkStatusTool(store, lookup))
    registry.register(StartBackgroundTool(store))
    registry.register(CancelWorkTool(store, lookup))
    registry.register(ScheduleTool(lookup))
    return registry
