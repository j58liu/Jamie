"""Task domain model: states, explicit transitions, and stored records.

Tasks are separate from conversation history. A task is rebuilt from validated
records after a restart; no Python thread or model state is serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TaskState(StrEnum):
    SCHEDULED = "scheduled"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})

TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.SCHEDULED: frozenset({TaskState.READY, TaskState.CANCELLED}),
    TaskState.READY: frozenset(
        {TaskState.RUNNING, TaskState.WAITING_APPROVAL, TaskState.CANCELLED}
    ),
    TaskState.RUNNING: frozenset(
        {
            TaskState.WAITING_APPROVAL,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.READY,  # lease lost or retry after an error
            TaskState.CANCELLED,
        }
    ),
    TaskState.WAITING_APPROVAL: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    """A task cannot move between these two states."""


def check_transition(current: TaskState, target: TaskState) -> None:
    if target not in TRANSITIONS[current]:
        raise InvalidTransition(f"A {current.value} task cannot become {target.value}.")


class StepStatus(StrEnum):
    STARTED = "started"  # recorded before execution; a crash leaves it here
    WAITING = "waiting"  # paused for approval; resumes with the stored arguments
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"  # may or may not have happened; the user decides


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    USED = "used"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ApprovalKind(StrEnum):
    TOOL = "tool"  # a tool call that policy says needs confirmation
    RETRY = "retry"  # an interrupted step that may already have run
    RUN_MISSED = "run_missed"  # a scheduled run the computer missed


class MissedPolicy(StrEnum):
    RUN_LATE = "run_late"
    SKIP = "skip"
    ASK = "ask"


class ScheduleKind(StrEnum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    INTERVAL = "interval"
    EVENT = "event"


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    instruction: str
    state: TaskState
    priority: int
    created_at: datetime
    updated_at: datetime
    scheduled_at: datetime | None
    attempt_count: int
    last_error: str | None
    revision: int
    scope: str
    workspace: str | None
    lease_owner: str | None
    lease_expires_at: datetime | None
    result_summary: str | None
    schedule_id: str | None


@dataclass(frozen=True)
class TaskStep:
    id: str
    task_id: str
    step_index: int
    tool_name: str
    summary: str
    status: StepStatus
    idempotency_key: str
    args_payload_id: str | None
    result_payload_id: str | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class TaskApproval:
    id: str
    task_id: str
    step_id: str | None
    kind: ApprovalKind
    tool: str
    fingerprint: str
    status: ApprovalStatus
    task_revision: int
    display_payload_id: str | None
    requested_at: datetime
    expires_at: datetime | None
    decided_at: datetime | None


@dataclass(frozen=True)
class Schedule:
    id: str
    title: str
    instruction: str
    scope: str
    workspace: str | None
    kind: ScheduleKind
    spec: dict
    timezone: str
    next_run_at: datetime | None
    missed_policy: MissedPolicy
    enabled: bool
    cursor: str | None
    created_at: datetime
