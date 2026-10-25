"""SQLite storage for durable tasks, steps, approvals, and schedules.

Tasks live in their own database because their lifecycle differs from conversation
memory. Full tool arguments, results, and approval displays are kept in
``task_payloads`` and deleted when a task finishes, so step and approval rows hold
only redacted summaries and references.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from assistant.tasks.models import (
    TERMINAL_STATES,
    ApprovalKind,
    ApprovalStatus,
    MissedPolicy,
    Schedule,
    ScheduleKind,
    StepStatus,
    Task,
    TaskApproval,
    TaskState,
    TaskStep,
    check_transition,
)

SCHEMA_VERSION = 1


def _check(values: Iterable[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    instruction TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ({_check(TaskState)})),
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    scheduled_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    scope TEXT NOT NULL,
    workspace TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    result_summary TEXT,
    schedule_id TEXT REFERENCES schedules(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS tasks_runnable ON tasks(state, priority, created_at);

CREATE TABLE IF NOT EXISTS task_payloads (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_steps (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_check(StepStatus)})),
    idempotency_key TEXT NOT NULL UNIQUE,
    args_payload_id TEXT REFERENCES task_payloads(id) ON DELETE SET NULL,
    result_payload_id TEXT REFERENCES task_payloads(id) ON DELETE SET NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (task_id, step_index)
);

CREATE TABLE IF NOT EXISTS task_approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    step_id TEXT REFERENCES task_steps(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ({_check(ApprovalKind)})),
    tool TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_check(ApprovalStatus)})),
    task_revision INTEGER NOT NULL,
    display_payload_id TEXT REFERENCES task_payloads(id) ON DELETE SET NULL,
    requested_at TEXT NOT NULL,
    expires_at TEXT,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS approvals_by_task ON task_approvals(task_id, status);

CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    instruction TEXT NOT NULL,
    scope TEXT NOT NULL,
    workspace TEXT,
    kind TEXT NOT NULL CHECK (kind IN ({_check(ScheduleKind)})),
    spec_json TEXT NOT NULL,
    timezone TEXT NOT NULL,
    next_run_at TEXT,
    missed_policy TEXT NOT NULL CHECK (missed_policy IN ({_check(MissedPolicy)})),
    enabled INTEGER NOT NULL DEFAULT 1,
    cursor TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class TaskStoreError(RuntimeError):
    """The task database could not be read or written."""


class StaleTaskError(TaskStoreError):
    """Another writer changed the task first; reload it before retrying."""


def summarize_arguments(args: dict) -> str:
    """Ids are kept; other values are reduced to their shape, so no content is stored."""
    summary: dict[str, object] = {}
    for key, value in args.items():
        if key.endswith("_id") and isinstance(value, str):
            summary[key] = value[:64]
        elif isinstance(value, str):
            summary[key] = f"<{len(value)} chars>"
        elif isinstance(value, list):
            summary[key] = f"<{len(value)} items>"
        elif isinstance(value, bool | int | float) or value is None:
            summary[key] = value
        else:
            summary[key] = "<object>"
    return json.dumps(summary, separators=(",", ":"))


def default_tasks_path() -> Path:
    override = os.environ.get("JAMIE_TASKS_PATH")
    if override:
        return Path(override).expanduser()
    from assistant.memory import default_memory_path

    return default_memory_path().with_name("tasks.sqlite3")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _text(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _new_id() -> str:
    return uuid4().hex


def _task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        instruction=row["instruction"],
        state=TaskState(row["state"]),
        priority=row["priority"],
        created_at=_time(row["created_at"]),
        updated_at=_time(row["updated_at"]),
        scheduled_at=_time(row["scheduled_at"]),
        attempt_count=row["attempt_count"],
        last_error=row["last_error"],
        revision=row["revision"],
        scope=row["scope"],
        workspace=row["workspace"],
        lease_owner=row["lease_owner"],
        lease_expires_at=_time(row["lease_expires_at"]),
        result_summary=row["result_summary"],
        schedule_id=row["schedule_id"],
    )


def _step(row: sqlite3.Row) -> TaskStep:
    return TaskStep(
        id=row["id"],
        task_id=row["task_id"],
        step_index=row["step_index"],
        tool_name=row["tool_name"],
        summary=row["summary"],
        status=StepStatus(row["status"]),
        idempotency_key=row["idempotency_key"],
        args_payload_id=row["args_payload_id"],
        result_payload_id=row["result_payload_id"],
        started_at=_time(row["started_at"]),
        finished_at=_time(row["finished_at"]),
    )


def _approval(row: sqlite3.Row) -> TaskApproval:
    return TaskApproval(
        id=row["id"],
        task_id=row["task_id"],
        step_id=row["step_id"],
        kind=ApprovalKind(row["kind"]),
        tool=row["tool"],
        fingerprint=row["fingerprint"],
        status=ApprovalStatus(row["status"]),
        task_revision=row["task_revision"],
        display_payload_id=row["display_payload_id"],
        requested_at=_time(row["requested_at"]),
        expires_at=_time(row["expires_at"]),
        decided_at=_time(row["decided_at"]),
    )


def _schedule(row: sqlite3.Row) -> Schedule:
    return Schedule(
        id=row["id"],
        title=row["title"],
        instruction=row["instruction"],
        scope=row["scope"],
        workspace=row["workspace"],
        kind=ScheduleKind(row["kind"]),
        spec=json.loads(row["spec_json"]),
        timezone=row["timezone"],
        next_run_at=_time(row["next_run_at"]),
        missed_policy=MissedPolicy(row["missed_policy"]),
        enabled=bool(row["enabled"]),
        cursor=row["cursor"],
        created_at=_time(row["created_at"]),
    )


class TaskStore:
    """Durable task records. Each method opens a short connection, so threads are safe."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = Path(path)
        self._clock = clock
        self._initialize()

    # -- connection -----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool = False):
        try:
            connection = self._connect()
        except sqlite3.Error as exc:
            raise TaskStoreError(f"Cannot open the task database: {exc}") from exc
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch(mode=0o600)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise TaskStoreError(
                        f"The task database uses a newer schema ({version}); update Jamie."
                    )
                if version < SCHEMA_VERSION:
                    connection.executescript(SCHEMA)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            raise TaskStoreError(f"Cannot initialize {self.path}: {exc}") from exc

    def now(self) -> datetime:
        # UTC so that time arithmetic is elapsed time, never wall-clock time.
        return self._clock().astimezone(UTC)

    # -- tasks ----------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        instruction: str,
        *,
        scope: str = "personal",
        workspace: str | None = None,
        priority: int = 0,
        state: TaskState = TaskState.READY,
        scheduled_at: datetime | None = None,
        schedule_id: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> Task:
        if state not in {TaskState.READY, TaskState.SCHEDULED, TaskState.WAITING_APPROVAL}:
            raise ValueError(f"A new task cannot start as {state.value}.")
        task_id = _new_id()
        now = _text(self.now())
        values = (
            task_id, title, instruction, state.value, priority, now, now,
            _text(scheduled_at), scope, workspace, schedule_id,
        )  # fmt: skip
        sql = (
            "INSERT INTO tasks (id, title, instruction, state, priority, created_at, "
            "updated_at, scheduled_at, scope, workspace, schedule_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        if connection is not None:
            connection.execute(sql, values)
            return _task(
                connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            )
        with self._transaction() as connection:
            connection.execute(sql, values)
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> Task:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return _task(row)

    def find_task(self, prefix: str) -> Task | None:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE id LIKE ? LIMIT 2", (f"{prefix}%",)
            ).fetchall()
        return _task(rows[0]) if len(rows) == 1 else None

    def list_tasks(
        self, states: Iterable[TaskState] | None = None, *, limit: int = 50
    ) -> list[Task]:
        query = "SELECT * FROM tasks"
        params: list[object] = []
        if states is not None:
            states = list(states)
            query += f" WHERE state IN ({', '.join('?' for _ in states)})"
            params += [state.value for state in states]
        query += " ORDER BY updated_at DESC LIMIT ?"
        with self._transaction() as connection:
            rows = connection.execute(query, [*params, limit]).fetchall()
        return [_task(row) for row in rows]

    def count(self, state: TaskState) -> int:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE state = ?", (state.value,)
            ).fetchone()
        return int(row[0])

    def count_runnable(self) -> int:
        """Ready tasks whose (optional) start time has arrived."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE state = ? "
                "AND (scheduled_at IS NULL OR scheduled_at <= ?)",
                (TaskState.READY.value, _text(self.now())),
            ).fetchone()
        return int(row[0])

    def transition(
        self,
        task_id: str,
        target: TaskState,
        *,
        expected: TaskState | None = None,
        connection: sqlite3.Connection | None = None,
        **fields: object,
    ) -> Task:
        """Move a task to ``target``; a concurrent change raises StaleTaskError."""
        if connection is None:
            with self._transaction(immediate=True) as connection:
                return self.transition(
                    task_id, target, expected=expected, connection=connection, **fields
                )
        row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        current = TaskState(row["state"])
        if expected is not None and current is not expected:
            raise StaleTaskError(f"Task {task_id} is {current.value}, not {expected.value}.")
        check_transition(current, target)
        updates: dict[str, object] = {"state": target.value, "updated_at": _text(self.now())}
        if target is not TaskState.RUNNING:
            updates |= {"lease_owner": None, "lease_expires_at": None}
        for key, value in fields.items():
            updates[key] = _text(value) if isinstance(value, datetime) else value
        assignments = ", ".join(f"{key} = ?" for key in updates)
        changed = connection.execute(
            f"UPDATE tasks SET {assignments} WHERE id = ? AND state = ?",
            [*updates.values(), task_id, current.value],
        ).rowcount
        if changed != 1:
            raise StaleTaskError(f"Task {task_id} changed during the update.")
        if target in TERMINAL_STATES:
            connection.execute("DELETE FROM task_payloads WHERE task_id = ?", (task_id,))
            connection.execute(
                "UPDATE task_approvals SET status = ? WHERE task_id = ? AND status IN (?, ?)",
                (ApprovalStatus.SUPERSEDED.value, task_id, ApprovalStatus.PENDING.value,
                 ApprovalStatus.APPROVED.value),
            )  # fmt: skip
        return _task(connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())

    def claim(self, worker_id: str, lease_seconds: float = 120) -> Task | None:
        """Atomically take the highest-priority ready task and lease it to one worker."""
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT id FROM tasks WHERE state = ? "
                "AND (scheduled_at IS NULL OR scheduled_at <= ?) "
                "ORDER BY priority DESC, created_at LIMIT 1",
                (TaskState.READY.value, _text(self.now())),
            ).fetchone()
            if row is None:
                return None
            return self.transition(
                row["id"],
                TaskState.RUNNING,
                expected=TaskState.READY,
                connection=connection,
                lease_owner=worker_id,
                lease_expires_at=self.now() + timedelta(seconds=lease_seconds),
            )

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: float = 120) -> bool:
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE tasks SET lease_expires_at = ? "
                "WHERE id = ? AND state = ? AND lease_owner = ?",
                (
                    _text(self.now() + timedelta(seconds=lease_seconds)),
                    task_id,
                    TaskState.RUNNING.value,
                    worker_id,
                ),
            ).rowcount
        return changed == 1

    def release_expired_leases(self) -> list[str]:
        """Return running tasks whose worker stopped heartbeating to the ready queue."""
        released = []
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT id FROM tasks WHERE state = ? AND lease_expires_at < ?",
                (TaskState.RUNNING.value, _text(self.now())),
            ).fetchall()
            for row in rows:
                self.transition(
                    row["id"],
                    TaskState.READY,
                    connection=connection,
                    last_error="The worker stopped before finishing; resuming.",
                )
                released.append(row["id"])
        return released

    def revise(self, task_id: str, instruction: str) -> Task:
        """Change the instruction; every outstanding approval stops matching."""
        with self._transaction(immediate=True) as connection:
            row = connection.execute("SELECT state FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            if TaskState(row["state"]) in TERMINAL_STATES | {TaskState.RUNNING}:
                raise StaleTaskError("Only a task that is not running or finished can be edited.")
            connection.execute(
                "UPDATE tasks SET instruction = ?, revision = revision + 1, updated_at = ? "
                "WHERE id = ?",
                (instruction, _text(self.now()), task_id),
            )
            connection.execute(
                "UPDATE task_approvals SET status = ?, decided_at = ? "
                "WHERE task_id = ? AND status IN (?, ?)",
                (ApprovalStatus.SUPERSEDED.value, _text(self.now()), task_id,
                 ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value),
            )  # fmt: skip
            if TaskState(row["state"]) is TaskState.WAITING_APPROVAL:
                self.transition(task_id, TaskState.READY, connection=connection)
        return self.get_task(task_id)

    # -- payloads -------------------------------------------------------------------

    def save_payload(self, task_id: str, value: object) -> str:
        payload_id = _new_id()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO task_payloads (id, task_id, json) VALUES (?, ?, ?)",
                (payload_id, task_id, json.dumps(value, ensure_ascii=False)),
            )
        return payload_id

    def load_payload(self, payload_id: str | None) -> object | None:
        if payload_id is None:
            return None
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT json FROM task_payloads WHERE id = ?", (payload_id,)
            ).fetchone()
        return json.loads(row["json"]) if row else None

    # -- steps ----------------------------------------------------------------------

    def start_step(self, task: Task, tool: str, args: dict) -> TaskStep:
        """Record the intended operation before it runs, with a stable operation key."""
        step_id = _new_id()
        with self._transaction(immediate=True) as connection:
            index = connection.execute(
                "SELECT COALESCE(MAX(step_index) + 1, 0) FROM task_steps WHERE task_id = ?",
                (task.id,),
            ).fetchone()[0]
            payload_id = _new_id()
            connection.execute(
                "INSERT INTO task_payloads (id, task_id, json) VALUES (?, ?, ?)",
                (payload_id, task.id, json.dumps(args, ensure_ascii=False)),
            )
            connection.execute(
                "INSERT INTO task_steps (id, task_id, step_index, tool_name, summary, status, "
                "idempotency_key, args_payload_id, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    step_id,
                    task.id,
                    index,
                    tool,
                    summarize_arguments(args),
                    StepStatus.STARTED.value,
                    f"{task.id}:{task.revision}:{index}",
                    payload_id,
                    _text(self.now()),
                ),
            )
        return self.get_step(step_id)

    def finish_step(self, step_id: str, status: StepStatus, result: dict | None = None) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT task_id FROM task_steps WHERE id = ?", (step_id,)
            ).fetchone()
            payload_id = None
            if result is not None:
                payload_id = _new_id()
                connection.execute(
                    "INSERT INTO task_payloads (id, task_id, json) VALUES (?, ?, ?)",
                    (payload_id, row["task_id"], json.dumps(result, ensure_ascii=False)),
                )
            finished = None if status is StepStatus.WAITING else _text(self.now())
            connection.execute(
                "UPDATE task_steps SET status = ?, result_payload_id = ?, finished_at = ? "
                "WHERE id = ?",
                (status.value, payload_id, finished, step_id),
            )

    def discard_step(self, step_id: str) -> None:
        """Forget an interrupted step that is safe to run again."""
        with self._transaction() as connection:
            connection.execute("DELETE FROM task_steps WHERE id = ?", (step_id,))

    def get_step(self, step_id: str) -> TaskStep:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM task_steps WHERE id = ?", (step_id,)).fetchone()
        if row is None:
            raise KeyError(step_id)
        return _step(row)

    def steps(self, task_id: str) -> list[TaskStep]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)
            ).fetchall()
        return [_step(row) for row in rows]

    def started_steps(self) -> list[TaskStep]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT task_steps.* FROM task_steps JOIN tasks ON tasks.id = task_steps.task_id "
                "WHERE task_steps.status = ? AND tasks.state = ? ORDER BY step_index",
                (StepStatus.STARTED.value, TaskState.READY.value),
            ).fetchall()
        return [_step(row) for row in rows]

    # -- approvals ------------------------------------------------------------------

    def create_approval(
        self,
        task: Task,
        *,
        tool: str,
        fingerprint: str,
        display: dict,
        kind: ApprovalKind = ApprovalKind.TOOL,
        step_id: str | None = None,
        expires_at: datetime | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> TaskApproval:
        if connection is None:
            with self._transaction(immediate=True) as connection:
                return self.create_approval(
                    task, tool=tool, fingerprint=fingerprint, display=display, kind=kind,
                    step_id=step_id, expires_at=expires_at, connection=connection,
                )  # fmt: skip
        approval_id = _new_id()
        payload_id = _new_id()
        connection.execute(
            "INSERT INTO task_payloads (id, task_id, json) VALUES (?, ?, ?)",
            (payload_id, task.id, json.dumps(display, ensure_ascii=False)),
        )
        connection.execute(
            "UPDATE task_approvals SET status = ?, decided_at = ? "
            "WHERE task_id = ? AND status = ?",
            (ApprovalStatus.SUPERSEDED.value, _text(self.now()), task.id,
             ApprovalStatus.PENDING.value),
        )  # fmt: skip
        connection.execute(
            "INSERT INTO task_approvals (id, task_id, step_id, kind, tool, fingerprint, status, "
            "task_revision, display_payload_id, requested_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (approval_id, task.id, step_id, kind.value, tool, fingerprint,
             ApprovalStatus.PENDING.value, task.revision, payload_id, _text(self.now()),
             _text(expires_at)),
        )  # fmt: skip
        row = connection.execute(
            "SELECT * FROM task_approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return _approval(row)

    def get_approval(self, approval_id: str) -> TaskApproval:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM task_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return _approval(row)

    def find_approval_by_prefix(self, prefix: str) -> TaskApproval | None:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM task_approvals WHERE id LIKE ? LIMIT 2", (f"{prefix}%",)
            ).fetchall()
        return _approval(rows[0]) if len(rows) == 1 else None

    def approvals(
        self, *, task_id: str | None = None, status: ApprovalStatus | None = None
    ) -> list[TaskApproval]:
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM task_approvals{where} ORDER BY requested_at", params
            ).fetchall()
        return [_approval(row) for row in rows]

    def set_approval_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        expected: ApprovalStatus,
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        sql = "UPDATE task_approvals SET status = ?, decided_at = ? WHERE id = ? AND status = ?"
        values = (status.value, _text(self.now()), approval_id, expected.value)
        if connection is not None:
            return connection.execute(sql, values).rowcount == 1
        with self._transaction() as connection:
            return connection.execute(sql, values).rowcount == 1

    def transaction(self):
        """An immediate transaction for multi-record updates (approvals, schedules)."""
        return self._transaction(immediate=True)

    # -- schedules ------------------------------------------------------------------

    def create_schedule(
        self,
        *,
        title: str,
        instruction: str,
        kind: ScheduleKind,
        spec: dict,
        timezone: str,
        next_run_at: datetime | None,
        scope: str = "personal",
        workspace: str | None = None,
        missed_policy: MissedPolicy = MissedPolicy.RUN_LATE,
    ) -> Schedule:
        schedule_id = _new_id()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO schedules (id, title, instruction, scope, workspace, kind, "
                "spec_json, timezone, next_run_at, missed_policy, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (schedule_id, title, instruction, scope, workspace, kind.value,
                 json.dumps(spec), timezone, _text(next_run_at), missed_policy.value,
                 _text(self.now())),
            )  # fmt: skip
        return self.get_schedule(schedule_id)

    def get_schedule(self, schedule_id: str) -> Schedule:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
        if row is None:
            raise KeyError(schedule_id)
        return _schedule(row)

    def list_schedules(self) -> list[Schedule]:
        with self._transaction() as connection:
            rows = connection.execute("SELECT * FROM schedules ORDER BY created_at").fetchall()
        return [_schedule(row) for row in rows]

    def due_schedules(self, now: datetime) -> list[Schedule]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM schedules WHERE enabled = 1 AND next_run_at IS NOT NULL "
                "AND next_run_at <= ? ORDER BY next_run_at",
                (_text(now),),
            ).fetchall()
        return [_schedule(row) for row in rows]

    def update_schedule(
        self,
        schedule_id: str,
        *,
        connection: sqlite3.Connection,
        next_run_at: datetime | None,
        cursor: str | None = None,
        enabled: bool = True,
    ) -> None:
        connection.execute(
            "UPDATE schedules SET next_run_at = ?, cursor = COALESCE(?, cursor), enabled = ? "
            "WHERE id = ?",
            (_text(next_run_at), cursor, int(enabled), schedule_id),
        )

    def disable_schedule(self, schedule_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("UPDATE schedules SET enabled = 0 WHERE id = ?", (schedule_id,))

    # -- worker state ---------------------------------------------------------------

    def get_flag(self, key: str) -> str | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT value FROM worker_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_flag(self, key: str, value: str | None) -> None:
        with self._transaction() as connection:
            if value is None:
                connection.execute("DELETE FROM worker_state WHERE key = ?", (key,))
            else:
                connection.execute(
                    "INSERT INTO worker_state (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
