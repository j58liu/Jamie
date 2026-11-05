"""Durable state for background tasks and the approvals they wait on.

Every Jamie process (the app, the terminal, `jamie serve`) opens the same SQLite file,
so a task queued in the terminal runs in the app, and an approval asked for by a
background task can be answered from any of them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'cli',
    progress TEXT NOT NULL DEFAULT '',
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    task_id INTEGER,
    tool TEXT NOT NULL,
    permission TEXT NOT NULL,
    arguments_digest TEXT NOT NULL,
    summary TEXT NOT NULL,
    attachment TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    answered_at TEXT,
    answered_by TEXT
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, id);
CREATE INDEX IF NOT EXISTS approvals_status ON approvals(status, created_at);
"""

QUEUED = "queued"
RUNNING = "running"
WAITING = "waiting_approval"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"
NEEDS_INPUT = "needs_input"  # stopped with questions only the user can answer
ACTIVE = frozenset({QUEUED, RUNNING, WAITING})
FINISHED = frozenset({DONE, FAILED, CANCELLED, INTERRUPTED, NEEDS_INPUT})

PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"
EXPIRED = "expired"
APPROVAL_LIFETIME = timedelta(hours=24)
MAX_TEXT = 20_000


class BackgroundStoreError(RuntimeError):
    """The background database could not be read or written."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def arguments_digest(tool: str, arguments: dict[str, Any]) -> str:
    """Bind an approval to one exact operation; any change needs a new approval."""
    canonical = json.dumps(
        {"tool": tool, "arguments": arguments}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class BackgroundTask:
    id: int
    kind: str
    prompt: str
    payload: dict[str, Any]
    status: str
    source: str
    progress: str
    result: str | None
    error: str | None
    cancel_requested: bool
    created_at: str
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class PendingApproval:
    id: str
    task_id: int | None
    tool: str
    permission: str
    arguments_digest: str
    summary: str
    attachment: str | None
    status: str
    created_at: str
    expires_at: str
    answered_by: str | None


def _task(row: sqlite3.Row) -> BackgroundTask:
    return BackgroundTask(
        id=row["id"],
        kind=row["kind"],
        prompt=row["prompt"],
        payload=json.loads(row["payload"] or "{}"),
        status=row["status"],
        source=row["source"],
        progress=row["progress"],
        result=row["result"],
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _approval(row: sqlite3.Row) -> PendingApproval:
    return PendingApproval(
        id=row["id"],
        task_id=row["task_id"],
        tool=row["tool"],
        permission=row["permission"],
        arguments_digest=row["arguments_digest"],
        summary=row["summary"],
        attachment=row["attachment"],
        status=row["status"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        answered_by=row["answered_by"],
    )


def default_background_path() -> Path:
    from assistant.memory import default_memory_path

    return default_memory_path().with_name("background.sqlite3")


class BackgroundStore:
    """Tasks and approvals; each call opens its own connection, so threads may share it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_background_path()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._connect() as connection:
                connection.executescript(SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise BackgroundStoreError(f"Cannot open background tasks at {self.path}.") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _write(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        try:
            with self._connect() as connection:
                return connection.execute(sql, parameters)
        except sqlite3.Error as exc:
            raise BackgroundStoreError("Cannot update background tasks.") from exc

    def _read(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        try:
            with self._connect() as connection:
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise BackgroundStoreError("Cannot read background tasks.") from exc

    # -- tasks ------------------------------------------------------------------------

    def add_task(
        self,
        prompt: str,
        *,
        kind: str = "agent",
        payload: dict[str, Any] | None = None,
        source: str = "cli",
    ) -> int:
        if not prompt.strip():
            raise ValueError("A background task needs a prompt.")
        cursor = self._write(
            "INSERT INTO tasks(kind, prompt, payload, status, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                kind,
                prompt.strip()[:MAX_TEXT],
                json.dumps(payload or {}),
                QUEUED,
                source,
                _iso(_now()),
            ),
        )
        return int(cursor.lastrowid)

    def task(self, task_id: int) -> BackgroundTask | None:
        rows = self._read("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return _task(rows[0]) if rows else None

    def tasks(self, *, active_only: bool = False, limit: int = 50) -> list[BackgroundTask]:
        where = f"WHERE status IN ({','.join('?' * len(ACTIVE))})" if active_only else ""
        rows = self._read(
            f"SELECT * FROM tasks {where} ORDER BY id DESC LIMIT ?",
            (*sorted(ACTIVE), limit) if active_only else (limit,),
        )
        return [_task(row) for row in rows]

    def claim_next(self, kinds: frozenset[str] | None = None) -> BackgroundTask | None:
        """Mark the oldest queued task running and return it; None when the queue is empty.

        The UPDATE is atomic, so two workers never claim the same task.
        """
        kind_filter = f"AND kind IN ({','.join('?' * len(kinds))})" if kinds else ""
        rows = self._read(
            "UPDATE tasks SET status = ?, started_at = ? WHERE id = ("
            f"SELECT id FROM tasks WHERE status = ? AND cancel_requested = 0 {kind_filter} "
            "ORDER BY id LIMIT 1) RETURNING *",
            (RUNNING, _iso(_now()), QUEUED, *sorted(kinds or ())),
        )
        return _task(rows[0]) if rows else None

    def set_status(self, task_id: int, status: str) -> None:
        self._write("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))

    def set_progress(self, task_id: int, text: str) -> None:
        self._write("UPDATE tasks SET progress = ? WHERE id = ?", (text[:500], task_id))

    def update_payload(self, task_id: int, **changes: Any) -> dict[str, Any]:
        task = self.task(task_id)
        if task is None:
            raise BackgroundStoreError(f"No background task {task_id}.")
        payload = {**task.payload, **changes}
        self._write("UPDATE tasks SET payload = ? WHERE id = ?", (json.dumps(payload), task_id))
        return payload

    def finish(self, task_id: int, result: str) -> None:
        self._write(
            "UPDATE tasks SET status = ?, result = ?, finished_at = ? WHERE id = ?",
            (DONE, result[:MAX_TEXT], _iso(_now()), task_id),
        )

    def fail(self, task_id: int, error: str, *, status: str = FAILED) -> None:
        self._write(
            "UPDATE tasks SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error[:MAX_TEXT], _iso(_now()), task_id),
        )

    def request_cancel(self, task_id: int) -> bool:
        """Queued tasks stop at once; running ones stop at their next check."""
        task = self.task(task_id)
        if task is None or task.status in FINISHED:
            return False
        if task.status == QUEUED:
            self.fail(task_id, "Cancelled before it started.", status=CANCELLED)
        else:
            self._write("UPDATE tasks SET cancel_requested = 1 WHERE id = ?", (task_id,))
        self._write(
            "UPDATE approvals SET status = ?, answered_at = ?, answered_by = 'cancel' "
            "WHERE task_id = ? AND status = ?",
            (DENIED, _iso(_now()), task_id, PENDING),
        )
        return True

    def cancel_requested(self, task_id: int) -> bool:
        task = self.task(task_id)
        return task is None or task.cancel_requested

    def retry(self, task_id: int) -> bool:
        task = self.task(task_id)
        if task is None or task.status not in FINISHED:
            return False
        self._write(
            "UPDATE tasks SET status = ?, error = NULL, result = NULL, cancel_requested = 0, "
            "started_at = NULL, finished_at = NULL, progress = '' WHERE id = ?",
            (QUEUED, task_id),
        )
        return True

    def recover_interrupted(self) -> list[int]:
        """Tasks left running by a process that quit cannot be resumed safely.

        A tool may have half finished, so they are marked interrupted for the user to
        retry rather than silently run twice.
        """
        rows = self._read(
            "UPDATE tasks SET status = ?, error = ?, finished_at = ? "
            "WHERE status IN (?, ?) RETURNING id",
            (
                INTERRUPTED,
                "Jamie quit while this was running. Retry it if it should run again.",
                _iso(_now()),
                RUNNING,
                WAITING,
            ),
        )
        self._write(
            "UPDATE approvals SET status = ? WHERE status = ? AND task_id IS NOT NULL",
            (EXPIRED, PENDING),
        )
        return [row["id"] for row in rows]

    # -- approvals --------------------------------------------------------------------

    def add_approval(
        self,
        *,
        task_id: int | None,
        tool: str,
        permission: str,
        digest: str,
        summary: str,
        attachment: str | None = None,
        lifetime: timedelta = APPROVAL_LIFETIME,
    ) -> str:
        approval_id = uuid4().hex[:8]
        now = _now()
        self._write(
            "INSERT INTO approvals(id, task_id, tool, permission, arguments_digest, summary, "
            "attachment, status, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                approval_id,
                task_id,
                tool,
                permission,
                digest,
                summary[:MAX_TEXT],
                attachment,
                PENDING,
                _iso(now),
                _iso(now + lifetime),
            ),
        )
        return approval_id

    def approval(self, approval_id: str) -> PendingApproval | None:
        rows = self._read("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        return _approval(rows[0]) if rows else None

    def pending_approvals(self) -> list[PendingApproval]:
        self._expire()
        rows = self._read(
            "SELECT * FROM approvals WHERE status = ? ORDER BY created_at", (PENDING,)
        )
        return [_approval(row) for row in rows]

    def answer(self, approval_id: str, approved: bool, *, by: str = "user") -> bool:
        """Record the user's answer once; False if unknown, expired, or already answered."""
        self._expire()
        cursor = self._write(
            "UPDATE approvals SET status = ?, answered_at = ?, answered_by = ? "
            "WHERE id = ? AND status = ?",
            (APPROVED if approved else DENIED, _iso(_now()), by, approval_id, PENDING),
        )
        return cursor.rowcount == 1

    def _expire(self) -> None:
        self._write(
            "UPDATE approvals SET status = ? WHERE status = ? AND expires_at <= ?",
            (EXPIRED, PENDING, _iso(_now())),
        )
