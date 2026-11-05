"""The window's view of background work: what is running, and what needs the user's OK.

Approvals asked for by background tasks wait in the shared store; the window shows them
as cards (with the review screenshot for job applications) and records the click.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from assistant.background import BackgroundStore
from assistant.background.store import ACTIVE, FINISHED

POLL_SECONDS = 2.0
RECENT = timedelta(hours=12)
MAX_RECENT = 5
MAX_IMAGE_BYTES = 4_000_000
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

type Payload = dict[str, Any]


def image_data_url(path: str | None) -> str | None:
    """The page may only show data: images (see its Content-Security-Policy)."""
    if not path:
        return None
    file = Path(path)
    kind = IMAGE_TYPES.get(file.suffix.lower())
    try:
        if kind is None or file.stat().st_size > MAX_IMAGE_BYTES:
            return None
        data = base64.b64encode(file.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{kind};base64,{data}"


def _recent(finished_at: str | None, now: datetime) -> bool:
    if not finished_at:
        return False
    try:
        return now - datetime.fromisoformat(finished_at) <= RECENT
    except ValueError:
        return False


class BackgroundBridge:
    """Push changes to the page; answer approvals and cancel tasks for it."""

    def __init__(self, store: BackgroundStore, push: Callable[[Payload], object]) -> None:
        self._store = store
        self._push = push
        self._last: Payload | None = None
        self._images: dict[str, str | None] = {}
        self._dismissed: set[int] = set()
        self._lock = threading.Lock()

    def _image(self, path: str | None) -> str | None:
        if not path:
            return None
        if path not in self._images:
            self._images[path] = image_data_url(path)
        return self._images[path]

    def snapshot(self, now: datetime | None = None) -> Payload:
        now = now or datetime.now(UTC)
        tasks = self._store.tasks(limit=40)
        active = [t for t in tasks if t.status in ACTIVE]
        recent = [
            t
            for t in tasks
            if t.status in FINISHED and t.id not in self._dismissed and _recent(t.finished_at, now)
        ][:MAX_RECENT]

        def task_view(task) -> Payload:
            payload = task.payload
            return {
                "id": task.id,
                "kind": task.kind,
                "status": task.status,
                "title": task.prompt if len(task.prompt) <= 90 else f"{task.prompt[:89]}…",
                "progress": task.progress,
                "result": task.result or task.error or "",
                "url": payload.get("url"),
                "confirmation": self._image(payload.get("confirmation_screenshot")),
            }

        approvals = [
            {
                "id": approval.id,
                "task": approval.task_id,
                "tool": approval.tool,
                "summary": approval.summary,
                "image": self._image(approval.attachment),
            }
            for approval in self._store.pending_approvals()
        ]
        return {
            "type": "background",
            "active": [task_view(t) for t in reversed(active)],
            "recent": [task_view(t) for t in recent],
            "approvals": approvals,
        }

    def refresh(self) -> None:
        with self._lock:
            payload = self.snapshot()
            if payload == self._last:
                return
            self._last = payload
        self._push(payload)

    def answer(self, approval_id: str, approved: bool) -> bool:
        answered = self._store.answer(str(approval_id), bool(approved), by="window")
        self.refresh()
        return answered

    def cancel(self, task_id: int) -> bool:
        cancelled = self._store.request_cancel(int(task_id))
        self.refresh()
        return cancelled

    def dismiss(self, task_id: int) -> bool:
        with self._lock:
            self._dismissed.add(int(task_id))
        self.refresh()
        return True


async def poll_background(bridge: BackgroundBridge, interval: float = POLL_SECONDS) -> None:
    while True:
        try:
            await asyncio.to_thread(bridge.refresh)
        except Exception as exc:  # noqa: BLE001 - keep polling; the store may recover
            print(f"Jamie: background view unavailable: {exc}", file=sys.stderr)
        await asyncio.sleep(interval)


ROUTE_LABELS = {
    "chat": "answering directly",
    "files": "files on this Mac",
    "personal": "email, calendar, and reminders",
    "web": "the web",
    "browser": "the browser",
    "macos": "Mac apps",
    "jobs": "job applications",
    "tasks": "Jamie's own tasks",
    "all": "all tools",
}


def route_label(decision) -> str:
    where = " + ".join(ROUTE_LABELS.get(s, s) for s in decision.scopes)
    if decision.timing == "background":
        return f"In the background · {where}"
    if decision.timing == "schedule":
        return "Scheduling it"
    return where[:1].upper() + where[1:]
