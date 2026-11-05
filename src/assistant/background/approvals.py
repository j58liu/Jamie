"""Approvals for work that runs while the user does something else.

The task waits; the request is saved, the user is notified, and the answer can come
from the app, the terminal (`jamie bg approve`), or a notification. Nothing runs
without that answer, and an unanswered request expires and counts as "no".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from assistant.background.store import (
    APPROVED,
    PENDING,
    RUNNING,
    WAITING,
    BackgroundStore,
    PendingApproval,
    arguments_digest,
)
from assistant.cancellation import CancellationToken
from assistant.security.approvals import ApprovalRequest

POLL_SECONDS = 1.0

type Notify = Callable[[PendingApproval], object]


@dataclass(frozen=True)
class Grant:
    """A permission the user gave ahead of time, bound to one tool and argument set.

    ``digest`` None allows the tool with any arguments for this task only, which is how
    "let Jamie submit this application by itself" is recorded.
    """

    tool: str
    digest: str | None = None

    def covers(self, tool: str, digest: str) -> bool:
        return tool == self.tool and self.digest in (None, digest)


def describe(approval: ApprovalRequest) -> str:
    from assistant.ui.approvals import summarize_approval

    try:
        summary = summarize_approval(approval)
    except Exception:  # noqa: BLE001 - a summary must never block the request
        return f"Jamie wants to run {approval.tool}."
    lines = [summary.heading]
    if summary.subject is not None:
        lines.append(summary.subject)
    lines += [f"{name}: {value}" for name, value in summary.details]
    lines.append(summary.footer)
    return "\n".join(lines)


class DeferredApprovalProvider:
    """Ask asynchronously; block only the background task's own thread."""

    def __init__(
        self,
        store: BackgroundStore,
        task_id: int | None,
        *,
        notify: Notify | None = None,
        grants: tuple[Grant, ...] = (),
        cancel: CancellationToken | None = None,
        poll_seconds: float = POLL_SECONDS,
        attachment: Callable[[], str | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._task_id = task_id
        self._notify = notify
        self._grants = grants
        self._cancel = cancel
        self._poll = poll_seconds
        self._attachment = attachment
        self._sleep = sleep

    def request(self, approval: ApprovalRequest) -> bool:
        digest = arguments_digest(approval.tool, dict(approval.arguments))
        if any(grant.covers(approval.tool, digest) for grant in self._grants):
            return True
        attachment = self._attachment() if self._attachment is not None else None
        approval_id = self._store.add_approval(
            task_id=self._task_id,
            tool=approval.tool,
            permission=approval.permission.value,
            digest=digest,
            summary=describe(approval),
            attachment=attachment,
        )
        if self._task_id is not None:
            self._store.set_status(self._task_id, WAITING)
        record = self._store.approval(approval_id)
        if self._notify is not None and record is not None:
            try:
                self._notify(record)
            except Exception:  # noqa: BLE001 - the request stays answerable elsewhere
                pass
        try:
            while True:
                pending = [p for p in self._store.pending_approvals() if p.id == approval_id]
                if not pending:
                    answered = self._store.approval(approval_id)
                    # Bound to the exact operation: a changed request is a new approval.
                    return (
                        answered is not None
                        and answered.status == APPROVED
                        and answered.arguments_digest == digest
                    )
                if self._cancel is not None and self._cancel.cancelled:
                    self._store.answer(approval_id, False, by="cancel")
                    return False
                if self._task_id is not None and self._store.cancel_requested(self._task_id):
                    self._store.answer(approval_id, False, by="cancel")
                    return False
                self._sleep(self._poll)
        finally:
            if self._task_id is not None:
                task = self._store.task(self._task_id)
                if task is not None and task.status == WAITING:
                    self._store.set_status(self._task_id, RUNNING)


__all__ = ["PENDING", "DeferredApprovalProvider", "Grant", "describe"]
