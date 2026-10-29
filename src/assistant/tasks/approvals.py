"""Deferred approvals bound to the exact operation and task revision.

A task that hits an action needing confirmation records a pending approval and
stops. The user approves later, and the task resumes by running exactly the stored
operation. Changing the tool, its arguments, the account, or the task revision
produces a different fingerprint, so an old approval no longer matches.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from assistant.security.approvals import ApprovalRequest
from assistant.tasks.models import (
    ApprovalKind,
    ApprovalStatus,
    TaskApproval,
    TaskState,
)
from assistant.tasks.store import StaleTaskError, TaskStore

DEFAULT_EXPIRY = timedelta(hours=24)

# The account each tool acts as; part of what an approval is bound to.
ACCOUNTS = {"gmail_": "google:default", "calendar_": "google:default",
            "reminders_": "apple:reminders"}  # fmt: skip


class ApprovalError(ValueError):
    """The approval cannot be decided (already decided, expired, or superseded)."""


def account_for(tool: str) -> str:
    return next((account for prefix, account in ACCOUNTS.items() if tool.startswith(prefix)),
                "local")  # fmt: skip


def fingerprint(tool: str, arguments: dict, task_id: str, revision: int) -> str:
    canonical = json.dumps(
        {
            "tool": tool,
            "arguments": arguments,
            "account": account_for(tool),
            "task": task_id,
            "revision": revision,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DeferredApprovalProvider:
    """An ApprovalProvider for unattended runs: approve from a stored decision or defer."""

    def __init__(self, store: TaskStore, task, *, expiry: timedelta = DEFAULT_EXPIRY) -> None:
        self._store = store
        self._task = task
        self._expiry = expiry
        self.current_step_id: str | None = None
        self.deferred: TaskApproval | None = None

    def request(self, approval: ApprovalRequest) -> bool:
        arguments = dict(approval.arguments)
        digest = fingerprint(approval.tool, arguments, self._task.id, self._task.revision)
        now = self._store.now()
        for stored in self._store.approvals(task_id=self._task.id, status=ApprovalStatus.APPROVED):
            if stored.fingerprint != digest or stored.task_revision != self._task.revision:
                continue
            if stored.expires_at is not None and stored.expires_at < now:
                continue
            if self._store.set_approval_status(
                stored.id, ApprovalStatus.USED, expected=ApprovalStatus.APPROVED
            ):
                return True
        self.deferred = self._store.create_approval(
            self._task,
            tool=approval.tool,
            fingerprint=digest,
            display={
                "tool": approval.tool,
                "permission": approval.permission.value,
                "arguments": arguments,
                "account": account_for(approval.tool),
            },
            step_id=self.current_step_id,
            expires_at=now + self._expiry,
        )
        return False


def approval_display(store: TaskStore, approval: TaskApproval) -> dict:
    display = store.load_payload(approval.display_payload_id)
    return display if isinstance(display, dict) else {"tool": approval.tool, "arguments": {}}


def approve(store: TaskStore, approval_id: str) -> TaskApproval:
    """Record the user's yes; the task becomes ready to resume."""
    approval = store.get_approval(approval_id)
    task = store.get_task(approval.task_id)
    if approval.status is not ApprovalStatus.PENDING:
        raise ApprovalError(f"This approval is already {approval.status.value}.")
    if approval.expires_at is not None and approval.expires_at < store.now():
        raise ApprovalError("This approval has expired.")
    if approval.task_revision != task.revision:
        raise ApprovalError("The task changed after this approval was requested.")
    with store.transaction() as connection:
        if not store.set_approval_status(
            approval.id, ApprovalStatus.APPROVED, expected=ApprovalStatus.PENDING,
            connection=connection,
        ):  # fmt: skip
            raise ApprovalError("This approval was decided elsewhere.")
        try:
            store.transition(task.id, TaskState.READY, expected=TaskState.WAITING_APPROVAL,
                             connection=connection)  # fmt: skip
        except StaleTaskError as exc:
            raise ApprovalError(str(exc)) from exc
    return store.get_approval(approval.id)


def reject(store: TaskStore, approval_id: str) -> TaskApproval:
    """Record the user's no; the pending operation never runs and the task is cancelled."""
    approval = store.get_approval(approval_id)
    if approval.status is not ApprovalStatus.PENDING:
        raise ApprovalError(f"This approval is already {approval.status.value}.")
    with store.transaction() as connection:
        store.set_approval_status(
            approval.id, ApprovalStatus.REJECTED, expected=ApprovalStatus.PENDING,
            connection=connection,
        )  # fmt: skip
        task = store.get_task(approval.task_id)
        if task.state is TaskState.WAITING_APPROVAL:
            store.transition(task.id, TaskState.CANCELLED, connection=connection,
                             last_error="The user rejected the pending action.")  # fmt: skip
    return store.get_approval(approval.id)


def expire_due(store: TaskStore) -> list[str]:
    """Fail tasks whose pending approval expired; returns the task ids."""
    expired = []
    now = store.now()
    for approval in store.approvals(status=ApprovalStatus.PENDING):
        if approval.expires_at is None or approval.expires_at >= now:
            continue
        with store.transaction() as connection:
            if not store.set_approval_status(
                approval.id, ApprovalStatus.EXPIRED, expected=ApprovalStatus.PENDING,
                connection=connection,
            ):  # fmt: skip
                continue
            task = store.get_task(approval.task_id)
            if task.state is TaskState.WAITING_APPROVAL:
                store.transition(task.id, TaskState.FAILED, connection=connection,
                                 last_error="The approval request expired.")  # fmt: skip
                expired.append(task.id)
    return expired


def request_decision(
    store: TaskStore, task, *, kind: ApprovalKind, tool: str, display: dict,
    fingerprint_value: str, step_id: str | None = None, connection=None,
) -> TaskApproval:  # fmt: skip
    """Ask the user about something other than a fresh tool call (retry, missed run)."""
    return store.create_approval(
        task, tool=tool, fingerprint=fingerprint_value, display=display, kind=kind,
        step_id=step_id, expires_at=store.now() + DEFAULT_EXPIRY, connection=connection,
    )  # fmt: skip
