"""Recover interrupted work without duplicating side effects.

A step left ``started`` means Jamie stopped between recording an operation and
recording its outcome. Reads are re-run. Operations with a way to look up an earlier
attempt are reconciled. Anything else is marked ambiguous and the user decides;
exactly-once delivery is never assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.connectors.base import to_json
from assistant.connectors.errors import ConnectorError
from assistant.security.policy import DEFAULT_TOOL_PERMISSIONS, PermissionLevel
from assistant.tasks.approvals import fingerprint, request_decision
from assistant.tasks.models import ApprovalKind, StepStatus, TaskState
from assistant.tasks.store import TaskStore


def _calendar_event(connectors, args: dict, key: str) -> str | None:
    found = connectors.calendar.find_by_operation(key) if connectors.calendar else None
    return to_json(found) if found is not None else None


def _gmail_draft(connectors, args: dict, key: str) -> str | None:
    found = connectors.gmail.find_draft_by_operation(key) if connectors.gmail else None
    return to_json(found) if found is not None else None


def _reminder(connectors, args: dict, key: str) -> str | None:
    found = connectors.reminders.find(args.get("title", "")) if connectors.reminders else None
    return to_json(found) if found is not None else None


RECONCILERS: dict[str, Callable[[object, dict, str], str | None]] = {
    "calendar_create_event": _calendar_event,
    "gmail_create_draft": _gmail_draft,
    "reminders_create": _reminder,
}


@dataclass
class RecoveryReport:
    released: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)
    reconciled: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)


def recover(store: TaskStore, connectors=None, permissions=DEFAULT_TOOL_PERMISSIONS
            ) -> RecoveryReport:  # fmt: skip
    report = RecoveryReport(released=store.release_expired_leases())
    for step in store.started_steps():
        task = store.get_task(step.task_id)
        if task.state is not TaskState.READY:
            continue  # an earlier step of this task already needs the user
        args = store.load_payload(step.args_payload_id) or {}
        if permissions.get(step.tool_name) is PermissionLevel.READ:
            store.discard_step(step.id)
            report.retried.append(step.id)
            continue
        reconciler = RECONCILERS.get(step.tool_name)
        if reconciler is not None and connectors is not None:
            try:
                found = reconciler(connectors, args, step.idempotency_key)
            except ConnectorError:
                found = None
                reconciler = None  # could not check; let the user decide
            else:
                if found is not None:
                    store.finish_step(step.id, StepStatus.SUCCEEDED,
                                      {"success": True, "output": found})  # fmt: skip
                    report.reconciled.append(step.id)
                else:
                    store.discard_step(step.id)  # it never happened; the retry reuses the key
                    report.retried.append(step.id)
                continue
        store.finish_step(step.id, StepStatus.AMBIGUOUS)
        with store.transaction() as connection:
            request_decision(
                store,
                task,
                kind=ApprovalKind.RETRY,
                tool=step.tool_name,
                display={
                    "tool": step.tool_name,
                    "arguments": args,
                    "note": "Jamie was interrupted while doing this and cannot tell whether "
                    "it finished. Approve to run it again, or reject to cancel the task.",
                },
                fingerprint_value=fingerprint(step.tool_name, args, task.id, task.revision),
                step_id=step.id,
                connection=connection,
            )
            store.transition(task.id, TaskState.WAITING_APPROVAL, connection=connection)
        report.ambiguous.append(step.id)
    return report
