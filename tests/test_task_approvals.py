from datetime import UTC, datetime, timedelta

import pytest

from assistant.security import ApprovalRequest, PermissionLevel
from assistant.tasks import ApprovalStatus, TaskState
from assistant.tasks.approvals import (
    ApprovalError,
    DeferredApprovalProvider,
    approve,
    expire_due,
    fingerprint,
    reject,
)
from assistant.tasks.store import TaskStore


class Clock:
    now = datetime(2026, 10, 25, 14, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "tasks.sqlite3", clock=Clock())


def request(body="Tuesday works.", to="alex@example.com"):
    return ApprovalRequest(
        tool="gmail_send_draft",
        permission=PermissionLevel.SEND,
        arguments={"to": [to], "body": body},
    )


def waiting_task(store):
    task = store.create_task("Reply to Alex", "reply")
    store.transition(task.id, TaskState.RUNNING)
    return task


def test_fingerprint_binds_tool_arguments_account_and_revision():
    base = fingerprint("gmail_send_draft", {"to": ["a@x.com"], "body": "hi"}, "t1", 1)
    assert base == fingerprint("gmail_send_draft", {"body": "hi", "to": ["a@x.com"]}, "t1", 1)
    assert base != fingerprint("gmail_send_draft", {"to": ["b@x.com"], "body": "hi"}, "t1", 1)
    assert base != fingerprint("gmail_send_draft", {"to": ["a@x.com"], "body": "hey"}, "t1", 1)
    assert base != fingerprint("gmail_send_draft", {"to": ["a@x.com"], "body": "hi"}, "t1", 2)
    assert base != fingerprint("gmail_create_draft", {"to": ["a@x.com"], "body": "hi"}, "t1", 1)


def test_deferred_then_approved_exact_operation_runs_once(store):
    task = waiting_task(store)
    provider = DeferredApprovalProvider(store, task)
    assert provider.request(request()) is False
    pending = provider.deferred
    assert pending.status is ApprovalStatus.PENDING
    store.transition(task.id, TaskState.WAITING_APPROVAL)
    approve(store, pending.id)
    assert store.get_task(task.id).state is TaskState.READY

    resumed = DeferredApprovalProvider(store, store.get_task(task.id))
    assert resumed.request(request(body="Changed text")) is False  # edited content: no match
    assert resumed.request(request()) is True
    assert resumed.request(request()) is False  # used once only


def test_editing_the_task_supersedes_the_approval(store):
    task = waiting_task(store)
    provider = DeferredApprovalProvider(store, task)
    provider.request(request())
    store.transition(task.id, TaskState.WAITING_APPROVAL)
    store.revise(task.id, "reply differently")
    assert store.get_approval(provider.deferred.id).status is ApprovalStatus.SUPERSEDED
    with pytest.raises(ApprovalError):
        approve(store, provider.deferred.id)


def test_reject_cancels_and_expiry_fails(store):
    first = waiting_task(store)
    provider = DeferredApprovalProvider(store, first)
    provider.request(request())
    store.transition(first.id, TaskState.WAITING_APPROVAL)
    reject(store, provider.deferred.id)
    assert store.get_task(first.id).state is TaskState.CANCELLED

    second = waiting_task(store)
    late = DeferredApprovalProvider(store, second, expiry=timedelta(minutes=5))
    late.request(request())
    store.transition(second.id, TaskState.WAITING_APPROVAL)
    store._clock.now = Clock.now + timedelta(minutes=10)
    assert expire_due(store) == [second.id]
    assert store.get_task(second.id).state is TaskState.FAILED
    with pytest.raises(ApprovalError):
        approve(store, late.deferred.id)
