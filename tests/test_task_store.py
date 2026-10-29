from datetime import UTC, datetime, timedelta

import pytest

from assistant.tasks import InvalidTransition, StepStatus, TaskState
from assistant.tasks.store import StaleTaskError, TaskStore


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 25, 14, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return TaskStore(tmp_path / "tasks.sqlite3", clock=clock)


def test_new_database_is_private_and_versioned(tmp_path, store):
    assert oct(store.path.stat().st_mode & 0o777) == "0o600"
    TaskStore(store.path)  # reopening an existing schema is a no-op


def test_transitions_follow_the_state_machine(store):
    task = store.create_task("t", "do it")
    assert task.state is TaskState.READY and task.revision == 1
    with pytest.raises(InvalidTransition):
        store.transition(task.id, TaskState.COMPLETED)
    running = store.transition(task.id, TaskState.RUNNING)
    done = store.transition(running.id, TaskState.COMPLETED, result_summary="ok")
    assert done.result_summary == "ok"
    with pytest.raises(InvalidTransition):
        store.transition(task.id, TaskState.READY)


def test_stale_expected_state_loses(store):
    task = store.create_task("t", "do it")
    store.transition(task.id, TaskState.CANCELLED)
    with pytest.raises(StaleTaskError):
        store.transition(task.id, TaskState.RUNNING, expected=TaskState.READY)


def test_only_one_worker_can_claim_a_task(tmp_path, clock):
    first = TaskStore(tmp_path / "tasks.sqlite3", clock=clock)
    second = TaskStore(tmp_path / "tasks.sqlite3", clock=clock)
    first.create_task("t", "do it")
    claimed = first.claim("worker-a")
    assert claimed.lease_owner == "worker-a" and claimed.state is TaskState.RUNNING
    assert second.claim("worker-b") is None


def test_claim_prefers_priority_and_respects_start_time(store, clock):
    later = store.create_task("later", "x", scheduled_at=clock.now + timedelta(hours=1))
    low = store.create_task("low", "x")
    high = store.create_task("high", "x", priority=5)
    assert store.claim("w").id == high.id
    assert store.claim("w").id == low.id
    assert store.claim("w") is None
    assert store.count_runnable() == 0
    clock.now += timedelta(hours=2)
    assert store.claim("w").id == later.id


def test_expired_leases_return_to_ready_and_heartbeats_extend(store, clock):
    task = store.create_task("t", "x")
    store.claim("w", lease_seconds=60)
    clock.now += timedelta(seconds=50)
    assert store.heartbeat(task.id, "w", 60)
    clock.now += timedelta(seconds=50)
    assert store.release_expired_leases() == []
    assert not store.heartbeat(task.id, "other", 60)
    clock.now += timedelta(seconds=120)
    assert store.release_expired_leases() == [task.id]
    assert store.get_task(task.id).state is TaskState.READY


def test_steps_keep_redacted_summaries_and_payloads_vanish_when_done(store):
    task = store.create_task("t", "x")
    step = store.start_step(task, "gmail_create_draft", {"to": ["a@x.com"], "body": "secret"})
    assert step.idempotency_key == f"{task.id}:1:0" and step.status is StepStatus.STARTED
    assert "secret" not in step.summary
    store.finish_step(step.id, StepStatus.SUCCEEDED, {"success": True, "output": "ok"})
    assert store.load_payload(store.get_step(step.id).args_payload_id)["body"] == "secret"
    second = store.start_step(task, "gmail_get_draft", {"draft_id": "d"})
    assert second.step_index == 1
    store.transition(task.id, TaskState.RUNNING)
    store.transition(task.id, TaskState.COMPLETED)
    assert store.load_payload(step.args_payload_id) is None
    assert store.get_step(step.id).args_payload_id is None


def test_revise_bumps_revision_and_requeues(store):
    task = store.create_task("t", "x")
    store.transition(task.id, TaskState.RUNNING)
    with pytest.raises(StaleTaskError):
        store.revise(task.id, "y")
    store.transition(task.id, TaskState.WAITING_APPROVAL)
    revised = store.revise(task.id, "y")
    assert (revised.instruction, revised.revision, revised.state) == ("y", 2, TaskState.READY)
