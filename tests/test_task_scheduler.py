from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import Connectors
from assistant.connectors.fakes import FakeGmail
from assistant.tasks import ApprovalKind, ApprovalStatus, MissedPolicy, ScheduleKind, TaskState
from assistant.tasks.scheduler import ScheduleError, next_run, once_at, parse_every, tick
from assistant.tasks.store import TaskStore

ZONE = "America/Toronto"
TZ = ZoneInfo(ZONE)


def at(*args):
    return datetime(*args, tzinfo=TZ)


class Clock:
    def __init__(self):
        self.now = at(2026, 10, 25, 7, 0)

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return TaskStore(tmp_path / "tasks.sqlite3", clock=clock)


def test_parse_every():
    assert parse_every("daily@8:00") == (ScheduleKind.DAILY, {"time": "08:00"})
    assert parse_every("weekly:sun@09:30") == (ScheduleKind.WEEKLY, {"weekday": 6, "time": "09:30"})
    assert parse_every("2h") == (ScheduleKind.INTERVAL, {"minutes": 120})
    for bad in ("hourly", "interval:1m", "weekly:xyz@09:00"):
        with pytest.raises(ScheduleError):
            parse_every(bad)


def test_daily_keeps_local_time_across_dst():
    after = at(2026, 10, 31, 9, 0)
    assert (
        next_run(ScheduleKind.DAILY, {"time": "08:00"}, ZONE, after).isoformat()
        == "2026-11-01T08:00:00-05:00"
    )
    before = at(2026, 10, 31, 7, 0)
    assert (
        next_run(ScheduleKind.DAILY, {"time": "08:00"}, ZONE, before).isoformat()
        == "2026-10-31T08:00:00-04:00"
    )


def test_weekly_interval_and_once():
    monday = at(2026, 10, 26, 12, 0)
    weekly = next_run(ScheduleKind.WEEKLY, {"weekday": 6, "time": "09:00"}, ZONE, monday)
    assert weekly.isoformat() == "2026-11-01T09:00:00-05:00"
    interval = next_run(ScheduleKind.INTERVAL, {"minutes": 120}, ZONE, at(2026, 11, 1, 0, 30))
    assert interval.astimezone(UTC) - at(2026, 11, 1, 0, 30).astimezone(UTC) == timedelta(hours=2)
    assert interval.astimezone(TZ).hour == 1  # elapsed time across the fall-back hour
    spec = {"at": at(2026, 10, 26, 8).isoformat()}
    assert next_run(ScheduleKind.ONCE, spec, ZONE, monday) is None
    assert once_at("tomorrow 8am", at(2026, 10, 25, 10)) == at(2026, 10, 26, 8)
    with pytest.raises(ScheduleError):
        once_at("tomorrow", at(2026, 10, 25, 10))


def schedule(store, policy=MissedPolicy.RUN_LATE, first=None):
    return store.create_schedule(
        title="Morning brief",
        instruction="Summarize my day",
        kind=ScheduleKind.DAILY,
        spec={"time": "08:00"},
        timezone=ZONE,
        next_run_at=first or at(2026, 10, 25, 8),
        missed_policy=policy,
    )


def test_on_time_run_creates_one_task_and_advances(store, clock):
    sched = schedule(store)
    assert tick(store, clock.now).created == []
    clock.now = at(2026, 10, 25, 8, 1)
    report = tick(store, clock.now)
    [task] = [store.get_task(task_id) for task_id in report.created]
    assert task.state is TaskState.READY and task.schedule_id == sched.id
    assert store.get_schedule(sched.id).next_run_at == at(2026, 10, 26, 8)


def test_missed_runs_collapse_into_one_late_run(store, clock):
    sched = schedule(store)
    clock.now = at(2026, 10, 28, 10, 0)  # asleep for three mornings
    assert len(tick(store, clock.now).created) == 1
    assert store.get_schedule(sched.id).next_run_at == at(2026, 10, 29, 8)
    assert tick(store, clock.now).created == []


def test_skip_and_ask_policies(store, clock):
    skipped = schedule(store, MissedPolicy.SKIP)
    asked = schedule(store, MissedPolicy.ASK)
    clock.now = at(2026, 10, 25, 10, 0)
    report = tick(store, clock.now)
    assert report.skipped == [skipped.id] and report.created == []
    [task_id] = report.asked
    assert store.get_task(task_id).state is TaskState.WAITING_APPROVAL
    [question] = store.approvals(task_id=task_id, status=ApprovalStatus.PENDING)
    assert question.kind is ApprovalKind.RUN_MISSED
    assert store.get_schedule(asked.id).next_run_at == at(2026, 10, 26, 8)


def test_once_schedule_disables_after_running(store, clock):
    sched = store.create_schedule(
        title="Call",
        instruction="remind",
        kind=ScheduleKind.ONCE,
        spec={"at": at(2026, 10, 25, 7, 30).isoformat()},
        timezone=ZONE,
        next_run_at=at(2026, 10, 25, 7, 30),
    )
    clock.now = at(2026, 10, 25, 7, 31)
    assert len(tick(store, clock.now).created) == 1
    assert not store.get_schedule(sched.id).enabled


def test_event_trigger_only_creates_tasks_for_new_messages(store, clock):
    message = {
        "id": "m1",
        "thread_id": "t1",
        "sender": "alex@example.com",
        "subject": "Invoice",
        "snippet": "",
        "received_at": "",
        "body": "",
    }
    gmail = FakeGmail([message])
    connectors = Connectors(tz=TZ, gmail=gmail)
    sched = store.create_schedule(
        title="Invoices",
        instruction="File new invoices",
        kind=ScheduleKind.EVENT,
        spec={"source": "gmail", "query": "invoice"},
        timezone=ZONE,
        next_run_at=clock.now,
    )
    assert tick(store, clock.now, connectors).created == []  # first poll sets the baseline
    clock.now += timedelta(minutes=6)
    assert tick(store, clock.now, connectors).created == []
    gmail.messages.append(gmail.messages[0].model_copy(update={"id": "m2"}))
    clock.now += timedelta(minutes=6)
    [task_id] = tick(store, clock.now, connectors).created
    assert "m2" in store.get_task(task_id).instruction
    assert "m1" not in store.get_task(task_id).instruction
    assert store.get_schedule(sched.id).enabled
