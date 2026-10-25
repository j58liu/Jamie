import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import Connectors
from assistant.connectors.fakes import FakeGmail
from assistant.connectors.google.auth import CALENDAR_EVENTS, CALENDAR_FREEBUSY
from assistant.connectors.google.calendar import CalendarConnector, event_id_for
from assistant.llm.base import LLMResponse
from assistant.tasks import ApprovalKind, ApprovalStatus, StepStatus, TaskState
from assistant.tasks.approvals import approve
from assistant.tasks.engine import TaskEngine
from assistant.tasks.recovery import recover
from assistant.tasks.store import TaskStore
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools

TZ = ZoneInfo("America/Toronto")
EVENTS = "/calendar/v3/calendars/primary/events"
EVENT_ARGS = {
    "title": "Sync",
    "start": "2026-10-27T14:00:00-04:00",
    "end": "2026-10-27T14:30:00-04:00",
}


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 25, 14, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "tasks.sqlite3", clock=Clock())


def crashed(store, tool, args):
    """A task whose worker died after recording a step but before its result."""
    task = store.create_task("t", "x")
    store.claim("dead-worker", lease_seconds=30)
    step = store.start_step(task, tool, args)
    store._clock.now += timedelta(minutes=5)
    return task, step


def test_interrupted_read_is_retried(store):
    task, step = crashed(store, "gmail_search_messages", {"query": "alex"})
    report = recover(store)
    assert report.released == [task.id] and report.retried == [step.id]
    assert store.steps(task.id) == []
    assert store.get_task(task.id).state is TaskState.READY


def test_interrupted_calendar_create_reconciles_by_event_id(store, google_api):
    calendar = CalendarConnector(google_api.session([CALENDAR_EVENTS, CALENDAR_FREEBUSY]), TZ)
    task, step = crashed(store, "calendar_create_event", EVENT_ARGS)
    event_id = event_id_for(step.idempotency_key)
    google_api.on(
        "GET",
        f"{EVENTS}/{event_id}",
        {
            "id": event_id,
            "summary": "Sync",
            "start": {"dateTime": "2026-10-27T14:00:00-04:00"},
            "end": {"dateTime": "2026-10-27T14:30:00-04:00"},
        },
    )
    report = recover(store, Connectors(tz=TZ, calendar=calendar))
    assert report.reconciled == [step.id]
    assert store.get_step(step.id).status is StepStatus.SUCCEEDED
    assert not google_api.sent("POST", EVENTS)  # nothing was created twice


def test_calendar_create_retry_with_same_key_returns_first_event(google_api):
    calendar = CalendarConnector(google_api.session([CALENDAR_EVENTS, CALENDAR_FREEBUSY]), TZ)
    key = "task:1:3"
    google_api.on("GET", EVENTS, {"items": []})
    google_api.on("POST", EVENTS, (409, {"error": {"message": "duplicate"}}))
    google_api.on(
        "GET",
        f"{EVENTS}/{event_id_for(key)}",
        lambda request: (
            (404, {})
            if not google_api.sent("POST", EVENTS)
            else (
                200,
                {
                    "id": event_id_for(key),
                    "summary": "Sync",
                    "start": {"dateTime": "2026-10-27T14:00:00-04:00"},
                    "end": {"dateTime": "2026-10-27T14:30:00-04:00"},
                },
            )
        ),
    )
    event = calendar.create_event(
        "Sync",
        datetime(2026, 10, 27, 14, tzinfo=TZ),
        datetime(2026, 10, 27, 14, 30, tzinfo=TZ),
        operation_key=key,
    )
    assert event.duplicate is True and event.id == event_id_for(key)
    sent = json.loads(google_api.sent("POST", EVENTS)[0].content)
    assert sent["id"] == event_id_for(key)


def test_unfound_draft_is_retried_with_the_same_operation_key(store):
    gmail = FakeGmail([])
    task, step = crashed(store, "gmail_create_draft", {"to": ["a@x.com"], "body": "hi"})
    report = recover(store, Connectors(tz=TZ, gmail=gmail))
    assert report.retried == [step.id]
    again = store.start_step(store.get_task(task.id), "gmail_create_draft", {})
    assert again.idempotency_key == step.idempotency_key


def test_interrupted_send_is_ambiguous_and_the_user_decides(store):
    gmail = FakeGmail([])
    gmail.create_draft(["alex@example.com"], "Hi", "Body")
    task, step = crashed(store, "gmail_send_draft", {"draft_id": "draft-1"})
    report = recover(store, Connectors(tz=TZ, gmail=gmail))
    assert report.ambiguous == [step.id]
    assert store.get_task(task.id).state is TaskState.WAITING_APPROVAL
    [question] = store.approvals(task_id=task.id, status=ApprovalStatus.PENDING)
    assert question.kind is ApprovalKind.RETRY
    assert gmail.sent == []

    approve(store, question.id)
    engine = TaskEngine(
        store,
        backend_factory=lambda: type(
            "M",
            (),
            {
                "chat": lambda self, m, **k: LLMResponse(
                    json.dumps({"action": "finish", "message": "Sent."})
                )
            },
        )(),
        registry_factory=lambda task, approvals: register_personal_tools(
            ToolRegistry(approvals=approvals), Connectors(tz=TZ, gmail=gmail)
        ),
    )
    done = engine.run_next()
    assert done.state is TaskState.COMPLETED
    assert gmail.sent == ["draft-1"]  # sent exactly once, after the user said yes
