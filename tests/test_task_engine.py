import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import Connectors
from assistant.connectors.fakes import FakeCalendar, FakeGmail
from assistant.llm.base import LLMError, LLMResponse
from assistant.tasks import ApprovalStatus, StepStatus, TaskState
from assistant.tasks.approvals import approve, reject
from assistant.tasks.engine import TaskEngine, rebuild_messages
from assistant.tasks.notifications import RecordingNotifier
from assistant.tasks.store import TaskStore
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 10, 25, 10, tzinfo=TZ)
REPLY = "Hi Alex, Tuesday at 2pm works for me."


def tool(name, **args):
    return json.dumps({"action": "tool", "tool": name, "args": args})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.requests.append(list(messages))
        if not self.responses:
            raise LLMError("script exhausted")
        return LLMResponse(self.responses.pop(0))


@pytest.fixture
def world(tmp_path):
    gmail = FakeGmail(
        [
            {
                "id": "m1",
                "thread_id": "t1",
                "sender": "Alex <alex@example.com>",
                "subject": "Meeting next week",
                "snippet": "Tuesday 2pm?",
                "received_at": "2026-10-23T14:30:00Z",
                "body": "Could we meet Tuesday at 2pm?",
            }
        ]
    )
    calendar = FakeCalendar([], TZ)
    connectors = Connectors(tz=TZ, gmail=gmail, calendar=calendar)
    store = TaskStore(tmp_path / "tasks.sqlite3", clock=lambda: datetime.now(UTC))
    notifier = RecordingNotifier()

    def engine(model):
        def registry_factory(task, approvals):
            return register_personal_tools(
                ToolRegistry(approvals=approvals), connectors, now=lambda: NOW
            )

        return TaskEngine(
            store,
            backend_factory=lambda: model,
            registry_factory=registry_factory,
            notifier=notifier,
        )

    return store, gmail, notifier, engine


SCRIPT = [
    tool("gmail_search_messages", query="from:alex meeting"),
    tool("gmail_get_message", message_id="m1"),
    tool(
        "calendar_check_availability",
        start="2026-10-27T14:00:00-04:00",
        end="2026-10-27T14:30:00-04:00",
    ),
    tool("gmail_create_draft", to=["alex@example.com"], body=REPLY, reply_to_message_id="m1"),
    json.dumps({"action": "finish", "message": "Saved a reply draft for Alex."}),
]


def test_task_pauses_for_approval_then_resumes_with_the_stored_action(world):
    store, gmail, notifier, engine = world
    task = store.create_task("Reply to Alex", "Find Alex's email and prepare a reply.")
    model = ScriptedModel(SCRIPT)

    paused = engine(model).run_next()
    assert paused.state is TaskState.WAITING_APPROVAL
    assert gmail.drafts == {}
    assert notifier.messages[-1][0] == "Jamie needs approval"
    [pending] = store.approvals(task_id=task.id, status=ApprovalStatus.PENDING)
    assert pending.tool == "gmail_create_draft"
    statuses = [step.status for step in store.steps(task.id)]
    assert statuses == [StepStatus.SUCCEEDED] * 3 + [StepStatus.WAITING]

    approve(store, pending.id)
    done = engine(model).run_next()
    assert done.state is TaskState.COMPLETED
    assert done.result_summary == "Saved a reply draft for Alex."
    [draft] = gmail.drafts.values()
    assert draft.body == REPLY and draft.thread_id == "t1"
    # The model resumed with every earlier step, including the approved draft.
    resumed = model.requests[-1]
    observations = [m.content for m in resumed if m.content.startswith("Tool observation:")]
    assert len(observations) == 4 and "draft-1" in observations[-1]
    assert store.get_approval(pending.id).status is ApprovalStatus.USED
    assert notifier.messages[-1][0] == "Jamie finished a task"


def test_rejection_cancels_without_running_the_action(world):
    store, gmail, _, engine = world
    task = store.create_task("Reply to Alex", "reply")
    engine(ScriptedModel(SCRIPT)).run_next()
    [pending] = store.approvals(task_id=task.id, status=ApprovalStatus.PENDING)
    reject(store, pending.id)
    assert store.get_task(task.id).state is TaskState.CANCELLED
    assert engine(ScriptedModel([])).run_next() is None
    assert gmail.drafts == {}


def test_rebuilt_history_matches_stored_steps(world):
    store, _, _, engine = world
    task = store.create_task("Reply", "reply")
    engine(ScriptedModel(SCRIPT)).run_next()
    messages = rebuild_messages(store, store.get_task(task.id))
    assert len(messages) == 6  # three finished steps; the waiting draft is not replayed
    assert json.loads(messages[0].content)["tool"] == "gmail_search_messages"


def test_model_errors_retry_with_backoff_then_fail(world):
    store, _, notifier, engine = world
    task = store.create_task("t", "x")
    retry = engine(ScriptedModel([])).run_next()
    assert retry.state is TaskState.READY and retry.attempt_count == 1
    assert retry.scheduled_at is not None and store.count_runnable() == 0
    store.transition(task.id, TaskState.RUNNING)
    store.transition(task.id, TaskState.READY, scheduled_at=None, attempt_count=2)
    failed = engine(ScriptedModel([])).run_next()
    assert failed.state is TaskState.FAILED and "script exhausted" in failed.last_error
    assert notifier.messages[-1][0] == "Jamie task failed"
