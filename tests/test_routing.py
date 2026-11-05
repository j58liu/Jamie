import asyncio
import json

import pytest

from assistant.background import BackgroundStore
from assistant.background.store import CANCELLED
from assistant.config import ProjectConfig
from assistant.jobs import JOB_KIND, queue_applications
from assistant.llm.base import LLMResponse, current_response_schema
from assistant.routing import BACKGROUND, NOW, SCHEDULE, find_urls, route, schedule_request
from assistant.runtime import JamieApplication
from assistant.runtime.dispatch import dispatch
from assistant.runtime.workspaces import WorkspaceTools
from assistant.tools.registry import ToolRegistry, create_workspace_registry
from assistant.tools.tasks import register_task_tools
from assistant.ui.gui import GuiSession


class Classifier:
    """Scores the subject question with ``subject`` and the timing one with ``timing``."""

    def __init__(self, subject=None, timing=None, replies=()):
        self.subject = subject or {}
        self.timing = timing or {}
        self.replies = list(replies)
        self.asked = []
        self.schemas = []

    def option_probabilities(self, messages, options):
        text = messages[-1].content
        self.asked.append("timing" if "When should" in text else "subject")
        scores = self.timing if "When should" in text else self.subject
        return {option: scores.get(option, 0.0) for option in options}

    def chat(self, messages, **options):
        self.schemas.append(current_response_schema())
        return LLMResponse(json.dumps(self.replies.pop(0)))


def test_clear_subjects_pick_one_scope():
    decision = route(Classifier({"C": 0.9}), "What's on my calendar tomorrow?")
    assert decision.scopes == ("personal",) and decision.timing == NOW


def test_chat_needs_confidence_to_drop_tools():
    assert route(Classifier({"A": 0.95}), "hi there").chat
    unsure = route(Classifier({"A": 0.4, "B": 0.1}), "what about the notes?")
    assert unsure.scopes == ("all",)


def test_an_unsure_subject_adds_the_runner_up():
    decision = route(Classifier({"B": 0.5, "C": 0.4}), "save Alex's attachment to Desktop")
    assert decision.scopes == ("files", "personal")


def test_unavailable_scopes_fall_back_to_all_tools():
    decision = route(Classifier({"E": 0.9}), "book a table", available={"files", "web"})
    assert decision.scopes == ("all",)


def test_without_a_classifier_everything_runs_now_with_all_tools():
    class Plain:
        pass

    decision = route(Plain(), "summarize my week")
    assert decision.timing == NOW and decision.scopes == ("all",) and not decision.scores


def test_background_and_schedule_phrases_win():
    assert route(Classifier({"D": 0.9}), "research GPUs while I work").timing == BACKGROUND
    assert route(Classifier({"C": 0.9}), "every morning summarize my email").timing == SCHEDULE
    # A reminder for the user is a Reminders task now, not a job for Jamie later.
    assert route(Classifier({"C": 0.9}), "remind me every day to stretch").timing == NOW


def test_the_model_decides_timing_only_with_a_clear_signal():
    sure = Classifier({"D": 0.9}, {"B": 0.8})
    assert route(sure, "compare these laptops for me").timing == BACKGROUND
    unsure = Classifier({"D": 0.9}, {"B": 0.4, "A": 0.3})
    assert route(unsure, "compare these laptops for me").timing == NOW
    chat = Classifier({"A": 0.9})
    route(chat, "hello")
    assert chat.asked == ["subject"]  # small talk never pays for the timing question


def test_job_requests_go_to_the_background_with_their_links():
    classifier = Classifier()
    decision = route(
        classifier,
        "Apply to these jobs: https://jobs.lever.co/acme/123, "
        "https://boards.greenhouse.io/x/jobs/9. Go ahead and submit them yourself.",
    )
    assert decision.apply and decision.timing == BACKGROUND
    assert decision.urls == (
        "https://jobs.lever.co/acme/123",
        "https://boards.greenhouse.io/x/jobs/9",
    )
    assert decision.auto_submit
    assert classifier.asked == []  # explicit requests need no model call
    review = route(Classifier(), "please apply for this role https://jobs.ashbyhq.com/a/b")
    assert not review.auto_submit


def test_find_urls_trims_punctuation_and_duplicates():
    assert find_urls("see https://a.com/x. and (https://a.com/x)") == ("https://a.com/x",)


def test_schedule_requests_are_filled_by_the_model_and_validated():
    backend = Classifier(
        replies=[
            {"repeat": "weekly", "day": "mon", "time": "09:00", "date": "", "interval": "",
             "task": "Summarize the news I should know about."},
        ]
    )  # fmt: skip
    request = schedule_request(backend, "every monday morning, summarize the news")
    assert (request.repeat, request.time) == ("weekly", "mon 09:00")
    assert request.prompt == "Summarize the news I should know about."
    assert backend.schemas[0]["required"]

    bad = Classifier(
        replies=[{"repeat": "every", "day": "", "time": "09:00", "date": "", "interval": "1m",
                  "task": "x"}]
    )  # fmt: skip
    with pytest.raises(ValueError, match="5 minutes"):
        schedule_request(bad, "every minute check")


@pytest.fixture
def started(tmp_path):
    def build(backend):
        application = JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: backend,
        )
        asyncio.run(application.start())
        return application, BackgroundStore(tmp_path / "background.sqlite3")

    return build


def test_dispatch_queues_background_work_with_its_tools(started):
    application, store = started(Classifier())
    decision = route(Classifier({"D": 0.9}), "find cheap flights to Tokyo while I work")
    reply = asyncio.run(dispatch(application, store, "find flights", decision, source="test"))
    task = store.tasks()[0]
    assert "background" in reply and str(task.id) in reply
    assert task.payload == {"scopes": ["web"]} and task.source == "test"


def test_dispatch_queues_one_application_per_link(started):
    application, store = started(Classifier())
    decision = route(Classifier(), "apply to these jobs https://a.co/1 https://b.co/2")
    reply = asyncio.run(dispatch(application, store, "apply", decision, source="test"))
    tasks = sorted(store.tasks(), key=lambda t: t.id)
    assert [t.kind for t in tasks] == [JOB_KIND, JOB_KIND]
    assert [t.payload["url"] for t in tasks] == ["https://a.co/1", "https://b.co/2"]
    assert all(t.payload["submit"] == "review" for t in tasks)
    assert "approve" in reply


def test_dispatch_searches_for_jobs_when_no_links_are_given(started):
    application, store = started(Classifier())
    decision = route(Classifier(), "apply to remote python jobs for me")
    reply = asyncio.run(dispatch(application, store, "remote python jobs", decision, source="t"))
    task = store.tasks()[0]
    assert task.kind == "agent" and "remote python jobs" in task.prompt
    assert "which ones" in reply


def test_dispatch_schedules_from_plain_words(started):
    backend = Classifier(
        replies=[{"repeat": "daily", "day": "", "time": "08:00", "date": "", "interval": "",
                  "task": "Summarize my unread email."}]
    )  # fmt: skip
    application, store = started(backend)
    decision = route(Classifier({"C": 0.9}), "every morning at 8 summarize my unread email")
    reply = asyncio.run(dispatch(application, store, "every morning...", decision, source="t"))
    [task] = application.memory.schedules()
    assert (task.repeat, task.time, task.prompt) == ("daily", "08:00", "Summarize my unread email.")
    assert "every day at 08:00" in reply


def test_dispatch_leaves_requests_for_now_to_the_agent(started):
    application, store = started(Classifier())
    decision = route(Classifier({"B": 0.9}), "find my resume")
    assert asyncio.run(dispatch(application, store, "find my resume", decision, source="t")) is None
    assert store.tasks() == []


def test_task_tools_list_start_and_stop_work(started):
    application, store = started(Classifier())
    registry = register_task_tools(ToolRegistry(), store, lambda: application.memory)
    first = registry.execute("jamie_start_background", {"task": "Find a dentist near me"})
    assert first.success and "task 1" in first.output
    queue_applications(store, ["https://jobs.lever.co/a/1"])
    status = json.loads(registry.execute("jamie_work_status", {}).output)
    assert [t["kind"] for t in status["background"]] == [JOB_KIND, "agent"]
    assert registry.execute("jamie_cancel_work", {"kind": "task", "id": 1}).success
    assert store.task(1).status == CANCELLED
    assert not registry.execute("jamie_cancel_work", {"kind": "task", "id": 1}).success

    scheduled = registry.execute("jamie_schedule", {"when": "daily 07:30", "task": "Check news"})
    assert scheduled.success
    status = json.loads(registry.execute("jamie_work_status", {}).output)
    assert status["scheduled"][0]["when"] == "every day at 07:30"
    schedule_id = status["scheduled"][0]["id"]
    assert registry.execute("jamie_cancel_work", {"kind": "schedule", "id": schedule_id}).success
    assert application.memory.schedules() == []
    bad = registry.execute("jamie_schedule", {"when": "sometime", "task": "x"})
    assert not bad.success


def test_the_window_queues_background_work_without_a_model_turn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = Classifier({"D": 0.9})
    application = JamieApplication(
        ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
        backend_factory=lambda: backend,
    )
    asyncio.run(application.start())
    store = BackgroundStore(tmp_path / "background.sqlite3")
    pushed = []
    tools = WorkspaceTools(lambda: create_workspace_registry(tmp_path))
    session = GuiSession(application, tools, pushed.append, asyncio.run, background=store)

    assert session.send("research the best standing desks while I work")
    types = [p["type"] for p in pushed]
    assert types == ["route", "answer", "idle"]
    assert pushed[0]["text"].startswith("In the background")
    assert store.tasks()[0].prompt == "research the best standing desks while I work"
    assert session.state.history[-1].content == pushed[1]["text"]
    assert backend.replies == []  # no generation happened


def test_background_snapshot_shows_approvals_and_recent_results(tmp_path):
    from assistant.ui.gui.work import BackgroundBridge

    store = BackgroundStore(tmp_path / "background.sqlite3")
    done = store.add_task("summarize downloads")
    store.claim_next()
    store.finish(done, "Three PDFs.")
    running = store.add_task("long research")
    store.claim_next()
    image = tmp_path / "review.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    approval = store.add_approval(
        task_id=running, tool="job_submit", permission="SEND", digest="d",
        summary="Submit?", attachment=str(image),
    )  # fmt: skip
    pushed = []
    bridge = BackgroundBridge(store, pushed.append)
    bridge.refresh()
    bridge.refresh()  # unchanged: nothing new is pushed
    assert len(pushed) == 1
    view = pushed[0]
    assert [t["id"] for t in view["active"]] == [running]
    assert view["recent"][0]["result"] == "Three PDFs."
    assert view["approvals"][0]["image"].startswith("data:image/png;base64,")
    assert bridge.answer(approval, True)
    assert pushed[-1]["approvals"] == []
    assert bridge.dismiss(done)
    assert pushed[-1]["recent"] == []
