import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import Connectors
from assistant.connectors.fakes import FailureInjector, FakeCalendar, FakeGmail
from assistant.llm.base import LLMResponse, current_response_schema
from assistant.planning import (
    Plan,
    PlanBudget,
    PlanExecutor,
    PlanningError,
    PlanValidationError,
    StepStatus,
    needs_plan,
    validate_plan,
)
from assistant.security import CallbackApprovalProvider
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 10, 25, 10, tzinfo=TZ)
PROFESSOR = {
    "id": "m1", "thread_id": "t1", "sender": "Prof. Rivera <rivera@uni.edu>",
    "subject": "Thesis next steps", "snippet": "Draft chapter by Nov 3",
    "received_at": "2026-10-24T15:00:00Z",
    "body": "Please send your draft chapter by November 3 and book a progress meeting "
            "before November 10.",
}  # fmt: skip


class Script:
    def __init__(self, *responses):
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.schemas = []

    def chat(self, messages, **options):
        self.schemas.append(current_response_schema())
        if not self.responses:
            raise AssertionError("unexpected model call")
        return LLMResponse(self.responses.pop(0))


class Recorder:
    def __init__(self, registry):
        self.registry, self.calls = registry, []

    def __getattr__(self, name):
        return getattr(self.registry, name)

    def execute(self, tool, args):
        self.calls.append(tool)
        return self.registry.execute(tool, args)


def world(gmail=None, calendar=None, approve=lambda request: True):
    gmail = gmail if gmail is not None else FakeGmail([PROFESSOR])
    calendar = calendar if calendar is not None else FakeCalendar([], TZ)
    approvals = []

    def decide(request):
        approvals.append(request.tool)
        return approve(request)

    registry = register_personal_tools(
        ToolRegistry(approvals=CallbackApprovalProvider(decide)),
        Connectors(tz=TZ, gmail=gmail, calendar=calendar),
        now=lambda: NOW,
    )
    return Recorder(registry), gmail, calendar, approvals


def step(n, tool, depends=(), **extra):
    return {"id": f"step_{n}", "description": f"do {tool}", "tool": tool,
            "depends_on": list(depends), **extra}  # fmt: skip


PLAN = {"goal": "Organize professor follow-up", "steps": [
    step(1, "gmail_search_messages"),
    step(2, "gmail_get_message", [ "step_1"]),
    step(3, "calendar_list_events", ["step_2"]),
    step(4, "calendar_create_event", ["step_2", "step_3"], multiple=True),
    step(5, "gmail_create_draft", ["step_2", "step_4"]),
]}  # fmt: skip
EVENTS = {"calls": [
    {"title": "Draft chapter due", "start": "2026-11-03T09:00:00-05:00",
     "end": "2026-11-03T09:30:00-05:00"},
    {"title": "Book progress meeting", "start": "2026-11-09T09:00:00-05:00",
     "end": "2026-11-09T09:30:00-05:00"},
]}  # fmt: skip
DRAFT = {"to": ["rivera@uni.edu"], "body": "I'll send the chapter by Nov 3.",
         "reply_to_message_id": "m1"}  # fmt: skip


def test_professor_workflow_runs_in_order_and_verifies_writes():
    tools, gmail, calendar, approvals = world()
    model = Script(
        PLAN,
        {"query": "from:rivera"},
        {"message_id": "m1"},
        {"start": "2026-11-01T00:00:00-04:00", "end": "2026-11-11T00:00:00-05:00"},
        EVENTS,
        DRAFT,
        {"answer": "Added 2 deadlines and saved a reply draft."},
    )
    outcome = PlanExecutor(model, tools).run(
        "Find my professor's latest email, summarize what needs to be done, add the deadlines "
        "to my calendar, and prepare a reply.")  # fmt: skip
    assert outcome.status == "completed", [(r.step.id, r.status, r.note) for r in outcome.steps]
    assert tools.calls == [
        "gmail_search_messages",
        "gmail_get_message",
        "calendar_list_events",
        "calendar_create_event",
        "calendar_get_event",
        "calendar_create_event",
        "calendar_get_event",
        "gmail_create_draft",
        "gmail_get_draft",
    ]  # every write is read back with its provider id
    assert all(call.verified for r in outcome.steps[3:] for call in r.calls)
    assert [e["title"] for e in calendar.events] == ["Draft chapter due", "Book progress meeting"]
    assert approvals == ["calendar_create_event"] * 2 + ["gmail_create_draft"]
    assert gmail.sent == [] and outcome.answer.startswith("Added 2 deadlines")
    assert model.schemas[0]["properties"]["steps"]["maxItems"] == 8
    assert model.schemas[4]["properties"]["calls"]["maxItems"] == 5


def test_validation_rejects_bad_plans_and_adds_runtime_rules():
    tools, *_ = world()
    schemas = {d["name"]: d["input_schema"] for d in tools.definitions()}
    for bad, message in [
        ({"goal": "g", "steps": [step(1, "gmail_get_message", ["step_2"]),
                                 step(2, "gmail_search_messages")]}, "earlier step"),
        ({"goal": "g", "steps": [step(1, "shell")]}, "unknown tool"),
        ({"goal": "g", "steps": [step(1, "ask_user")]}, "no question"),
        ({"goal": "g", "steps": [step(1, "gmail_search_messages", args={"to": "x"})]},
         "unknown arguments"),
    ]:  # fmt: skip
        with pytest.raises(PlanValidationError, match=message):
            validate_plan(Plan.model_validate(bad), schemas)
    plan = validate_plan(Plan.model_validate(PLAN), schemas)
    assert plan.steps[0].expected_output == "nonempty"
    assert plan.steps[3].verification.tool == "calendar_get_event"
    with pytest.raises(ValueError):
        Plan.model_validate({"goal": "g", "steps": [step(i, "gmail_search_messages")
                                                    for i in range(1, 10)]})  # fmt: skip


def test_router_sends_only_multi_step_requests_to_the_planner():
    assert needs_plan("Find Alex's email, check my calendar, and prepare a reply.")
    assert not needs_plan("Open Safari")
    assert not needs_plan("Find the email from Alex")


def test_denied_approval_stops_the_plan_and_nothing_is_sent():
    tools, gmail, _, approvals = world(approve=lambda request: False)
    plan = {
        "goal": "Reply",
        "steps": [
            step(1, "gmail_search_messages"),
            step(2, "gmail_create_draft", ["step_1"]),
            step(3, "gmail_send_draft", ["step_2"]),
        ],
    }
    model = Script(plan, {"query": "rivera"}, DRAFT, {"answer": "Stopped."})
    outcome = PlanExecutor(model, tools).run("Find, draft, and send a reply to Rivera")
    assert outcome.status == "stopped"
    assert [r.status for r in outcome.steps] == [StepStatus.SUCCEEDED, StepStatus.STOPPED]
    assert "gmail_send_draft" not in tools.calls and gmail.drafts == {} and gmail.sent == []


def test_rate_limit_is_retried_within_bounds():
    gmail = FailureInjector(FakeGmail([PROFESSOR]), {"search_messages": ["RateLimitedError"]})
    tools, *_ = world(gmail=gmail)
    plan = {"goal": "Find", "steps": [step(1, "gmail_search_messages")]}
    sleeps = []
    outcome = PlanExecutor(Script(plan, {"query": "rivera"}, {"answer": "Found it."}), tools,
                           sleep=sleeps.append).run("find the professor email")  # fmt: skip
    assert outcome.status == "completed" and sleeps == [1.0]
    assert tools.calls == ["gmail_search_messages", "gmail_search_messages"]


def test_empty_search_retries_once_then_asks_the_user():
    tools, *_ = world(gmail=FakeGmail([]))
    plan = {
        "goal": "Find",
        "steps": [step(1, "gmail_search_messages"), step(2, "gmail_get_message", ["step_1"])],
    }
    model = Script(plan, {"query": "rivera"}, {"query": "professor"})
    outcome = PlanExecutor(model, tools).run("find and read the professor email")
    assert outcome.status == "needs_user"
    assert "Can you give me more detail" in outcome.answer
    assert tools.calls == ["gmail_search_messages", "gmail_search_messages"]


def test_not_found_replans_once_and_then_gives_up():
    gmail = FailureInjector(
        FakeGmail([PROFESSOR]), {"get_message": ["NotFoundError", "NotFoundError"]}
    )
    tools, *_ = world(gmail=gmail)
    plan = {
        "goal": "Read",
        "steps": [step(1, "gmail_search_messages"), step(2, "gmail_get_message", ["step_1"])],
    }
    replan = {"goal": "Read", "steps": [step(1, "gmail_get_message")]}
    model = Script(plan, {"query": "rivera"}, {"message_id": "zzz"}, replan,
                   {"message_id": "m1"}, {"answer": "Could not read it."})  # fmt: skip
    outcome = PlanExecutor(model, tools).run("find and read the professor email")
    assert outcome.replans == 1 and outcome.status == "partial"
    assert [r.step.id for r in outcome.steps] == ["step_1", "step_2", "step_3"]
    assert outcome.steps[-1].status is StepStatus.FAILED


def test_ambiguous_send_requires_review_but_idempotent_create_is_retried():
    gmail = FailureInjector(FakeGmail([PROFESSOR]), {"send_draft": ["ProviderError"]})
    tools, inner, *_ = world(gmail=gmail)
    inner.create_draft(["a@x.com"], "Hi", "Body")
    plan = {"goal": "Send", "steps": [step(1, "gmail_send_draft", args={"draft_id": "draft-1"})]}
    outcome = PlanExecutor(Script(plan, {"answer": "Unclear."}), tools).run("send my draft")
    assert outcome.status == "stopped" and outcome.steps[0].status is StepStatus.NEEDS_REVIEW
    assert tools.calls == ["gmail_send_draft"]  # never resent blindly

    calendar = FailureInjector(FakeCalendar([], TZ), {"create_event": ["ProviderError"]})
    tools, _, inner_calendar, _ = world(calendar=calendar)
    event = {"title": "Sync", "start": "2026-10-27T14:00:00-04:00",
             "end": "2026-10-27T14:30:00-04:00"}  # fmt: skip
    plan = {"goal": "Add", "steps": [step(1, "calendar_create_event", args=event)]}
    outcome = PlanExecutor(Script(plan, {"answer": "Added."}), tools).run("add the sync")
    assert outcome.status == "completed"
    assert len(inner_calendar._target.events if hasattr(inner_calendar, "_target")
               else calendar._target.events) == 1  # fmt: skip


def test_model_call_budget_is_enforced():
    tools, *_ = world()
    with pytest.raises(PlanningError):
        PlanExecutor(Script(PLAN), tools, budget=PlanBudget(max_model_calls=1)).run(
            "find, check, and draft")  # fmt: skip


def test_stale_reads_are_refreshed_before_a_write():
    tools, *_ = world()
    now = [0.0]

    def clock():  # every look at the clock is a minute later
        now[0] += 60
        return now[0]

    plan = {"goal": "Add", "steps": [
        step(1, "calendar_check_availability", args={
            "start": "2026-10-27T14:00:00-04:00", "end": "2026-10-27T15:00:00-04:00"}),
        step(2, "calendar_create_event", ["step_1"], args={
            "title": "Sync", "start": "2026-10-27T14:00:00-04:00",
            "end": "2026-10-27T14:30:00-04:00"}),
    ]}  # fmt: skip
    executor = PlanExecutor(Script(plan, {"answer": "Added."}), tools,
                            budget=PlanBudget(stale_after_seconds=10, total_seconds=1e9,
                                              step_seconds=1e9),
                            clock=clock)  # fmt: skip
    outcome = executor.run("check my calendar and add the sync")
    assert tools.calls.count("calendar_check_availability") == 2
    assert outcome.steps[0].note == "refreshed before a later write"
    assert outcome.status == "completed"
