"""The target workflow end to end: search, read, check availability, draft, approve."""

import base64
import json
from zoneinfo import ZoneInfo

from assistant.agent import run_agent
from assistant.agent_evaluation import load_agent_cases
from assistant.connectors import Connectors
from assistant.connectors.google.auth import CALENDAR_FREEBUSY, GMAIL_COMPOSE, GMAIL_READONLY
from assistant.connectors.google.calendar import CalendarConnector
from assistant.connectors.google.gmail import GmailConnector
from assistant.llm.base import LLMResponse
from assistant.security import CallbackApprovalProvider, PermissionLevel
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools
from assistant.tools.scopes import ToolScope, select_tools

TZ = ZoneInfo("America/Toronto")
GMAIL = "/gmail/v1/users/me"


def tool(name, **args):
    return json.dumps({"action": "tool", "tool": name, "args": args})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.requests.append(list(messages))
        return LLMResponse(next(self.responses))


def message(full=False):
    payload = {
        "headers": [
            {"name": "From", "value": "Alex Kim <alex@example.com>"},
            {"name": "Subject", "value": "Meeting next week"},
            {"name": "Message-ID", "value": "<m1@example.com>"},
        ]
    }
    if full:
        body = "Could we meet next Tuesday at 2pm for 30 minutes?\n\nOn Fri Me wrote:\n> old"
        payload.update(
            {
                "mimeType": "text/plain",
                "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()},
            }
        )
    return {
        "id": "m1",
        "threadId": "t1",
        "snippet": "Could we meet next Tuesday at 2pm?",
        "internalDate": "1792765800000",
        "payload": payload,
    }


def test_find_email_check_calendar_and_save_approved_draft(google_api):
    api = google_api
    session = api.session([GMAIL_READONLY, GMAIL_COMPOSE, CALENDAR_FREEBUSY])
    api.on("GET", f"{GMAIL}/messages", {"messages": [{"id": "m1"}]})
    api.on(
        "GET",
        f"{GMAIL}/messages/m1",
        lambda request: message(full=request.url.params.get("format") == "full"),
    )
    api.on(
        "POST",
        "/calendar/v3/freeBusy",
        {
            "calendars": {
                "primary": {
                    "busy": [{"start": "2026-10-27T17:00:00Z", "end": "2026-10-27T18:00:00Z"}]
                }
            }
        },
    )
    api.on("POST", f"{GMAIL}/drafts", {"id": "r-1", "message": {"id": "d1", "threadId": "t1"}})

    from datetime import datetime

    connectors = Connectors(
        tz=TZ, gmail=GmailConnector(session), calendar=CalendarConnector(session, TZ)
    )
    approvals = []
    registry = register_personal_tools(
        ToolRegistry(approvals=CallbackApprovalProvider(lambda r: approvals.append(r) or True)),
        connectors,
        now=lambda: datetime(2026, 10, 25, 10, tzinfo=TZ),
    )
    tools = select_tools(registry, ToolScope.PERSONAL)
    reply = "Hi Alex, Tuesday at 2pm for 30 minutes works for me. See you then."
    model = ScriptedModel(
        [
            tool("gmail_search_messages", query="from:alex meeting newer_than:14d"),
            tool("gmail_get_message", message_id="m1"),
            tool(
                "calendar_check_availability",
                start="2026-10-27T14:00:00-04:00",
                end="2026-10-27T14:30:00-04:00",
                duration_minutes=30,
            ),
            tool(
                "gmail_create_draft", to=["alex@example.com"], body=reply, reply_to_message_id="m1"
            ),
            json.dumps(
                {
                    "action": "finish",
                    "message": "You're free Tuesday 2:00-2:30pm. I saved a reply draft (r-1).",
                }
            ),
        ]
    )

    answer = run_agent(
        model,
        tools,
        "Find Alex's email about meeting next week, check my calendar, and prepare a reply.",
        max_steps=16,
    )

    assert "r-1" in answer
    # The system prompt carries "now" so the model can resolve "next Tuesday".
    assert "Current local time: Sunday 2026-10-25T10:00-04:00" in model.requests[0][0].content
    # The model saw concise data: the search result is a summary, the body has no quotes.
    observed = []
    for item in model.requests[-1][2:]:
        if item.role == "user" and item.content.startswith("Tool observation:"):
            observed.append(json.loads(item.content.splitlines()[1])["output"])
    assert len(observed) == 4
    assert any('"subject":"Meeting next week"' in text for text in observed)
    assert not any("> old" in text for text in observed)
    assert any('"free":true' in text for text in observed)
    # Exactly one approval, for the draft, showing what will be saved. Nothing was sent.
    assert [(r.tool, r.permission) for r in approvals] == [
        ("gmail_create_draft", PermissionLevel.EXTERNAL)
    ]
    assert approvals[0].arguments["body"] == reply
    assert not api.sent("POST", f"{GMAIL}/drafts/send")
    raw = json.loads(api.sent("POST", f"{GMAIL}/drafts")[0].content)["message"]
    assert raw["threadId"] == "t1"


def test_denied_draft_is_reported_and_nothing_is_saved(google_api):
    api = google_api
    session = api.session([GMAIL_READONLY, GMAIL_COMPOSE])
    connectors = Connectors(tz=TZ, gmail=GmailConnector(session))
    registry = register_personal_tools(
        ToolRegistry(approvals=CallbackApprovalProvider(lambda r: False)), connectors
    )
    model = ScriptedModel(
        [
            tool("gmail_create_draft", to=["alex@example.com"], body="Works for me."),
            json.dumps({"action": "finish", "message": "The draft was not saved."}),
        ]
    )
    answer = run_agent(model, registry, "Draft a reply to Alex.")
    assert answer == "Stopped: gmail_create_draft was not approved."
    assert not api.requests


def test_personal_eval_case_runs_against_fake_connectors():
    from assistant.agent_evaluation import _personal_registry

    [case] = [c for c in load_agent_cases() if c.id == "personal-reply-001"]
    registry = _personal_registry(case.personal)
    found = registry.execute("gmail_search_messages", {"query": "from:alex meeting"})
    assert json.loads(found.output)[0]["id"] == "m1"
    busy = registry.execute(
        "calendar_check_availability",
        {"start": "2026-10-27T14:00:00-04:00", "end": "2026-10-27T14:30:00-04:00"},
    )
    assert json.loads(busy.output)["free"] is True
    draft = registry.execute(
        "gmail_create_draft",
        {"to": ["Alex Kim <alex@example.com>"], "body": "Works.", "reply_to_message_id": "m1"},
    )
    assert json.loads(draft.output)["id"] == "draft-1"
