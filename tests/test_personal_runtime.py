"""Personal memory and planning inside a real JamieApplication turn."""

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from assistant.config import ProjectConfig
from assistant.connectors import Connectors
from assistant.connectors.fakes import FakeCalendar, FakeGmail
from assistant.knowledge import MemoryCandidate, MemorySource, MemoryType, PersonalMemory
from assistant.llm.base import LLMResponse
from assistant.runtime import JamieApplication
from assistant.security import CallbackApprovalProvider
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools

TZ = ZoneInfo("America/Toronto")


class Backend:
    def __init__(self, *responses):
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.requests = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024, **options):
        self.requests.append(list(messages))
        return LLMResponse(self.responses.pop(0))

    def close(self):
        pass


def run(application, prompt, **options):
    async def turn():
        async with application:
            return await application.run_turn(prompt, **options)

    return asyncio.run(turn())


def test_relevant_personal_facts_reach_the_model_with_provenance(tmp_path):
    database = tmp_path / "memory.sqlite3"
    personal = PersonalMemory(database)
    personal.remember(
        MemoryCandidate(
            MemoryType.PREFERENCE,
            "user",
            "meeting_time",
            "morning",
            MemorySource.EXPLICIT_USER_STATEMENT,
            "I prefer mornings",
        )
    )
    personal.remember(
        MemoryCandidate(
            MemoryType.PROFILE,
            "user",
            "medication",
            "insulin",
            MemorySource.EXPLICIT_USER_STATEMENT,
            "I take insulin",
        )
    )
    backend = Backend("Morning works best for you.")
    application = JamieApplication(ProjectConfig(memory_path=str(database)),
                                   backend_factory=lambda: backend)  # fmt: skip
    run(application, "When should I schedule the meeting?")
    system = backend.requests[0][0].content
    assert "meeting time: morning" in system and "stated by user" in system
    assert "insulin" not in system


def test_extraction_learns_from_the_users_words_after_the_turn(tmp_path):
    database = tmp_path / "memory.sqlite3"
    proposal = {"items": [{"statement": "explicit_user_statement", "type": "preference",
                           "subject": "user", "key": "meeting_time", "value": "morning",
                           "evidence": "I prefer morning meetings"}]}  # fmt: skip
    backend = Backend("Noted.", proposal)
    application = JamieApplication(
        ProjectConfig(memory_path=str(database), tasks_path=str(tmp_path / "tasks.sqlite3")),
        backend_factory=lambda: backend, extract_memories=True,
    )  # fmt: skip
    run(application, "From now on, I prefer morning meetings.")
    [record] = PersonalMemory(database).records()
    assert (record.key, record.value) == ("meeting_time", "morning")
    assert application.model_calls == 2


def test_multi_step_requests_use_the_planner_and_fall_back_when_planning_fails(tmp_path):
    gmail = FakeGmail([{"id": "m1", "thread_id": "t1", "sender": "alex@example.com",
                        "subject": "Meeting", "snippet": "Tuesday?", "received_at": "",
                        "body": "Can we meet Tuesday at 2pm?"}])  # fmt: skip

    def tools():
        return register_personal_tools(
            ToolRegistry(approvals=CallbackApprovalProvider(lambda r: True)),
            Connectors(tz=TZ, gmail=gmail, calendar=FakeCalendar([], TZ)),
            now=lambda: datetime(2026, 10, 25, 10, tzinfo=TZ),
        )

    plan = {"goal": "Reply to Alex", "steps": [
        {"id": "step_1", "description": "find Alex's email", "tool": "gmail_search_messages",
         "depends_on": []},
        {"id": "step_2", "description": "draft a reply", "tool": "gmail_create_draft",
         "depends_on": ["step_1"]}]}  # fmt: skip
    backend = Backend(plan, {"query": "from:alex"},
                      {"to": ["alex@example.com"], "body": "Tuesday works.",
                       "reply_to_message_id": "m1"},
                      {"answer": "Saved a draft reply to Alex."})  # fmt: skip
    application = JamieApplication(ProjectConfig(memory_path=str(tmp_path / "m.sqlite3")),
                                   backend_factory=lambda: backend)  # fmt: skip
    result = run(application, "Find Alex's email, check my calendar, and prepare a reply.",
                 tools=tools(), plan="auto")  # fmt: skip
    assert result.message == "Saved a draft reply to Alex."
    assert application.last_plan.status == "completed" and len(gmail.drafts) == 1

    fallback = Backend(
        "not a plan",
        "still not a plan",
        '{"action":"finish","message":"Handled by the agent loop."}',
    )
    application = JamieApplication(ProjectConfig(memory_path=str(tmp_path / "m2.sqlite3")),
                                   backend_factory=lambda: fallback)  # fmt: skip
    result = run(application, "Find Alex's email, check my calendar, and prepare a reply.",
                 tools=tools(), plan="auto")  # fmt: skip
    assert result.message == "Handled by the agent loop."
    assert application.last_plan is None
