import asyncio

import pytest

from assistant.agent.loop import ToolResult
from assistant.config import ProjectConfig
from assistant.llm.base import (
    ContextUsage,
    LLMError,
    LLMResponse,
    Message,
    constrained_to,
    current_response_schema,
)
from assistant.memory import MemoryStore
from assistant.runtime import JamieApplication
from assistant.runtime.context import SUMMARY_INSTRUCTION, RequestBudget


class SizedModel:
    def __init__(self, *, limit=2048, fail_summary=False):
        self.limit = limit
        self.fail_summary = fail_summary
        self.requests = []
        self.measured = []

    def context_usage(self, messages):
        self.measured.append(list(messages))
        return ContextUsage(16 + sum(len(m.content) for m in messages) // 4, self.limit)

    def chat(self, messages, *, max_tokens=1024, **options):
        usage = self.context_usage(messages)
        assert usage.used + max_tokens <= self.limit, "an oversized request reached inference"
        self.requests.append(list(messages))
        if messages[0].content == SUMMARY_INSTRUCTION:
            assert "keep" in current_response_schema()["properties"]
            if self.fail_summary:
                raise LLMError("summary failed")
            return LLMResponse('{"keep":[1]}', finish_reason="stop")
        return LLMResponse("SQLite", finish_reason="stop")


def history():
    return [
        Message(role, content)
        for index in range(8)
        for role, content in (
            (
                "user",
                "Heron uses SQLite; the deadline is October 3. "
                + f"Background note {index}: "
                + "background " * 200,
            ),
            ("assistant", "Noted."),
        )
    ]


def test_compaction_chunks_large_history_and_keeps_current_observations_and_schema():
    backend = SizedModel()
    older = history()
    system = Message("system", "Tool instructions and recalled memory.")
    tail = [
        Message("user", "Which database?"),
        Message("assistant", '{"action":"tool"}'),
        Message("user", "Tool observation: important current result"),
    ]
    messages = [system, *older, *tail]
    summaries = []
    with constrained_to({"type": "object"}):
        result = RequestBudget(backend, lambda: summaries.append(True)).prepare(
            messages, reserve=512, history_count=len(older)
        )
        assert current_response_schema() == {"type": "object"}
    assert messages[0] == system and messages[-3:] == tail
    assert len(summaries) > 1
    assert result.usage.used + result.usage.output_reserved <= result.usage.maximum
    assert "SQLite" in result.history[1].content


@pytest.mark.parametrize("failure", ["summary", "oversized_prompt"])
def test_failure_preserves_history_and_never_sends_an_oversized_request(failure):
    backend = SizedModel(fail_summary=failure == "summary")
    older = history() if failure == "summary" else []
    messages = [*older, Message("user", "x" * (20000 if failure == "oversized_prompt" else 10))]
    original = list(messages)
    with pytest.raises(LLMError):
        RequestBudget(backend, lambda: None).prepare(
            messages, reserve=512, history_count=len(older)
        )
    assert messages == original
    if failure == "oversized_prompt":
        assert backend.requests == []


def test_runtime_returns_compacted_history_and_reports_complete_request_with_recall(tmp_path):
    backend = SizedModel(limit=4096)
    database = tmp_path / "memory.sqlite3"
    store = MemoryStore(database)
    seed = store.start_session("ask")
    store.remember_exchange(seed, "Which database does Heron use?", "SQLite")
    store.finish_session(seed, success=True)

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(database)), backend_factory=lambda: backend
        ) as app:
            older = history()
            snapshot = list(older)
            result = await app.run_turn(
                "Which database does Heron use?", history=older, max_tokens=256
            )
            assert older == snapshot
            assert result.compacted_history is not None
            assert result.message == "SQLite"
            sent = backend.requests[-1]
            assert "SQLite" in sent[0].content  # recalled memory is included in the budget
            assert result.context.used == backend.context_usage(sent).used
            updated = [
                *result.compacted_history,
                Message("user", "Which database does Heron use?"),
                Message("assistant", result.message),
            ]
            assert app.context_usage(updated) == result.context
            assert result.context.output_reserved == 256

    asyncio.run(run())


def test_last_context_includes_model_requests_made_inside_a_tool(tmp_path):
    class Model(SizedModel):
        def chat(self, messages, **options):
            if messages[-1].content == "Run research":
                return LLMResponse('{"action":"tool","tool":"research","args":{}}')
            return LLMResponse("Nested answer")

    model = Model(limit=4096)
    nested = [Message("system", "Evidence review"), Message("user", "Source content " * 20)]

    class Tools:
        def execute(self, tool, args):
            RequestBudget(model, lambda: None).prepare(nested, reserve=99, history_count=0)
            return ToolResult(success=True, output=model.chat(nested).content, final=True)

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: model,
        ) as app:
            result = await app.run_turn("Run research", tools=Tools())
            assert result.message == "Nested answer"
            assert result.context.output_reserved == 99
            assert result.context.used == model.context_usage(nested).used

    asyncio.run(run())
