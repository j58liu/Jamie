import asyncio
import sqlite3
from pathlib import Path

import pytest

from assistant.config import ProjectConfig
from assistant.llm.base import LLMResponse, Message
from assistant.runtime import JamieApplication, JamieApplicationError, TurnResult


class FakeBackend:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[list[Message], float, int]] = []
        self.close_calls = 0

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.requests.append((list(messages), temperature, max_tokens))
        return next(self._responses)

    def close(self) -> None:
        self.close_calls += 1


def test_application_initializes_once_reuses_backend_and_closes(tmp_path: Path):
    backend = FakeBackend(
        [
            LLMResponse("First answer", "local-qwen", "stop", 4),
            LLMResponse("Second answer", "local-qwen", "stop", 5),
        ]
    )
    factory_calls = 0

    def create_backend() -> FakeBackend:
        nonlocal factory_calls
        factory_calls += 1
        return backend

    database = tmp_path / "memory.sqlite3"
    application = JamieApplication(
        ProjectConfig(memory_path=str(database)),
        backend_factory=create_backend,
    )

    async def run() -> tuple[TurnResult, TurnResult]:
        await application.start()
        await application.start()
        first = await application.run_turn("Alpha request")
        second = await application.run_turn(
            "Continue with beta",
            temperature=0.2,
            max_tokens=64,
        )
        await application.close()
        await application.close()
        return first, second

    first, second = asyncio.run(run())

    assert factory_calls == 1
    assert first == TurnResult("First answer", "local-qwen", "stop", 4)
    assert second == TurnResult("Second answer", "local-qwen", "stop", 5)
    assert backend.requests == [
        ([Message("user", "Alpha request")], 0.6, 1024),
        ([Message("user", "Continue with beta")], 0.2, 64),
    ]
    assert backend.close_calls == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4


def test_application_requires_start_and_rejects_work_after_close(tmp_path: Path):
    application = JamieApplication(
        ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
        backend_factory=lambda: FakeBackend([LLMResponse("unused")]),
    )

    with pytest.raises(JamieApplicationError, match="Start"):
        asyncio.run(application.run_turn("hello"))

    asyncio.run(application.close())
    with pytest.raises(JamieApplicationError, match="closed"):
        asyncio.run(application.start())
    with pytest.raises(JamieApplicationError, match="closed"):
        asyncio.run(application.run_turn("hello"))


def test_application_context_closes_backend_after_failed_turn(tmp_path: Path):
    class FailingBackend(FakeBackend):
        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            raise RuntimeError("model failed")

    backend = FailingBackend([])
    database = tmp_path / "memory.sqlite3"
    application = JamieApplication(
        ProjectConfig(memory_path=str(database)),
        backend_factory=lambda: backend,
    )

    async def run() -> None:
        async with application:
            await application.run_turn("hello")

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(run())
    assert backend.close_calls == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT status FROM sessions").fetchone()[0] == "failed"
