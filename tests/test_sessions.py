import asyncio
import io
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from assistant.cancellation import CancellationToken, TurnCancelled
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, Message
from assistant.memory import MemoryStore
from assistant.runtime import JamieApplication
from assistant.ui.commands import ago
from assistant.ui.repl import run_repl

# Store


def test_conversations_list_latest_first_and_hide_empty_ones(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = store.start_conversation("/work/a")
    store.start_conversation("/work/empty")
    second = store.start_conversation("/work/b")
    store.save_exchange(first, "Research Python free threading", "Sure.", "/work/a")
    store.save_exchange(second, "Organize Downloads", "Done.", "/work/b2")
    store.save_exchange(first, "And the GIL?", "It is optional.", "/work/a")

    recent = store.recent_conversations()

    assert [c.id for c in recent] == [first, second]
    assert recent[0].title == "Research Python free threading"
    assert recent[0].messages == 4
    assert recent[1].working_directory == "/work/b2"
    assert store.conversation_messages(first) == [
        Message("user", "Research Python free threading"),
        Message("assistant", "Sure."),
        Message("user", "And the GIL?"),
        Message("assistant", "It is optional."),
    ]


def test_find_conversation_by_unique_prefix_only(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    ids = [store.start_conversation("/w") for _ in range(20)]
    for conversation in ids:
        store.save_exchange(conversation, "hi", "hello", "/w")

    assert store.find_conversation(ids[0]).id == ids[0]
    assert store.find_conversation(ids[0][:12]).id == ids[0]
    assert store.find_conversation("") is None
    assert store.find_conversation("%") is None  # not a LIKE wildcard
    assert store.find_conversation("zz") is None


def test_existing_databases_gain_the_conversations_table(tmp_path: Path):
    path = tmp_path / "memory.sqlite3"
    MemoryStore(path)
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE conversations")  # as created by the previous release
    assert "conversations" in MemoryStore(path).table_names()


# Runtime


class Backend:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024, thinking=None):
        self.requests.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _application(tmp_path: Path, backend: Backend) -> JamieApplication:
    config = ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3"))
    return JamieApplication(config, backend_factory=lambda: backend)


def test_only_completed_visible_exchanges_are_saved(tmp_path: Path):
    backend = Backend(
        [
            LLMResponse("<think>private plan</think>Visible answer"),
            LLMError("offline"),
            LLMResponse("late answer"),
        ]
    )
    application = _application(tmp_path, backend)

    async def run():
        async with application:
            conversation = application.start_conversation()
            await application.run_turn("first", conversation_id=conversation)
            with pytest.raises(LLMError):
                await application.run_turn("fails", conversation_id=conversation)
            token = CancellationToken()
            token.cancel()
            with pytest.raises(TurnCancelled):
                await application.run_turn("cancelled", conversation_id=conversation, cancel=token)
            return application.conversation_messages(conversation)

    assert asyncio.run(run()) == [Message("user", "first"), Message("assistant", "Visible answer")]
    database = (tmp_path / "memory.sqlite3").read_bytes()
    assert b"private plan" not in database


def test_recall_skips_the_active_conversation_but_not_others(tmp_path: Path):
    backend = Backend([LLMResponse("Kyoto in April."), LLMResponse("ok"), LLMResponse("ok")])
    application = _application(tmp_path, backend)

    async def run():
        async with application:
            trip = application.start_conversation()
            await application.run_turn("Plan my Kyoto trip", conversation_id=trip)
            await application.run_turn("Kyoto trip budget?", conversation_id=trip)
            other = application.start_conversation()
            await application.run_turn("Kyoto trip again?", conversation_id=other)

    asyncio.run(run())
    same, other = backend.requests[1], backend.requests[2]
    assert not any(m.role == "system" for m in same)
    assert "Kyoto in April." in other[0].content and other[0].role == "system"


# Terminal


def _repl(tmp_path: Path, keys: str, backend: Backend) -> str:
    console = Console(file=io.StringIO(), width=200, no_color=True)
    application = _application(tmp_path, backend)

    async def main():
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                await run_repl(application, console=console, session=session)

    asyncio.run(main())
    return console.file.getvalue()


def test_conversation_survives_exit_and_resumes_without_reloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    _repl(
        tmp_path,
        "Research Python free threading\nWhat about the GIL?\n/new\nOrganize Downloads\n/quit\n",
        Backend([LLMResponse("Threads."), LLMResponse("GIL is optional."), LLMResponse("Ok.")]),
    )

    monkeypatch.chdir(tmp_path)
    backend = Backend([LLMResponse("Continuing.")])
    factory_calls = []
    config = ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3"))

    def factory():
        factory_calls.append(1)
        return backend

    console = Console(file=io.StringIO(), width=200, no_color=True)
    application = JamieApplication(config, backend_factory=factory)

    async def main():
        with create_pipe_input() as pipe:
            pipe.send_text("/history\n/resume 2\nAnd since 3.13?\n/history\n/quit\n")
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                await run_repl(application, console=console, session=session)

    asyncio.run(main())
    output = console.file.getvalue()

    assert "Recent conversations" in output
    assert "1  just now   Organize Downloads" in output
    assert "2  just now   Research Python free threading" in output
    assert "Resumed: Research Python free threading (2 exchanges)" in output
    assert f"Working directory: {project.resolve()}" in output
    assert Path.cwd() == project.resolve()
    assert backend.requests[0] == [
        Message("user", "Research Python free threading"),
        Message("assistant", "Threads."),
        Message("user", "What about the GIL?"),
        Message("assistant", "GIL is optional."),
        Message("user", "And since 3.13?"),
    ]
    assert "* 1  just now   Research Python free threading" in output
    assert factory_calls == [1]  # the model backend was not recreated by /resume


def test_resume_reports_bad_ids_and_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    _repl(tmp_path, "hello\n/quit\n", Backend([LLMResponse("hi")]))
    monkeypatch.chdir(tmp_path)
    os.rmdir(gone)

    output = _repl(tmp_path, "/resume\n/resume 9\n/resume nope\n/resume 1\n/quit\n", Backend([]))

    assert "Usage: /resume <number|id>" in output
    assert "No conversation matches 9." in output
    assert "No conversation matches nope." in output
    assert "Resumed: hello (1 exchanges)" in output
    assert f"Saved directory is unavailable; staying in {tmp_path}" in output


def test_new_conversation_is_not_saved_until_the_first_exchange(tmp_path: Path):
    output = _repl(tmp_path, "/new\n/new\n/history\n/quit\n", Backend([]))
    assert "No saved conversations yet." in output


def test_relative_times():
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)

    def at(delta: timedelta) -> str:
        return ago((now - delta).isoformat(), now)

    assert at(timedelta(seconds=5)) == "just now"
    assert at(timedelta(minutes=2)) == "2m ago"
    assert at(timedelta(hours=1, minutes=5)) == "1h ago"
    assert at(timedelta(hours=30)) == "yesterday"
    assert at(timedelta(days=5)) == "5d ago"
