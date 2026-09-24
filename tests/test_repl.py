import asyncio
import inspect
import io
import os
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from assistant.agent.events import (
    AgentEvent,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnStarted,
)
from assistant.agent.loop import ToolResult
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, Message
from assistant.runtime import JamieApplication, TurnResult
from assistant.ui.repl import COMMANDS, create_prompt_session, run_repl


class FakeBackend:
    def __init__(self, responses: list[LLMResponse | LLMError]) -> None:
        self._responses = iter(responses)
        self.requests: list[list[Message]] = []
        self.thinking: list[bool | None] = []
        self.close_calls = 0

    def chat(self, messages, *, temperature=0.6, max_tokens=1024, thinking=None):
        self.requests.append(list(messages))
        self.thinking.append(thinking)
        response = next(self._responses)
        if isinstance(response, LLMError):
            raise response
        return response

    def close(self) -> None:
        self.close_calls += 1


def _run(tmp_path: Path, keys: str, backend: FakeBackend) -> tuple[str, FakeBackend]:
    console = Console(file=io.StringIO(), width=200, no_color=True)
    application = JamieApplication(
        ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
        backend_factory=lambda: backend,
    )

    async def main() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                await run_repl(application, console=console, session=session)

    asyncio.run(main())
    return console.file.getvalue(), backend


def test_repl_keeps_transcript_between_turns_and_exits_on_quit(tmp_path: Path):
    backend = FakeBackend([LLMResponse("Hello there"), LLMResponse("You asked hello")])

    output, backend = _run(tmp_path, "hello\n\nwhat did I just ask?\n/quit\n", backend)

    assert "Jamie › Hello there" in output
    assert "Jamie › You asked hello" in output
    assert "local assistant" in output and "Esc+Enter adds a line" in output
    assert backend.requests[1] == [
        Message("user", "hello"),
        Message("assistant", "Hello there"),
        Message("user", "what did I just ask?"),
    ]
    assert backend.close_calls == 1


def test_repl_reports_turn_failure_and_continues_until_ctrl_d(tmp_path: Path):
    backend = FakeBackend([LLMError("model offline"), LLMResponse("Recovered")])

    output, backend = _run(tmp_path, "first\nsecond\n\x04", backend)

    assert "Error: model offline" in output
    assert "Jamie › Recovered" in output
    assert backend.close_calls == 1


def test_repl_thinking_command_toggles_reasoning_without_calling_the_model(tmp_path: Path):
    backend = FakeBackend([LLMResponse("Default"), LLMResponse("Quick"), LLMResponse("Deep")])

    output, backend = _run(
        tmp_path,
        "/thinking\nhi\n/thinking off\nhi\n/thinking maybe\n/thinking ON\nhi\n/quit\n",
        backend,
    )

    assert "Thinking: server default" in output
    assert "Thinking: off" in output
    assert "Usage: /thinking on|off" in output
    assert "Thinking: on" in output
    assert backend.thinking == [None, False, True]
    assert all(m.content != "/thinking off" for request in backend.requests for m in request)


def _run_with_history(tmp_path: Path, keys: str, backend: FakeBackend) -> Path:
    history_path = tmp_path / "data" / "input_history"
    application = JamieApplication(
        ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
        backend_factory=lambda: backend,
    )

    async def main() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = create_prompt_session(history_path, input=pipe, output=DummyOutput())
            async with application:
                await run_repl(
                    application,
                    console=Console(file=io.StringIO(), width=200),
                    session=session,
                )

    asyncio.run(main())
    return history_path


def test_history_persists_typed_prompts_only(tmp_path: Path):
    backend = FakeBackend([LLMResponse("Secret answer"), LLMResponse("ok"), LLMResponse("again")])

    history_path = _run_with_history(tmp_path, "first\n hidden\n\x04", backend)
    stored = history_path.read_text()

    assert "+first" in stored
    assert "hidden" not in stored
    assert "Secret answer" not in stored
    assert history_path.stat().st_mode & 0o777 == 0o600

    # Up arrow recalls the previous session's prompt.
    _run_with_history(tmp_path, "\x1b[A\n\x04", backend)
    assert backend.requests[-1][-1] == Message("user", "first")


def test_emacs_editing_keys(tmp_path: Path):
    backend = FakeBackend([LLMResponse("ok"), LLMResponse("ok")])

    # Ctrl+A/E move, Ctrl+W deletes a word, Ctrl+U/K clear around the cursor.
    keys = "bc\x01a\x05d drop\x17\n" + "junk\x15keep\x01\x0bnew\n" + "\x04"
    _run_with_history(tmp_path, keys, backend)

    assert [request[-1].content for request in backend.requests] == ["abcd", "new"]


def test_multiline_prompt_reaches_the_model_as_one_turn(tmp_path: Path):
    backend = FakeBackend([LLMResponse("Understood")])

    _run_with_history(tmp_path, "first line\x1b\r    second line\r\x04", backend)

    assert backend.requests == [[Message("user", "first line\n    second line")]]


def _completions(text: str) -> list[str]:
    completer = COMMANDS.completer()
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


def test_tab_completes_slash_commands_from_the_registry():
    assert _completions("/thi") == ["/thinking", "/thinking on", "/thinking off"]
    assert _completions("/thinking of") == ["/thinking off"]
    assert {c.name for c in COMMANDS} <= set(_completions("/"))
    assert _completions("hello") == []


def test_slash_commands_stay_local_and_new_resets_the_conversation(tmp_path: Path):
    backend = FakeBackend([LLMResponse("One", "qwen", "stop", 3), LLMResponse("Two")])

    output, backend = _run(
        tmp_path,
        "first\n/help\n/model\n/nope\n/new\n/context\nsecond\n/stats\n/exit\n",
        backend,
    )

    assert "/cwd [path]" in output
    assert "Model: qwen" in output
    assert "Unknown command /nope. Type /help for commands." in output
    assert "Started a new conversation." in output
    assert "Messages: 0 (0 exchanges)" in output
    assert re.search(r"Turns\s+2\n", output)
    assert re.search(r"Generated tokens\s+3\n", output)
    assert backend.requests == [[Message("user", "first")], [Message("user", "second")]]
    assert backend.close_calls == 1


def test_paths_with_a_leading_slash_go_to_the_model(tmp_path: Path):
    backend = FakeBackend([LLMResponse("It is a file")])

    _, backend = _run(tmp_path, "/Users/me/notes.txt what is this?\n/quit\n", backend)

    assert backend.requests[0][-1].content == "/Users/me/notes.txt what is this?"


# The tests below replace the runtime with a script, so no model is needed.


@dataclass
class Turn:
    """One scripted turn: progress events, then an answer, an error, or a wait for Ctrl+C."""

    answer: str = "fake answer"
    events: list[AgentEvent] = field(default_factory=list)
    first_token_seconds: float | None = None
    error: Exception | None = None
    wait_for_cancel: bool = False


@dataclass
class Call:
    prompt: str
    history: list[Message]
    conversation_id: str | None


class FakeJamieApplication:
    """Same interface as JamieApplication, driven by a script instead of a model."""

    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)
        self.calls: list[Call] = []
        self.starts = 0
        self.closes = 0
        self.conversations = 0
        self.model_calls = 0

    async def __aenter__(self):
        self.starts += 1
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.closes += 1

    async def run_turn(
        self,
        prompt,
        *,
        history=None,
        temperature=0.6,
        max_tokens=1024,
        thinking=None,
        cancel=None,
        conversation_id=None,
        event_sink=lambda event: None,
    ) -> TurnResult:
        self.calls.append(Call(prompt, list(history or []), conversation_id))
        self.model_calls += 1
        turn = self.turns.pop(0) if self.turns else Turn()
        event_sink(TurnStarted(prompt))
        for event in turn.events:
            event_sink(event)
        try:
            if turn.wait_for_cancel:
                os.kill(os.getpid(), signal.SIGINT)  # the user presses Ctrl+C mid-turn
                while not cancel.cancelled:
                    await asyncio.sleep(0.01)
                cancel.raise_if_cancelled()
            if turn.error is not None:
                raise turn.error
        except Exception as exc:
            event_sink(TurnFailed(str(exc)))
            raise
        event_sink(AssistantFinished(turn.answer))
        return TurnResult(
            turn.answer, output_tokens=10, first_token_seconds=turn.first_token_seconds
        )

    def start_conversation(self) -> str:
        self.conversations += 1
        return f"conversation-{self.conversations}"

    def context_usage(self, history):
        return None

    def recent_conversations(self, limit=10):
        return []

    def find_conversation(self, id_prefix):
        return None

    def conversation_messages(self, conversation_id):
        return []


def test_fake_runtime_matches_the_real_interface():
    for name in ("run_turn", "start_conversation", "context_usage", "recent_conversations"):
        real = inspect.signature(getattr(JamieApplication, name)).parameters
        fake = inspect.signature(getattr(FakeJamieApplication, name)).parameters
        assert list(fake) == list(real), name


def drive(application: FakeJamieApplication, keys: str) -> str:
    """Run the real REPL over scripted keystrokes; return what it printed."""
    console = Console(file=io.StringIO(), width=200, no_color=True)

    async def main() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                await run_repl(application, console=console, session=session)

    asyncio.run(main())
    return console.file.getvalue()


def test_runtime_starts_once_and_multiple_prompts_reuse_it():
    application = FakeJamieApplication(Turn("one"), Turn("two"), Turn("three"))

    output = drive(application, "a\nb\nc\n/quit\n")

    assert (application.starts, application.closes) == (1, 1)
    assert [call.prompt for call in application.calls] == ["a", "b", "c"]
    assert application.calls[2].history == [
        Message("user", "a"),
        Message("assistant", "one"),
        Message("user", "b"),
        Message("assistant", "two"),
    ]
    assert "Jamie › three" in output


def test_new_resets_the_conversation_but_not_the_runtime():
    application = FakeJamieApplication()

    output = drive(application, "a\nb\n/new\nc\n/quit\n")

    assert application.starts == 1
    assert application.calls[2].history == []
    assert [call.conversation_id for call in application.calls] == [
        "conversation-1",
        "conversation-1",
        "conversation-2",
    ]
    assert "Started a new conversation." in output


@pytest.mark.parametrize("exit_keys", ["/quit\n", "/exit\n", "\x04"])
def test_quit_and_ctrl_d_close_cleanly(exit_keys: str):
    application = FakeJamieApplication()

    output = drive(application, f"hello\n{exit_keys}never sent\n")

    assert [call.prompt for call in application.calls] == ["hello"]
    assert application.closes == 1
    assert "Error" not in output


def test_ctrl_c_cancels_the_turn_and_jamie_keeps_running():
    application = FakeJamieApplication(Turn(wait_for_cancel=True), Turn("still here"))
    before = signal.getsignal(signal.SIGINT)

    output = drive(application, "long task\nnext\n/quit\n")

    assert "Cancelled." in output
    assert "Jamie › still here" in output
    assert application.calls[1].history == []  # the cancelled exchange was not kept
    assert signal.getsignal(signal.SIGINT) is before  # Ctrl+C handling restored


def test_slash_commands_never_reach_the_model():
    application = FakeJamieApplication()
    commands = [command.name for command in COMMANDS if command.name not in {"/quit", "/clear"}]

    output = drive(application, "".join(f"{name}\n" for name in commands) + "/typo\n/quit\n")

    assert application.calls == []
    assert "Unknown command /typo" in output


def test_tool_events_render_as_progress_lines():
    events = [
        ToolStarted("read_file", {"path": "notes.md"}),
        ToolFinished("read_file", ToolResult(success=True, output="a\nb\nc"), 0.1),
        ToolStarted("web_search", {"query": "free threading"}),
        ToolFailed("web_search", "network unavailable", 0.2),
    ]
    application = FakeJamieApplication(Turn("Summarized.", events=events))

    output = drive(application, "summarize\n/quit\n")

    assert "▸ Reading notes.md" in output
    assert "✓ Read notes.md · 3 lines" in output
    assert '✗ Searching web for "free threading" failed' in output
    assert "network unavailable" in output
    assert "Jamie › Summarized." in output
    assert '"path"' not in output  # never the protocol arguments


def test_activity_stops_before_answers_and_after_failures(monkeypatch: pytest.MonkeyPatch):
    activity: list[str] = []
    console = Console(file=io.StringIO(), force_terminal=True, width=200)

    class Status:
        def start(self):
            activity.append("start")

        def update(self, message):
            activity.append(message)

        def stop(self):
            activity.append("stop")

    monkeypatch.setattr(console, "status", lambda *args, **kwargs: Status())
    application = FakeJamieApplication(
        Turn("done", events=[ToolStarted("read_file", {"path": "notes.md"})]),
        Turn(error=LLMError("offline")),
    )

    async def main() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text("first\nsecond\n/quit\n")
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                await run_repl(application, console=console, session=session)

    asyncio.run(main())

    assert activity == ["start", "Jamie is using a tool…", "stop", "start", "stop"]
    assert "Jamie › done" in console.file.getvalue()
    assert "Error: offline" in console.file.getvalue()


def test_a_failed_model_request_does_not_kill_the_repl():
    application = FakeJamieApplication(
        Turn(error=LLMError("Cannot reach llama.cpp")), Turn("recovered")
    )

    output = drive(application, "first\nsecond\n/stats\n/quit\n")

    assert "Error: Cannot reach llama.cpp" in output
    assert "Jamie › recovered" in output
    assert re.search(r"Turns\s+1\n", output)
    assert re.search(r"Failed turns\s+1\n", output)
    assert application.calls[1].history == []  # the failed exchange was not kept
    assert application.closes == 1


def test_stats_count_turns_model_calls_tools_first_tokens_and_startup():
    tool = [
        ToolStarted("read_file", {"path": "a.md"}),
        ToolFinished("read_file", ToolResult(success=True, output="x"), 0.5),
    ]
    application = FakeJamieApplication(
        Turn("one", events=tool, first_token_seconds=1.0),
        Turn(error=LLMError("offline")),
        Turn("two", first_token_seconds=2.0),
    )
    console = Console(file=io.StringIO(), width=200, no_color=True)

    async def main() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text("a\nb\nc\n/stats\n/quit\n")
            session = PromptSession(input=pipe, output=DummyOutput())
            async with application:
                started = time.monotonic() - 2.5
                await run_repl(application, console=console, session=session, started=started)

    asyncio.run(main())
    output = console.file.getvalue()

    for row in (
        r"Turns\s+2\n",
        r"Failed turns\s+1\n",
        r"Model calls\s+3\n",
        r"Tool calls\s+1\n",
        r"Generated tokens\s+20\n",
        r"Average first token\s+1\.5 s\n",
        r"Average tool\s+0\.5 s\n",
        r"Startup\s+2\.5 s\n",
    ):
        assert re.search(row, output), row
