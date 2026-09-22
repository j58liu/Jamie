import asyncio
import io
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, Message
from assistant.runtime import JamieApplication
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
    assert "Turns: 2 completed, 0 failed" in output
    assert "Output tokens: 3" in output
    assert backend.requests == [[Message("user", "first")], [Message("user", "second")]]
    assert backend.close_calls == 1


def test_paths_with_a_leading_slash_go_to_the_model(tmp_path: Path):
    backend = FakeBackend([LLMResponse("It is a file")])

    _, backend = _run(tmp_path, "/Users/me/notes.txt what is this?\n/quit\n", backend)

    assert backend.requests[0][-1].content == "/Users/me/notes.txt what is this?"
