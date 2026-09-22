import asyncio
import io
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, Message
from assistant.runtime import JamieApplication
from assistant.ui.repl import run_repl


class FakeBackend:
    def __init__(self, responses: list[LLMResponse | LLMError]) -> None:
        self._responses = iter(responses)
        self.requests: list[list[Message]] = []
        self.close_calls = 0

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.requests.append(list(messages))
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
