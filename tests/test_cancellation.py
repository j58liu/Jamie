import asyncio
import json
import os
import re
import select
import signal
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from assistant.agent.events import TurnFailed
from assistant.agent.loop import ToolResult, run_agent
from assistant.cancellation import (
    CancellationToken,
    TurnCancelled,
    cancellation_scope,
    current_token,
)
from assistant.config import ProjectConfig
from assistant.llm.base import LLMResponse, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import MemoryStore
from assistant.runtime import JamieApplication
from assistant.tools.macos import SubprocessCommandRunner


def _cancel_soon(token: CancellationToken, delay: float = 0.3) -> None:
    threading.Timer(delay, token.cancel).start()


def _wait_for_cancel() -> None:
    token = current_token()
    assert token is not None, "the token must reach model and tool code"
    for _ in range(200):
        token.raise_if_cancelled()
        time.sleep(0.02)
    raise AssertionError("cancellation never arrived")


# Model generation


def test_cancel_model_generation_closes_the_streamed_request():
    async def never_finishes(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        await asyncio.sleep(30)
        raise AssertionError("the request should have been cancelled")

    backend = LlamaCppBackend(transport=httpx.MockTransport(never_finishes))
    token = CancellationToken()
    _cancel_soon(token)
    started = time.monotonic()

    with cancellation_scope(token), pytest.raises(TurnCancelled):
        backend.chat([Message("user", "Write an essay.")])
    backend.close()

    assert time.monotonic() - started < 2


def test_streamed_answer_is_assembled_with_model_and_usage():
    chunks = [
        {"model": "qwen", "choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"model": "qwen", "choices": [{"delta": {"content": "Hel"}}]},
        {"model": "qwen", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"model": "qwen", "choices": [], "usage": {"completion_tokens": 7}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    with cancellation_scope(CancellationToken()):
        result = backend.chat([Message("user", "Hi")])

    assert result == LLMResponse("Hello", "qwen", "stop", 7, result.first_token_seconds)
    assert result.first_token_seconds is not None and result.first_token_seconds >= 0
    assert requests[0]["stream"] is True
    backend.close()


def test_requests_without_a_token_are_unchanged():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "Hi"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    LlamaCppBackend(transport=httpx.MockTransport(handler)).chat([Message("user", "Hi")])

    assert requests[0]["stream"] is False
    assert "stream_options" not in requests[0]


# Runtime


class _TurnBackend:
    """Blocks on prompts starting with "slow" until cancelled; answers others."""

    def __init__(self, *, honours_token: bool = True) -> None:
        self.honours_token = honours_token
        self.prompts: list[str] = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if prompt.startswith("slow"):
            if self.honours_token:
                _wait_for_cancel()
            time.sleep(0.5)  # a backend that ignores the token finishes anyway
        return LLMResponse(f"answer to {prompt}")


@pytest.mark.parametrize("honours_token", [True, False])
def test_cancelled_turn_stores_nothing_and_the_next_turn_works(tmp_path: Path, honours_token: bool):
    backend = _TurnBackend(honours_token=honours_token)
    database = tmp_path / "memory.sqlite3"
    application = JamieApplication(
        ProjectConfig(memory_path=str(database)), backend_factory=lambda: backend
    )
    events = []

    async def run():
        async with application:
            token = CancellationToken()
            _cancel_soon(token, 0.1)
            with pytest.raises(TurnCancelled):
                await application.run_turn(
                    "slow river essay", cancel=token, event_sink=events.append
                )
            return await application.run_turn("quick hello", cancel=CancellationToken())

    result = asyncio.run(run())

    assert result.message == "answer to quick hello"
    assert TurnFailed("Cancelled.") in events
    assert backend.prompts == ["slow river essay", "quick hello"]
    remembered = MemoryStore(database).search_memories("river essay")
    assert all("river" not in record.content for record in remembered)


# Agent loop and tools


class _AgentBackend:
    def __init__(self, tools: list[str]) -> None:
        self.tools = list(tools)
        self.calls = 0

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.calls += 1
        if self.tools:
            tool = self.tools.pop(0)
            return LLMResponse(json.dumps({"action": "tool", "tool": tool, "args": {}}))
        return LLMResponse('{"action":"finish","message":"done"}')


class _Tools:
    def __init__(self, token: CancellationToken | None = None, pid_file: Path | None = None):
        self.token = token
        self.pid_file = pid_file
        self.executed: list[str] = []

    def execute(self, tool: str, args: dict) -> ToolResult:
        self.executed.append(tool)
        if tool == "wait":
            _wait_for_cancel()
        if tool == "interrupted_quick":  # Ctrl+C lands while an uncancellable tool runs
            self.token.cancel()
        if tool == "subprocess":
            write_pid = f"open({str(self.pid_file)!r}, 'w').write(str(os.getpid()))"
            code = f"import os, time; {write_pid}; time.sleep(30)"
            SubprocessCommandRunner().run([sys.executable, "-c", code], timeout=60)
        return ToolResult(success=True, output="ok")


def test_cancel_tool_execution():
    token = CancellationToken()
    backend, tools = _AgentBackend(["wait", "never"]), _Tools()
    _cancel_soon(token)

    with pytest.raises(TurnCancelled):
        run_agent(backend, tools, "do it", cancel=token)

    assert tools.executed == ["wait"]
    assert backend.calls == 1


def test_cancel_subprocess_tool_terminates_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("assistant.tools.macos.sys.platform", "darwin")
    pid_file = tmp_path / "pid"
    token = CancellationToken()
    backend, tools = _AgentBackend(["subprocess"]), _Tools(pid_file=pid_file)
    _cancel_soon(token, 0.5)
    started = time.monotonic()

    with pytest.raises(TurnCancelled):
        run_agent(backend, tools, "run it", cancel=token)

    assert time.monotonic() - started < 5
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_between_tool_steps_stops_before_the_next_model_request():
    token = CancellationToken()
    backend = _AgentBackend(["interrupted_quick", "never"])
    tools = _Tools(token)

    with pytest.raises(TurnCancelled):
        run_agent(backend, tools, "do it", cancel=token)

    assert tools.executed == ["interrupted_quick"]
    assert backend.calls == 1


def test_agent_runs_normally_after_a_cancelled_turn():
    token = CancellationToken()
    backend, tools = _AgentBackend(["wait"]), _Tools()
    _cancel_soon(token)
    with pytest.raises(TurnCancelled):
        run_agent(backend, tools, "first", cancel=token)

    backend.tools = ["quick"]
    assert run_agent(backend, tools, "second", cancel=CancellationToken()) == "done"
    assert tools.executed == ["wait", "quick"]


# Interactive terminal: real Ctrl+C and Ctrl+D keys through a pseudo-terminal

CHILD = """
import asyncio, pathlib, sys, time
from assistant.cancellation import current_token
from assistant.config import ProjectConfig
from assistant.llm.base import LLMResponse
from assistant.runtime import JamieApplication
from assistant.ui.repl import create_prompt_session, run_repl

class Backend:
    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        prompt = messages[-1].content
        if prompt.startswith("slow"):
            print("TURN RUNNING", flush=True)
            token = current_token()
            for _ in range(500):
                token.raise_if_cancelled()
                time.sleep(0.02)
        return LLMResponse(f"answer to {prompt}")

root = pathlib.Path(sys.argv[1])
app = JamieApplication(ProjectConfig(memory_path=str(root / "m.sqlite3")), backend_factory=Backend)

async def main():
    async with app:
        await run_repl(app, session=create_prompt_session(root / "history"))

asyncio.run(main())
print("EXITED")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pseudo-terminal")
def test_terminal_ctrl_c_cancels_the_turn_and_the_next_prompt_works(tmp_path: Path):
    import pty

    pid, fd = pty.fork()
    if pid == 0:  # child
        os.execv(sys.executable, [sys.executable, "-c", CHILD, str(tmp_path)])
    output = b""

    def screen() -> str:
        text = re.sub(rb"\x1b\[[0-9;?]*[a-zA-Z]|\r", b"", output)
        return text.decode(errors="replace")

    def read(seconds: float) -> None:
        # Keep draining: a full pty buffer would block the child mid-redraw.
        nonlocal output
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    output += os.read(fd, 4096)
                except OSError:
                    return

    def wait_for(text: str, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        while text not in screen():
            assert time.monotonic() < deadline, f"missing {text!r} in:\n{screen()}"
            read(0.05)

    try:
        wait_for("You ›")
        os.write(fd, b"draft\x03")  # Ctrl+C at the prompt discards the input
        read(0.3)
        os.write(fd, b"slow essay\r")
        wait_for("TURN RUNNING")
        os.write(fd, b"\x04")  # Ctrl+D during a turn does not exit
        read(0.2)
        os.write(fd, b"\x03")  # Ctrl+C during a turn cancels it
        wait_for("Cancelled.")
        os.write(fd, b"quick\r")
        wait_for("Jamie › answer to quick")
        os.write(fd, b"\x04")  # Ctrl+D at the prompt exits
        wait_for("EXITED")
    finally:
        done, status = os.waitpid(pid, os.WNOHANG)
        if not done:
            time.sleep(0.5)
            done, status = os.waitpid(pid, os.WNOHANG)
        if not done:  # a failed assertion must not leave the child running
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
        os.close(fd)

    assert os.waitstatus_to_exitcode(status) == 0
    assert "answer to slow" not in screen()
    assert "answer to draft" not in screen()
