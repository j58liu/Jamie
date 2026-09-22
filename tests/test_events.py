import asyncio
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from threading import Event, get_ident

import pytest
from rich.console import Console

from assistant.agent.events import (
    ApprovalRequested,
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnStarted,
)
from assistant.agent.loop import AgentTimeoutError, ToolResult, run_agent
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse
from assistant.runtime import JamieApplication
from assistant.security import CallbackApprovalProvider
from assistant.tools import create_workspace_registry
from assistant.ui.renderer import OneShotRenderer


class Backend:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(response, model="local", finish_reason="stop", output_tokens=10)


class Tools:
    def __init__(self, result):
        self.result = result

    def execute(self, tool, args):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


CALL = '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}'
FINISH = '{"action":"finish","message":"Done."}'


def test_agent_success_event_order_and_caller_thread(capsys):
    events = []
    threads = []

    def handle(event):
        events.append(event)
        threads.append(get_ident())

    result = ToolResult(success=True, output="hello world")
    answer = run_agent(Backend(CALL, FINISH), Tools(result), "Read", event_sink=handle)

    assert answer == "Done."
    assert [type(event) for event in events] == [
        TurnStarted,
        ToolStarted,
        ToolFinished,
        AssistantFinished,
    ]
    assert events[1] == ToolStarted("read_file", {"path": "README.md"})
    assert events[2].result == result
    assert events[2].duration >= 0
    assert events[3].text == answer
    assert threads == [get_ident()] * 4
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("raises", [False, True])
def test_failed_tools_emit_failure_instead_of_finished(raises):
    events = []
    result = (
        RuntimeError("read failed") if raises else ToolResult(success=False, output="read failed")
    )
    if raises:
        with pytest.raises(RuntimeError, match="read failed"):
            run_agent(Backend(CALL), Tools(result), "Read", event_sink=events.append)
    else:
        run_agent(Backend(CALL, FINISH), Tools(result), "Read", event_sink=events.append)
    assert [type(event) for event in events] == [
        TurnStarted,
        ToolStarted,
        ToolFailed,
        TurnFailed if raises else AssistantFinished,
    ]
    assert events[2].error == "read failed"
    assert events[2].duration >= 0


@pytest.mark.parametrize("approved", [True, False])
def test_approval_event_keeps_existing_authorization(tmp_path: Path, approved):
    events = []
    requests = []

    def approve(request):
        requests.append(request)
        return approved

    registry = create_workspace_registry(tmp_path, approvals=CallbackApprovalProvider(approve))
    call = json.dumps({"action": "tool", "tool": "create_directory", "args": {"path": "new"}})
    verify = '{"action":"tool","tool":"list_directory","args":{"path":"new"}}'
    backend = Backend(call, verify, FINISH) if approved else Backend(call, FINISH)
    run_agent(backend, registry, "Create a directory", event_sink=events.append)
    assert [type(event) for event in events[:4]] == [
        TurnStarted,
        ToolStarted,
        ApprovalRequested,
        ToolFinished if approved else ToolFailed,
    ]
    assert len(requests) == 1
    assert events[2].request == requests[0]
    assert events[2].request.arguments == {"path": "new", "parents": False}
    assert (tmp_path / "new").exists() is approved


def test_raw_reasoning_and_invalid_actions_are_not_assistant_events():
    events = []
    backend = Backend(
        "<think>private reasoning</think>not a valid action",
        "<think>more private reasoning</think>" + CALL,
        "```json\n" + FINISH + "\n```",
    )
    run_agent(
        backend, Tools(ToolResult(success=True, output="file")), "Read", event_sink=events.append
    )
    assert [e for e in events if isinstance(e, AssistantFinished)] == [AssistantFinished("Done.")]
    assert not any(isinstance(e, AssistantDelta) for e in events)
    assert "private reasoning" not in repr(events)


def test_timeout_is_terminal_even_when_worker_finishes_later():
    release = Event()
    stopped = Event()
    events = []

    class SlowTools:
        def execute(self, tool, args):
            release.wait(2)
            return ToolResult(success=True, output="late")

        def end_turn(self):
            stopped.set()

    try:
        with pytest.raises(AgentTimeoutError):
            run_agent(
                Backend(CALL, FINISH),
                SlowTools(),
                "Read",
                timeout_seconds=0.05,
                event_sink=events.append,
            )
        assert [type(event) for event in events] == [TurnStarted, ToolStarted, TurnFailed]
        before = list(events)
    finally:
        release.set()
    assert stopped.wait(2)
    assert events == before


def test_runtime_events_preserve_result_and_hide_think_blocks(tmp_path: Path, capsys):
    events = []

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: Backend("<think>private reasoning</think>\nHello!"),
        ) as app:
            return await app.run_turn("hello", event_sink=events.append)

    result = asyncio.run(run())
    assert events == [TurnStarted("hello"), AssistantFinished("Hello!", "local", "stop", 10)]
    assert result.message == "Hello!"
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("response", [LLMError("offline"), "<think>unfinished reasoning"])
def test_runtime_failures_emit_no_answer(tmp_path: Path, response):
    events = []

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: Backend(response),
        ) as app:
            await app.run_turn("hello", event_sink=events.append)

    with pytest.raises(LLMError):
        asyncio.run(run())
    assert [type(event) for event in events] == [TurnStarted, TurnFailed]
    assert "unfinished reasoning" not in repr(events)


def test_one_shot_renderer_prints_literal_final_answer_once():
    output, errors = StringIO(), StringIO()
    renderer = OneShotRenderer(Console(file=output), Console(file=errors))
    renderer.handle(TurnStarted("hello"))
    renderer.handle(AssistantDelta("Hello"))
    renderer.handle(ToolFailed("read_file", "not found"))
    renderer.handle(AssistantFinished("Hello [bold]Jamie[/bold]", finish_reason="length"))
    assert output.getvalue() == "Hello [bold]Jamie[/bold]\n"
    assert errors.getvalue().count("--max-tokens") == 1


def test_events_import_without_terminal_packages():
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockTerminal(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"rich", "typer", "prompt_toolkit", "textual"}:
            raise AssertionError(f"Unexpected terminal dependency: {fullname}")

sys.meta_path.insert(0, BlockTerminal())
from assistant.agent.events import AgentEvent, TurnStarted
assert TurnStarted("hello").prompt == "hello"
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
