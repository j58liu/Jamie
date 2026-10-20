import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.agent.events import (
    ApprovalRequested,
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
)
from assistant.agent.loop import ToolResult
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, ServerInfo
from assistant.runtime import JamieApplication
from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel
from assistant.tools.registry import create_workspace_registry
from assistant.ui import gui
from assistant.ui.gui import GuiSession, StartupFailed, ensure_server, event_payload, page_html


def test_payloads_describe_tool_steps_without_their_output():
    started = event_payload(ToolStarted("read_file", {"path": "notes.md"}))
    finished = event_payload(
        ToolFinished("read_file", ToolResult(success=True, output="secret\ntext"), 0.2),
        {"path": "notes.md"},
    )
    failed = event_payload(ToolFailed("read_file", "Path does not exist", 0.1), {"path": "x"})

    assert started == {"type": "tool", "state": "running", "text": "Reading notes.md"}
    assert finished == {"type": "tool", "state": "done", "text": "Read notes.md · 2 lines · 0.2 s"}
    assert "secret" not in json.dumps(finished)
    assert failed == {
        "type": "tool",
        "state": "failed",
        "text": "Reading x failed · 0.1 s",
        "detail": ["Path does not exist"],
    }


def test_payloads_carry_answer_text_and_skip_events_shown_elsewhere():
    request = ApprovalRequest(
        tool="write_file", permission=PermissionLevel.SAFE_WRITE, arguments={}
    )

    assert event_payload(AssistantDelta("Hel")) == {"type": "delta", "text": "Hel"}
    assert event_payload(AssistantFinished("Hi", finish_reason="length")) == {
        "type": "answer",
        "text": "Hi",
        "truncated": True,
    }
    assert event_payload(TurnFailed("boom")) is None  # reported from the turn's exception
    assert event_payload(ApprovalRequested(request)) is None


class Backend:
    """Answers each agent request with the next scripted reply."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(json.dumps(reply))


def finish(message):
    return {"action": "finish", "message": message}


@pytest.fixture
def session_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def build(backend, pending=None):
        application = JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: backend,
        )
        asyncio.run(application.start())
        pushed = []
        # Run each turn to completion, or hold it to test a turn that is still running.
        submit = pending.append if pending is not None else asyncio.run
        tools = create_workspace_registry(tmp_path)
        session = GuiSession(application, tools, pushed.append, submit)
        return session, pushed

    return build


def test_turns_keep_the_conversation_and_end_idle(session_for, tmp_path):
    (tmp_path / "notes.md").write_text("hello\n")
    backend = Backend(
        {"action": "tool", "tool": "read_file", "args": {"path": "notes.md"}},
        finish("It says hello."),
        finish("You asked about notes.md."),
    )
    session, pushed = session_for(backend)

    assert session.send("  what does notes.md say?  ")
    assert session.send("what did I ask?")

    assert [p["type"] for p in pushed] == ["tool", "tool", "answer", "idle", "answer", "idle"]
    assert pushed[1]["text"].startswith("Read notes.md · 1 line")
    assert pushed[2]["text"] == "It says hello."
    last_request = [m.content for m in backend.requests[-1]]
    assert "what does notes.md say?" in last_request and "It says hello." in last_request


def test_one_turn_at_a_time_and_new_conversation_waits(session_for):
    pending = []
    session, pushed = session_for(Backend(finish("first"), finish("second")), pending)

    assert session.send("one")
    assert not session.send("two")  # still running
    assert not session.send("   ")
    assert not session.new_conversation()
    asyncio.run(pending.pop())

    assert pushed[-1] == {"type": "idle"}
    assert session.history and session.new_conversation()
    assert session.history == [] and session.conversation_id is None


def test_cancel_stops_the_turn_and_the_next_one_runs(session_for):
    pending = []
    backend = Backend(finish("ready"))
    session, pushed = session_for(backend, pending)

    session.send("slow task")
    session.cancel()
    asyncio.run(pending.pop())
    session.send("again")
    asyncio.run(pending.pop())

    assert [p["type"] for p in pushed] == ["cancelled", "idle", "answer", "idle"]
    assert pushed[2]["text"] == "ready"
    assert len(backend.requests) == 1  # cancelled before the model was asked
    assert [m.content for m in session.history] == ["again", "ready"]


def test_failures_are_shown_and_the_window_recovers(session_for):
    session, pushed = session_for(Backend(LLMError("llama.cpp went away"), finish("back")))

    session.send("first")
    session.send("second")

    assert pushed[0] == {"type": "error", "lines": ["llama.cpp went away"]}
    assert [p["type"] for p in pushed[1:]] == ["idle", "answer", "idle"]


def test_a_closed_window_does_not_fail_the_turn(session_for, tmp_path):
    backend = Backend(finish("done"))
    application = JamieApplication(
        ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
        backend_factory=lambda: backend,
    )
    asyncio.run(application.start())

    def closed(payload):
        raise RuntimeError("window closed")

    session = GuiSession(application, create_workspace_registry(tmp_path), closed, asyncio.run)

    assert session.send("hi")
    assert session.history[-1].content == "done"


class Server:
    def __init__(self, *states):
        self.states = list(states)

    def server_info(self):
        state = self.states.pop(0)
        if state == "down":
            raise LLMError("Cannot reach llama.cpp.")
        return ServerInfo("Qwen3.5-2B", None, 32768)


class Process:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def test_a_running_server_is_used_without_starting_another():
    started = []
    info = ensure_server(Server("up"), print, start=lambda: started.append(1))

    assert info.model == "Qwen3.5-2B" and started == []


def test_a_missing_server_is_started_and_waited_for():
    statuses, started = [], []

    def start():
        started.append(1)
        return Process()

    info = ensure_server(Server("down", "down", "up"), statuses.append, start=start, interval=0)

    assert info.context == 32768
    assert statuses == ["Starting the local model…"] and started == [1]


def test_a_server_that_exits_or_never_loads_is_reported(tmp_path, monkeypatch):
    log = tmp_path / "llama-server.log"
    log.write_text("loading\nerror: model file not found\n")
    monkeypatch.setattr(gui, "MODEL_LOG", log)

    with pytest.raises(StartupFailed, match="model file not found"):
        ensure_server(Server("down", "down"), print, start=lambda: Process(1), interval=0)
    with pytest.raises(StartupFailed, match="did not load in time"):
        ensure_server(Server("down", "down"), print, start=Process, timeout=0, interval=0)


def test_the_start_script_runs_detached_with_homebrew_on_the_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "MODEL_LOG", tmp_path / "logs" / "llama-server.log")
    monkeypatch.setattr(subprocess, "Popen", lambda args, **options: calls.append((args, options)))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    gui.start_model_server()

    ((args, options),) = calls
    assert args == [str(gui.MODEL_SCRIPT)] and Path(args[0]).is_file()
    assert options["start_new_session"] is True
    assert options["env"]["PATH"].startswith("/opt/homebrew/bin:")


def test_the_page_embeds_its_fonts_and_escapes_the_workspace():
    page = page_html("~/<b>work</b>")

    assert page.count("data:font/woff2;base64,") == 3
    assert "~/&lt;b&gt;work&lt;/b&gt;" in page and "<b>work" not in page
    assert "{{" not in page
    # Nothing loads from outside the page; model text is inserted as text, not markup.
    assert 'src="http' not in page and "url(http" not in page
    assert ".innerHTML" not in page and "default-src 'none'" in page


def test_gui_command_reports_bad_configuration_without_a_window(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "run_gui", opened.append)
    bad = tmp_path / "assistant.toml"
    bad.write_text("unknown = 1\n")

    result = CliRunner().invoke(cli.app, ["gui", "--config", str(bad)])

    assert result.exit_code == 1 and "Invalid configuration" in result.output
    assert opened == []
