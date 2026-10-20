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
from assistant.cancellation import CancellationToken, cancellation_scope
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, LLMResponse, ServerInfo
from assistant.runtime import JamieApplication
from assistant.runtime.workspaces import WorkspaceTools
from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel
from assistant.tools.registry import create_workspace_registry
from assistant.ui import gui
from assistant.ui.gui import (
    GuiApprovalProvider,
    GuiSession,
    StartupFailed,
    ensure_server,
    event_payload,
    page_html,
)


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


def kinds(pushed):
    return [payload["type"] for payload in pushed]


@pytest.fixture
def session_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def build(backend, pending=None, push=None, approvals=None, **options):
        application = JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: backend,
        )
        asyncio.run(application.start())
        pushed = []
        # Run each turn to completion, or hold it to test a turn that is still running.
        submit = pending.append if pending is not None else asyncio.run
        tools = WorkspaceTools(lambda: create_workspace_registry(tmp_path, approvals=approvals))
        session = GuiSession(application, tools, push or pushed.append, submit, **options)
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

    assert kinds(pushed) == ["tool", "tool", "answer", "idle", "answer", "idle"]
    assert pushed[1]["text"].startswith("Read notes.md · 1 line")
    assert pushed[2]["text"] == "It says hello."
    assert pushed[3] == {"type": "idle", "workspace": str(tmp_path)}
    last_request = [m.content for m in backend.requests[-1]]
    assert "what does notes.md say?" in last_request and "It says hello." in last_request
    assert session.state.turns == 2 and session.state.tool_calls == 1


def test_one_turn_at_a_time_and_new_conversation_waits(session_for):
    pending = []
    session, pushed = session_for(Backend(finish("first"), finish("second")), pending)

    assert session.send("one")
    assert not session.send("two")  # still running
    assert not session.send("   ")
    assert not session.new_conversation()
    asyncio.run(pending.pop())

    assert pushed[-1]["type"] == "idle"
    assert session.state.history and session.new_conversation()
    assert session.state.history == [] and session.state.conversation_id is None


def test_cancel_stops_the_turn_and_the_next_one_runs(session_for):
    pending = []
    backend = Backend(finish("ready"))
    session, pushed = session_for(backend, pending)

    session.send("slow task")
    session.cancel()
    asyncio.run(pending.pop())
    session.send("again")
    asyncio.run(pending.pop())

    assert kinds(pushed) == ["cancelled", "idle", "answer", "idle"]
    assert pushed[2]["text"] == "ready"
    assert len(backend.requests) == 1  # cancelled before the model was asked
    assert [m.content for m in session.state.history] == ["again", "ready"]
    assert session.state.cancelled == 1


def test_failures_are_shown_and_the_window_recovers(session_for):
    session, pushed = session_for(Backend(LLMError("llama.cpp went away"), finish("back")))

    session.send("first")
    session.send("second")

    assert pushed[0] == {"type": "error", "lines": ["llama.cpp went away"]}
    assert kinds(pushed[1:]) == ["idle", "answer", "idle"]
    assert session.state.failures == 1


def test_a_closed_window_does_not_fail_the_turn(session_for):
    def closed(payload):
        raise RuntimeError("window closed")

    session, _ = session_for(Backend(finish("done")), push=closed)

    assert session.send("hi")
    assert session.state.history[-1].content == "done"


def test_commands_print_into_the_window_without_calling_the_model(session_for):
    backend = Backend()
    session, pushed = session_for(backend)

    session.send("/help")
    session.send("/mode chat")
    session.send("/nonsense")

    outputs = [p["text"] for p in pushed if p["type"] == "command"]
    assert "/resume <n|id>" in outputs[0] and outputs[1] == "Mode: chat"
    assert outputs[2] == "Unknown command /nonsense. Type /help for commands."
    assert kinds(pushed).count("idle") == 3 and backend.requests == []
    assert session.state.mode == "chat"


def test_resume_and_new_redraw_the_conversation(session_for):
    session, pushed = session_for(Backend(finish("Paris.")))
    session.send("capital of France?")
    saved = session.state.conversation_id
    session.send("/new")
    session.send("/resume 1")

    redraws = [(p["messages"], p["note"]) for p in pushed if p["type"] == "conversation"]
    (cleared, started), (restored, resumed) = redraws
    assert cleared == [] and started == "Started a new conversation."
    assert restored == [
        {"role": "user", "text": "capital of France?"},
        {"role": "assistant", "text": "Paris."},
    ]
    assert resumed.startswith("Resumed: capital of France? (1 exchanges)")
    assert session.state.conversation_id == saved


def test_clear_and_quit_reach_the_window(session_for):
    quit_calls = []
    session, pushed = session_for(Backend(), on_quit=lambda: quit_calls.append(1))

    session.send("/clear")
    session.send("/quit")

    assert kinds(pushed) == ["clear", "idle", "idle"] and quit_calls == [1]


def test_local_commands_run_tools_without_the_model(session_for, tmp_path):
    (tmp_path / "notes.md").write_text("alpha\nbeta\n")
    backend = Backend()
    session, pushed = session_for(backend)

    session.send("/read notes.md")

    output = next(p["text"] for p in pushed if p["type"] == "command")
    assert "Read notes.md" in output and "beta" in output
    assert backend.requests == []


def answering(provider, choice, pushed):
    """A window that clicks ``choice`` as soon as an approval card appears."""

    def push(payload):
        pushed.append(payload)
        if payload["type"] == "approval":
            provider.answer(payload["id"], choice)

    return push


def approval(tool="write_file", permission=PermissionLevel.SAFE_WRITE, **arguments):
    return ApprovalRequest(tool=tool, permission=permission, arguments=arguments)


def test_approval_card_shows_the_exact_action_and_waits_for_a_click():
    pushed = []
    provider = GuiApprovalProvider(lambda payload: pushed.append(payload))
    provider._push = answering(provider, "once", pushed)

    assert provider.request(approval(path="notes.md", content="hi"))

    card, closed = pushed
    assert card["heading"] == "Jamie wants to write:" and card["subject"] == "notes.md"
    assert [tuple(detail) for detail in card["details"]] == [("content", "hi")]
    assert card["session"] is True and "SAFE_WRITE" in card["footer"]
    assert closed == {"type": "approval-closed", "id": card["id"], "choice": "once"}


def test_denial_and_session_approval():
    pushed = []
    provider = GuiApprovalProvider(pushed.append)
    provider._push = answering(provider, "deny", pushed)
    assert not provider.request(approval(path="a"))

    provider._push = answering(provider, "session", pushed)
    assert provider.request(approval(path="b"))
    pushed.clear()
    assert provider.request(approval(path="c")) and pushed == []  # remembered for the tool


def test_answers_must_match_a_waiting_request():
    pushed = []
    provider = GuiApprovalProvider(pushed.append)

    assert not provider.answer("unknown", "once")

    def push(payload):
        pushed.append(payload)
        if payload["type"] == "approval":
            assert not provider.answer(payload["id"], "always")  # not a choice
            assert provider.answer(payload["id"], "deny")
            assert not provider.answer(payload["id"], "once")  # already answered

    provider._push = push
    assert not provider.request(approval(path="x"))


def test_cancelling_the_turn_denies_a_waiting_approval():
    token = CancellationToken()
    pushed = []
    provider = GuiApprovalProvider(lambda payload: (pushed.append(payload), token.cancel()))

    with cancellation_scope(token):
        assert not provider.request(approval(path="x"))
    assert pushed[-1]["choice"] == "deny"


@pytest.mark.usefixtures("confirm_every_tool")
def test_a_turn_waits_for_approval_and_a_denial_ends_it(session_for, tmp_path):
    pushed = []
    provider = GuiApprovalProvider(pushed.append)
    write = {"action": "tool", "tool": "write_file", "args": {"path": "a.txt", "content": "1"}}
    read = {"action": "tool", "tool": "read_file", "args": {"path": "a.txt"}}  # verification
    session, _ = session_for(
        Backend(write, read, finish("Wrote it."), write), push=pushed.append, approvals=provider
    )

    provider._push = answering(provider, "once", pushed)
    session.send("write a.txt")
    assert (tmp_path / "a.txt").read_text() == "1"

    provider._push = answering(provider, "deny", pushed)
    (tmp_path / "a.txt").unlink()
    session.send("write it again")

    assert not (tmp_path / "a.txt").exists()
    answers = [p["text"] for p in pushed if p["type"] == "answer"]
    assert answers == ["Wrote it.", "Stopped: write_file was not approved."]


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
