import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.llm.base import LLMResponse
from assistant.tools import CommandResult, create_macos_registry

runner = CliRunner()


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                '{"action":"tool","tool":"open_application","args":{"name":"Safari"}}'
            )
        if '"success":false' in messages[-1].content:
            return LLMResponse('{"action":"finish","message":"Safari was not opened."}')
        return LLMResponse('{"action":"finish","message":"Safari was opened."}')


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append((arguments, input_text, timeout))
        return CommandResult(stdout="", stderr="")


@pytest.fixture
def agent_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(tmp_path / "memory.sqlite3"))
    backend = FakeBackend()
    command_runner = FakeRunner()
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: backend)

    def registry(root, *, approvals, web, backend):
        return create_macos_registry(root, approvals=approvals, runner=command_runner)

    monkeypatch.setattr(cli, "create_agent_registry", registry)
    return backend, command_runner


def test_agent_cli_asks_user_before_opening_application(agent_cli):
    backend, command_runner = agent_cli
    result = runner.invoke(cli.app, ["agent", "open Safari"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Approval required" in result.output
    assert "Jamie wants to open:" in result.output
    assert "Safari" in result.output
    assert "Tool: open_application" in result.output
    assert "Risk: SAFE_WRITE" in result.output
    assert "[y] allow once  [a] allow session  [n] deny" in result.output
    assert "Safari was opened." in result.output
    assert command_runner.calls == [(["open", "-a", "Safari"], None, 30)]
    assert backend.calls == 2
    with sqlite3.connect(Path.cwd() / "memory.sqlite3") as connection:
        tool = connection.execute("SELECT tool, success FROM tool_calls").fetchone()
    assert tool == ("open_application", 1)


def test_prompt_without_command_runs_agent(agent_cli):
    backend, command_runner = agent_cli
    result = runner.invoke(cli.app, ["open Safari"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Approval required" in result.output
    assert "Safari was opened." in result.output
    assert command_runner.calls == [(["open", "-a", "Safari"], None, 30)]
    assert backend.calls == 2


def test_agent_cli_denial_prevents_application_launch(agent_cli):
    backend, command_runner = agent_cli
    result = runner.invoke(cli.app, ["agent", "open Safari"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Stopped: open_application was not approved." in result.output
    assert command_runner.calls == []
    assert backend.calls == 1  # a denial ends the turn instead of asking the model again


@pytest.mark.parametrize(
    "arguments",
    [
        [""],
        ["   "],
        ["open Safari", "--max-steps", "0"],
        ["open Safari", "--timeout", "0"],
        ["open Safari", "--timeout", "nan"],
    ],
)
def test_agent_cli_rejects_invalid_options_without_model(agent_cli, arguments):
    backend, command_runner = agent_cli
    result = runner.invoke(cli.app, ["agent", *arguments])
    assert result.exit_code == 2
    assert backend.calls == 0
    assert command_runner.calls == []
