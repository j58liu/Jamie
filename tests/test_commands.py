import io
import os
import re
import time
from pathlib import Path

import pytest
from rich.console import Console

from assistant.llm.base import Message
from assistant.ui.commands import (
    ReplState,
    create_command_registry,
)


def _state() -> ReplState:
    return ReplState(Console(file=io.StringIO(), width=200, no_color=True))


def _run(state: ReplState, line: str) -> str:
    assert create_command_registry().dispatch(state, line)
    return state.console.file.getvalue()


def test_help_lists_every_registered_command():
    registry = create_command_registry()
    output = _run(_state(), "/help")

    for command in registry:
        assert command.name in output
        assert command.description in output


def test_quit_alias_and_case_insensitive_dispatch():
    state = _state()
    _run(state, "/EXIT")
    assert state.running is False


def test_prompts_are_not_commands():
    registry = create_command_registry()
    assert not registry.dispatch(_state(), "hello /help")
    assert not registry.dispatch(_state(), "/tmp/file.txt summarize")


def test_new_clears_conversation_but_keeps_settings():
    state = _state()
    state.history = [Message("user", "hi"), Message("assistant", "hello")]
    state.thinking = False

    _run(state, "/new")

    assert state.history == []
    assert state.thinking is False


def test_token_limit_command_validates_without_calling_the_model():
    state = _state()
    assert "Maximum answer tokens: 256" in _run(state, "/tokens 256")
    for value in ("0", "-1", "32769", "many"):
        assert "Usage: /tokens" in _run(state, f"/tokens {value}")
        assert state.max_tokens == 256


def test_context_describes_the_current_conversation():
    state = _state()
    state.history = [Message("user", "a" * 40), Message("assistant", "b" * 40)]

    output = _run(state, "/context")

    assert "Messages: 2 (1 exchanges)" in output
    assert "Used:      ~20 tokens (estimated)" in output
    assert "Maximum:    unknown" in output
    assert "Use /compact to summarize the conversation" in output


def test_cwd_shows_changes_and_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    (tmp_path / "file.txt").write_text("x")
    monkeypatch.setenv("HOME", str(tmp_path))
    state = _state()

    assert str(tmp_path) in _run(state, "/cwd")
    _run(state, "/cwd ~/project")
    assert Path.cwd() == (tmp_path / "project").resolve()

    assert "Not a directory:" in _run(_state(), "/cwd ../file.txt")
    assert "Not a directory:" in _run(_state(), "/cwd missing")
    assert Path.cwd() == (tmp_path / "project").resolve()


def test_cwd_rejects_unreadable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o000)
    try:
        if os.access(locked, os.X_OK):
            pytest.skip("running with permissions that bypass directory modes")
        assert "Cannot open directory" in _run(_state(), f"/cwd {locked}")
        assert Path.cwd() == tmp_path.resolve()
    finally:
        locked.chmod(0o700)


def test_tools_and_permissions_come_from_the_policy():
    state = _state()
    output = _run(state, "/tools") + _run(state, "/permissions")

    assert "read_file" in output and "READ" in output
    assert "READ (runs without asking)" in output
    assert "EXTERNAL (runs without asking): fetch_url, research, web_search" in output
    assert "asks for approval" not in output  # no registered tool is irreversible


def test_thinking_and_model_report_settings():
    state = _state()
    output = _run(state, "/thinking off") + _run(state, "/model")

    assert state.thinking is False
    assert "Model: reported after the first reply" in output
    assert "Thinking: off" in output


def test_stats_layout_matches_the_session_statistics_example(
    monkeypatch: pytest.MonkeyPatch,
):
    class Application:
        model_calls = 14

    state = _state()
    state.application = Application()
    state.turns, state.output_tokens, state.turn_seconds = 7, 2_481, 60.9
    state.tool_calls, state.tool_seconds = 9, 2.7
    state.first_token_seconds = [1.2, 1.6]
    state.startup_seconds = 0.8
    monkeypatch.setattr(time, "monotonic", lambda: state.started + 18 * 60)

    lines = _run(state, "/stats").splitlines()

    assert lines == [
        "Session statistics",
        "",
        "Turns                     7",
        "Model calls              14",
        "Tool calls                9",
        "Generated tokens      2,481",
        "Average first token   1.4 s",
        "Average turn          8.7 s",
        "Average tool          0.3 s",
        "Tokens per second      40.7",
        "Startup               0.8 s",
        "Session age          18 min",
    ]


def test_failed_and_cancelled_rows_appear_only_when_nonzero():
    assert "Failed turns" not in _run(_state(), "/stats")

    state = _state()
    state.failures, state.cancelled = 1, 2
    output = _run(state, "/stats")

    assert re.search(r"Failed turns\s+1\n", output)
    assert re.search(r"Cancelled turns\s+2\n", output)
