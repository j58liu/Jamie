import io
import os
from pathlib import Path

import pytest
from rich.console import Console

from assistant.llm.base import Message
from assistant.ui.commands import (
    CommandRegistry,
    ReplState,
    SlashCommand,
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


def test_registry_rejects_duplicate_names_and_aliases():
    registry = CommandRegistry()
    registry.register(SlashCommand("/quit", "Leave.", lambda s, a: None, aliases=("/exit",)))

    with pytest.raises(ValueError, match="/exit"):
        registry.register(SlashCommand("/exit", "Again.", lambda s, a: None))


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


def test_context_describes_the_current_conversation():
    state = _state()
    state.history = [Message("user", "a" * 40), Message("assistant", "b" * 40)]

    output = _run(state, "/context")

    assert "Messages: 2 (1 exchanges)" in output
    assert "Estimated tokens: ~20 (80 characters)" in output


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
    assert "asks for approval" in output


def test_thinking_and_model_report_settings():
    state = _state()
    output = _run(state, "/thinking off") + _run(state, "/model")

    assert state.thinking is False
    assert "Model: reported after the first reply" in output
    assert "Thinking: off" in output


def test_stats_reports_speed():
    state = _state()
    state.turns, state.output_tokens, state.model_seconds = 2, 100, 20.0

    output = _run(state, "/stats")

    assert "Turns: 2 completed, 0 failed" in output
    assert "Speed: 5.0 tokens/s" in output
