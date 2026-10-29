import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.llm.base import LLMResponse

runner = CliRunner()


class MemoryBackend:
    def __init__(self):
        self.calls = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.calls.append(list(messages))
        prompt = messages[-1].content
        if messages[0].content.startswith("You extract long-term memory"):
            return LLMResponse('{"items": []}')  # the memory extraction pass
        if "Remember that" in prompt:
            return LLMResponse(
                '{"action":"finish","message":"I will remember that we chose SQLite."}'
            )
        memory_messages = [message.content for message in messages if message.role == "system"]
        assert any("SQLite" in message for message in memory_messages)
        return LLMResponse(
            '{"action":"finish","message":"We discussed SQLite as the database library."}'
        )


def test_agent_recalls_a_previous_cli_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(database))
    backend = MemoryBackend()
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: backend)

    first = runner.invoke(
        cli.app,
        ["agent", "Remember that the database library we chose is SQLite."],
    )
    second = runner.invoke(
        cli.app,
        ["agent", "What was the database library we discussed?"],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "SQLite" in second.stdout
    assert len(backend.calls) == 3  # two answers plus one extraction for "Remember that"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
