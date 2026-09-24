import asyncio
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from assistant.ui.prompt import JamieCompleter, command_preview
from assistant.ui.repl import COMMANDS, create_prompt_session


def _suggest(text: str) -> list[str]:
    completer = JamieCompleter(COMMANDS)
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


def test_slash_and_path_completion_use_the_current_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    (tmp_path / "notes.md").write_text("notes")

    assert _suggest("/thi") == ["/thinking", "/thinking on", "/thinking off"]
    assert "ject" in _suggest("/cwd ./pro")
    assert "tes.md" not in _suggest("/cwd ./no")
    assert "tes.md" in _suggest("Summarize ./no")
    assert "tes.md" in _suggest("First line\nSummarize ./no")
    assert _suggest("First line\n/thi") == []
    assert _suggest("ordinary prompt") == []
    assert _suggest("/cwd ./missing/") == []


def test_command_preview_is_dimmed_only_for_one_matching_command() -> None:
    assert command_preview(COMMANDS, "/thi").startswith("/thinking [on|off] · ")
    assert command_preview(COMMANDS, "/cwd ~/Documents").startswith("/cwd [path] · ")
    assert command_preview(COMMANDS, "/") == ""
    assert command_preview(COMMANDS, "/Users/me/file") == ""
    assert command_preview(COMMANDS, "hello") == ""


def test_alt_enter_adds_a_line_and_enter_sends(tmp_path: Path) -> None:
    async def run() -> str:
        with create_pipe_input() as pipe:
            pipe.send_text("first line\x1b\rsecond line\r")
            session = create_prompt_session(tmp_path / "history", input=pipe, output=DummyOutput())
            return await session.prompt_async("You › ")

    assert asyncio.run(run()) == "first line\nsecond line"


def test_tab_completion_is_applied_before_enter_sends(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()

    async def run(keys: str) -> str:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            session = create_prompt_session(tmp_path / "history", input=pipe, output=DummyOutput())
            return await session.prompt_async("You › ")

    assert asyncio.run(run("/cwd ./pro\t\r")) == "/cwd ./project"
    assert asyncio.run(run("/thi\t\r")) == "/thinking"
