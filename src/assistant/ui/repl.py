"""Interactive prompt_toolkit loop over one initialized application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console

from assistant import __version__
from assistant.llm.base import LLMError, Message
from assistant.memory import MemoryStoreError, default_memory_path
from assistant.runtime import JamieApplication, JamieApplicationError
from assistant.ui.renderer import EventRenderer

EXIT_COMMANDS = frozenset({"/quit", "/exit"})
THINKING_MODES = {"on": True, "off": False}
# sentence=True matches the whole line, so "/" is not treated as a word break.
COMMANDS = WordCompleter(
    [*(f"/thinking {mode}" for mode in THINKING_MODES), *sorted(EXIT_COMMANDS)], sentence=True
)


def default_history_path() -> Path:
    """Keep typed prompts beside Jamie's memory database."""
    return default_memory_path().parent / "input_history"


class PromptHistory(FileHistory):
    """Typed prompts only; a leading space keeps a line out of the file, as in shells."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        super().__init__(path)

    def store_string(self, string: str) -> None:
        if not string.startswith(" "):
            super().store_string(string)


def create_prompt_session(history_path: Path | None = None, **options: Any) -> PromptSession:
    """Emacs line editing with persistent history and slash-command completion."""
    return PromptSession(
        history=PromptHistory(history_path or default_history_path()),
        completer=COMMANDS,
        **options,
    )


async def run_repl(
    application: JamieApplication,
    *,
    console: Console | None = None,
    session: PromptSession | None = None,
) -> None:
    """Send prompts to the already initialized application until the user exits."""
    console = console or Console()
    session = session or create_prompt_session()
    renderer = EventRenderer(console, answer_prefix="Jamie › ")
    history: list[Message] = []
    thinking: bool | None = None

    console.print(f"Jamie {__version__}", markup=False, highlight=False)
    console.print(
        "Ctrl+D, /quit, or /exit to leave. /thinking on|off toggles reasoning.",
        markup=False,
        highlight=False,
    )

    while True:
        try:
            prompt = (await session.prompt_async("You › ")).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not prompt:
            continue
        if prompt in EXIT_COMMANDS:
            break
        command, _, argument = prompt.partition(" ")
        if command == "/thinking":
            argument = argument.strip().lower()
            if argument in THINKING_MODES:
                thinking = THINKING_MODES[argument]
            elif argument:
                console.print("Usage: /thinking on|off", markup=False, highlight=False)
                continue
            state = "server default" if thinking is None else ("on" if thinking else "off")
            console.print(f"Thinking: {state}", markup=False, highlight=False)
            continue
        try:
            result = await application.run_turn(
                prompt,
                history=history,
                thinking=thinking,
                event_sink=renderer.handle,
            )
        except (JamieApplicationError, LLMError, MemoryStoreError, ValueError) as exc:
            console.print(f"Error: {exc}", markup=False, highlight=False)
            continue
        # ponytail: transcript grows unbounded; trim when llama.cpp context overflows.
        history += [Message("user", prompt), Message("assistant", result.message)]
