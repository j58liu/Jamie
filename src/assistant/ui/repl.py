"""Interactive prompt_toolkit loop over one initialized application."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from assistant import __version__
from assistant.llm.base import LLMError, Message
from assistant.memory import MemoryStoreError, default_memory_path
from assistant.runtime import JamieApplication, JamieApplicationError
from assistant.ui.commands import ReplState, create_command_registry
from assistant.ui.renderer import EventRenderer

COMMANDS = create_command_registry()


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
        completer=COMMANDS.completer(),
        **options,
    )


async def run_repl(
    application: JamieApplication,
    *,
    console: Console | None = None,
    session: PromptSession | None = None,
) -> None:
    """Send prompts to the already initialized application until the user exits."""
    state = ReplState(console or Console())
    session = session or create_prompt_session()
    renderer = EventRenderer(state.console, answer_prefix="Jamie › ")

    state.say(f"Jamie {__version__}")
    state.say("Type /help for commands. Ctrl+D or /quit to leave.")

    while state.running:
        try:
            prompt = (await session.prompt_async("You › ")).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not prompt or COMMANDS.dispatch(state, prompt):
            continue
        started = time.monotonic()
        try:
            result = await application.run_turn(
                prompt,
                history=state.history,
                temperature=state.temperature,
                max_tokens=state.max_tokens,
                thinking=state.thinking,
                event_sink=renderer.handle,
            )
        except (JamieApplicationError, LLMError, MemoryStoreError, ValueError) as exc:
            state.failures += 1
            state.say(f"Error: {exc}")
            continue
        finally:
            state.model_seconds += time.monotonic() - started
        state.turns += 1
        state.output_tokens += result.output_tokens or 0
        state.model = result.model or state.model
        # ponytail: transcript grows unbounded; trim when llama.cpp context overflows.
        state.history += [Message("user", prompt), Message("assistant", result.message)]
