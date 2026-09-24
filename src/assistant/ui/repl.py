"""Interactive prompt_toolkit loop over one initialized application."""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console

from assistant import __version__
from assistant.agent.events import AgentEvent, ToolFailed, ToolFinished
from assistant.cancellation import CancellationToken, TurnCancelled
from assistant.llm.base import LLMError, Message
from assistant.memory import MemoryStoreError, default_memory_path
from assistant.runtime import JamieApplication, JamieApplicationError
from assistant.ui.commands import ReplState, create_command_registry, refresh_context
from assistant.ui.renderer import EventRenderer
from assistant.ui.status import status_line

COMMANDS = create_command_registry()
# Plain dim text instead of prompt_toolkit's default reverse-video bar.
TOOLBAR_STYLE = Style.from_dict({"bottom-toolbar": "noreverse #888888"})


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


@contextmanager
def interrupt_cancels(token: CancellationToken) -> Iterator[None]:
    """While a turn runs, Ctrl+C cancels it instead of stopping Jamie.

    At the prompt, prompt_toolkit reads Ctrl+C as a key, so no signal is involved.
    """
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    try:
        loop.add_signal_handler(signal.SIGINT, token.cancel)
    except (NotImplementedError, RuntimeError, ValueError):  # not the main thread
        yield
        return
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous)


async def run_repl(
    application: JamieApplication,
    *,
    console: Console | None = None,
    session: PromptSession | None = None,
    model: str | None = None,
    started: float | None = None,
) -> None:
    """Send prompts to the already initialized application until the user exits.

    ``started`` is the time.monotonic() value when Jamie began starting up.
    """
    state = ReplState(console or Console(), application=application, model=model)
    session = session or create_prompt_session()
    renderer = EventRenderer(state.console, answer_prefix="Jamie › ")

    def show_and_count(event: AgentEvent) -> None:
        if isinstance(event, ToolFinished | ToolFailed):
            state.tool_calls += 1
            state.tool_seconds += event.duration
        renderer.handle(event)

    state.say(f"Jamie {__version__}")
    state.say("Type /help for commands. Ctrl+D or /quit to leave.")

    if started is not None:
        state.startup_seconds = time.monotonic() - started
    while state.running:
        try:
            prompt = (
                await session.prompt_async(
                    "You › ",
                    bottom_toolbar=lambda: status_line(state),
                    style=TOOLBAR_STYLE,
                )
            ).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not prompt or COMMANDS.dispatch(state, prompt):
            continue
        turn_started = time.monotonic()
        token = CancellationToken()
        try:
            if state.conversation_id is None:
                state.conversation_id = application.start_conversation()
            with interrupt_cancels(token):
                result = await application.run_turn(
                    prompt,
                    history=state.history,
                    temperature=state.temperature,
                    max_tokens=state.max_tokens,
                    thinking=state.thinking,
                    cancel=token,
                    conversation_id=state.conversation_id,
                    event_sink=show_and_count,
                )
        except TurnCancelled:
            state.cancelled += 1
            state.say("Cancelled.")
            continue
        except (JamieApplicationError, LLMError, MemoryStoreError, ValueError) as exc:
            state.failures += 1
            state.say(f"Error: {exc}")
            continue
        state.turns += 1
        state.turn_seconds += time.monotonic() - turn_started
        state.output_tokens += result.output_tokens or 0
        if result.first_token_seconds is not None:
            state.first_token_seconds.append(result.first_token_seconds)
        state.model = result.model or state.model
        # ponytail: transcript grows unbounded; compaction is a separate task needing evals.
        state.history += [Message("user", prompt), Message("assistant", result.message)]
        refresh_context(state)  # a failure falls back to the estimate in the status bar
