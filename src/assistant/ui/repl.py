"""Interactive prompt_toolkit loop over one initialized application."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from rich.console import Console

from assistant import __version__
from assistant.llm.base import LLMError, Message
from assistant.memory import MemoryStoreError
from assistant.runtime import JamieApplication, JamieApplicationError
from assistant.ui.renderer import ReplRenderer

EXIT_COMMANDS = frozenset({"/quit", "/exit"})


async def run_repl(
    application: JamieApplication,
    *,
    console: Console | None = None,
    session: PromptSession | None = None,
) -> None:
    """Send prompts to the already initialized application until the user exits."""
    console = console or Console()
    session = session or PromptSession()
    renderer = ReplRenderer(console)
    history: list[Message] = []

    console.print(f"Jamie {__version__}", markup=False, highlight=False)
    console.print("Ctrl+D, /quit, or /exit to leave.", markup=False, highlight=False)

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
        try:
            result = await application.run_turn(
                prompt,
                history=history,
                event_sink=renderer.handle,
            )
        except (JamieApplicationError, LLMError, MemoryStoreError, ValueError) as exc:
            console.print(f"Error: {exc}", markup=False, highlight=False)
            continue
        # ponytail: transcript grows unbounded; trim when llama.cpp context overflows.
        history += [Message("user", prompt), Message("assistant", result.message)]
