"""Interactive prompt_toolkit loop over one initialized application."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.status import Status
from rich.text import Text

from assistant import __version__
from assistant.agent.events import (
    AgentEvent,
    ApprovalRequested,
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
)
from assistant.agent.loop import MAX_STEPS, AgentError, ToolExecutor
from assistant.cancellation import CancellationToken, TurnCancelled, interrupt_cancels
from assistant.llm.base import LLMError, Message
from assistant.memory import MemoryStoreError, default_memory_path
from assistant.runtime import JamieApplication, JamieApplicationError
from assistant.runtime.workspaces import WorkspaceTools
from assistant.tools.scopes import SCOPE_MAX_STEPS
from assistant.ui.commands import ReplState, create_command_registry, refresh_context
from assistant.ui.prompt import JamieCompleter, command_preview, continuation, input_bindings
from assistant.ui.renderer import EventRenderer, render_error
from assistant.ui.status import status_line

COMMANDS = create_command_registry()
# Plain dim text instead of prompt_toolkit's default reverse-video bar.
TOOLBAR_STYLE = Style.from_dict(
    {"bottom-toolbar": "noreverse #888888", "command-preview": "#888888 italic"}
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
    """Emacs editing, multiline prompts, history, and context-aware completion."""
    return PromptSession(
        history=PromptHistory(history_path or default_history_path()),
        completer=JamieCompleter(COMMANDS),
        complete_while_typing=False,
        complete_style=CompleteStyle.READLINE_LIKE,
        multiline=True,
        key_bindings=input_bindings(),
        prompt_continuation=continuation,
        **options,
    )


def _preview() -> str | list[tuple[str, str]]:
    description = command_preview(COMMANDS, get_app().current_buffer.text)
    return [("class:command-preview", description)] if description else ""


def _welcome(console: Console) -> None:
    banner = Text("Jamie ", style="bold cyan")
    banner.append(__version__, style="bold")
    banner.append("  ·  local assistant", style="dim")
    console.print(banner)
    console.print(Text("Enter sends · Esc+Enter adds a line · /help lists commands", style="dim"))
    console.print(Text("Ctrl+D or /quit to leave", style="dim"))


async def run_repl(
    application: JamieApplication,
    *,
    console: Console | None = None,
    session: PromptSession | None = None,
    model: str | None = None,
    started: float | None = None,
    agent_tools_factory: Callable[[], ToolExecutor] | None = None,
) -> None:
    """Own workspace registries until the interactive session exits."""
    workspace_tools = WorkspaceTools(agent_tools_factory) if agent_tools_factory else None
    try:
        await _run_repl(
            application,
            console=console,
            session=session,
            model=model,
            started=started,
            workspace_tools=workspace_tools,
        )
    finally:
        if workspace_tools is not None:
            workspace_tools.close()


async def _run_repl(
    application: JamieApplication,
    *,
    console: Console | None,
    session: PromptSession | None,
    model: str | None,
    started: float | None,
    workspace_tools: WorkspaceTools | None,
) -> None:
    """Send prompts to the already initialized application until the user exits.

    ``started`` is the time.monotonic() value when Jamie began starting up.
    """
    state = ReplState(
        console or Console(),
        application=application,
        model=model,
        workspace_tools=workspace_tools,
        mode="task" if workspace_tools is not None else "chat",
    )
    session = session or create_prompt_session()
    renderer = EventRenderer(state.console, answer_prefix="Jamie › ", markdown=True)
    _welcome(state.console)

    if started is not None:
        state.startup_seconds = time.monotonic() - started
    while state.running:
        try:
            prompt = await session.prompt_async(
                "You › ",
                bottom_toolbar=lambda: status_line(state),
                rprompt=_preview,
                style=TOOLBAR_STYLE,
            )
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not prompt.strip():
            continue
        if "\n" not in prompt:
            prompt = prompt.strip()
        handled = "\n" not in prompt and COMMANDS.dispatch(state, prompt)
        if inspect.isawaitable(handled):
            await handled  # a command that calls the model, such as /compact
        if handled:
            continue
        turn_started = time.monotonic()
        token = CancellationToken()
        activity: Status | None = (
            state.console.status("Jamie is thinking…", spinner="dots")
            if state.console.is_terminal
            else None
        )

        def show_and_count(event: AgentEvent) -> None:
            nonlocal activity
            if isinstance(
                event, AssistantDelta | AssistantFinished | ApprovalRequested | TurnFailed
            ):
                if activity is not None:
                    activity.stop()
                    activity = None
            elif activity is not None and isinstance(event, ToolStarted):
                activity.update("Jamie is using a tool…")
            elif activity is not None and isinstance(event, ToolFinished | ToolFailed):
                activity.update("Jamie is thinking…")
            if isinstance(event, ToolFinished | ToolFailed):
                state.tool_calls += 1
                state.tool_seconds += event.duration
            renderer.handle(event)

        try:
            if activity is not None:
                activity.start()
            if state.conversation_id is None:
                state.conversation_id = application.start_conversation()
            with interrupt_cancels(token):
                agent_options = (
                    {
                        "tools": workspace_tools.current(state.scope),
                        "max_steps": SCOPE_MAX_STEPS.get(state.scope, MAX_STEPS),
                    }
                    if workspace_tools is not None and state.mode == "task"
                    else {}
                )
                result = await application.run_turn(
                    prompt,
                    history=state.history,
                    temperature=state.temperature,
                    max_tokens=state.max_tokens,
                    thinking=state.thinking,
                    cancel=token,
                    conversation_id=state.conversation_id,
                    event_sink=show_and_count,
                    **agent_options,
                )
        except TurnCancelled:
            state.cancelled += 1
            state.say("Cancelled.")
            continue
        except (AgentError, JamieApplicationError, LLMError, MemoryStoreError, ValueError) as exc:
            state.failures += 1
            render_error(state.console, str(exc))
            continue
        finally:
            if activity is not None:
                activity.stop()
        state.turns += 1
        state.turn_seconds += time.monotonic() - turn_started
        state.output_tokens += result.output_tokens or 0
        if result.first_token_seconds is not None:
            state.first_token_seconds.append(result.first_token_seconds)
        state.model = result.model or state.model
        # ponytail: transcript grows unbounded; compaction is a separate task needing evals.
        if result.compacted_history is not None:
            state.history = list(result.compacted_history)
        state.history += [Message("user", prompt), Message("assistant", result.message)]
        refresh_context(state)  # a failure falls back to the estimate in the status bar
