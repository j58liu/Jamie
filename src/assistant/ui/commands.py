"""Local slash commands for the interactive terminal; only /compact calls the model."""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import WordCompleter
from rich.console import Console

from assistant.cancellation import CancellationToken, TurnCancelled, interrupt_cancels
from assistant.llm.base import ContextUsage, LLMError, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import ConversationSummary
from assistant.runtime.application import JamieApplicationError
from assistant.runtime.workspaces import WorkspaceTools
from assistant.security.audit import AuditError
from assistant.security.policy import PermissionLevel, PermissionPolicy, PolicyAction
from assistant.tools.base import ToolError
from assistant.tools.registry import create_agent_registry
from assistant.tools.web import WebClient
from assistant.ui.renderer import render_error

if TYPE_CHECKING:
    from assistant.runtime import JamieApplication

# ponytail: rough English average; use llama.cpp /tokenize if exact counts matter.
CHARACTERS_PER_TOKEN = 4
THINKING_MODES = {"on": True, "off": False}
CONTEXT_WARNING_PERCENT = 80


@dataclass
class ReplState:
    """Session-scoped interactive state; the application and model stay loaded."""

    console: Console
    application: JamieApplication | None = None
    conversation_id: str | None = None  # created with the first saved exchange
    listed: list[ConversationSummary] = field(default_factory=list)  # last /history
    history: list[Message] = field(default_factory=list)
    context: ContextUsage | None = None  # last exact measurement of ``history``
    thinking: bool | None = None
    temperature: float = 0.6
    max_tokens: int = 1024
    model: str | None = None
    # Session statistics; kept in memory only and never written or sent anywhere.
    turns: int = 0
    failures: int = 0
    cancelled: int = 0
    output_tokens: int = 0
    turn_seconds: float = 0.0  # completed turns only
    first_token_seconds: list[float] = field(default_factory=list)
    tool_calls: int = 0
    tool_seconds: float = 0.0
    startup_seconds: float | None = None
    started: float = field(default_factory=time.monotonic)
    running: bool = True
    workspace_tools: WorkspaceTools | None = None

    def estimated_tokens(self) -> int:
        return sum(len(m.content) for m in self.history) // CHARACTERS_PER_TOKEN

    def say(self, text: str) -> None:
        self.console.print(text, markup=False, highlight=False)


# A handler that needs the model returns an awaitable, which the REPL runs.
CommandHandler = Callable[[ReplState, str], Awaitable[None] | None]


def refresh_context(state: ReplState) -> str | None:
    """Re-measure after the history changes; returns the error text, if any."""
    state.context = None
    if state.application is None:
        return None
    try:
        state.context = state.application.context_usage(state.history)
    except LLMError as exc:
        return str(exc)
    return None


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: CommandHandler
    usage: str = ""
    aliases: tuple[str, ...] = ()
    completions: tuple[str, ...] = ()


class CommandRegistry:
    """One source for dispatch, /help, and Tab completion."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._lookup: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        for name in (command.name, *command.aliases):
            self._lookup[name] = command
        self._commands[command.name] = command

    def __iter__(self):
        return iter(self._commands.values())

    def completer(self) -> WordCompleter:
        words = []
        for command in self:
            words.append(command.name)
            words.extend(f"{command.name} {value}" for value in command.completions)
            words.extend(command.aliases)
        # sentence=True matches the whole line, so "/" is not treated as a word break.
        return WordCompleter(words, sentence=True)

    def dispatch(self, state: ReplState, line: str) -> bool | Awaitable[None]:
        """Run a slash command; return False when the line is a prompt for the model.

        A command that calls the model returns its awaitable, which is also truthy.
        """
        name, _, argument = line.partition(" ")
        # A second slash means a path such as /Users/me/notes.txt, not a command.
        if not name.startswith("/") or name.count("/") > 1:
            return False
        command = self._lookup.get(name.lower())
        if command is None:
            state.say(f"Unknown command {name}. Type /help for commands.")
            return True
        return command.handler(state, argument.strip()) or True


def _help(registry: CommandRegistry) -> CommandHandler:
    def handler(state: ReplState, argument: str) -> None:
        rows = [(f"{c.name} {c.usage}".strip(), c.description) for c in registry]
        width = max(len(syntax) for syntax, _ in rows)
        for syntax, description in rows:
            state.say(f"  {syntax:<{width}}  {description}")
        state.say("Start a prompt with a space to keep it out of input history.")

    return handler


def _quit(state: ReplState, argument: str) -> None:
    state.running = False


def _new(state: ReplState, argument: str) -> None:
    state.history.clear()
    state.conversation_id = None
    refresh_context(state)
    state.say("Started a new conversation.")


def _clear(state: ReplState, argument: str) -> None:
    state.console.clear()


def _workspace_command(command: str) -> CommandHandler:
    def handler(state: ReplState, argument: str) -> None:
        if argument:
            state.say(f"Usage: {command}")
            return
        if state.workspace_tools is None:
            state.say("Workspace tools are not available.")
            return
        try:
            tools = state.workspace_tools.current()
            handle = getattr(tools, "handle_command", None)
            result = handle(command) if handle is not None else None
            state.say(result or "Workspace changes are not available.")
        except (AuditError, ToolError, OSError, ValueError) as exc:
            state.say(f"Error: {exc}")

    return handler


def ago(timestamp: str, now: datetime | None = None) -> str:
    seconds = ((now or datetime.now(UTC)) - datetime.fromisoformat(timestamp)).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86_400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 2 * 86_400:
        return "yesterday"
    return f"{int(seconds // 86_400)}d ago"


def _title(text: str, limit: int = 50) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else f"{line[: limit - 1]}…"


def _history(state: ReplState, argument: str) -> None:
    if state.application is None:
        state.say("Conversation history is not available.")
        return
    state.listed = state.application.recent_conversations()
    if not state.listed:
        state.say("No saved conversations yet.")
        return
    state.say("Recent conversations")
    for number, conversation in enumerate(state.listed, start=1):
        marker = "*" if conversation.id == state.conversation_id else " "
        state.say(
            f" {marker}{number:>2}  {ago(conversation.updated_at):<10} "
            f"{_title(conversation.title)}  [{conversation.id[:8]}]"
        )
    state.say("Resume one with /resume <number> or /resume <id>.")


MIN_ID_PREFIX = 4


def _resolve(state: ReplState, argument: str) -> ConversationSummary | None:
    """Short numbers are list positions only; ids need a few characters to be unambiguous."""
    if argument.isdigit() and len(argument) < MIN_ID_PREFIX:
        listed = state.listed or state.application.recent_conversations()
        number = int(argument)
        return listed[number - 1] if 1 <= number <= len(listed) else None
    if len(argument) < MIN_ID_PREFIX:
        return None
    return state.application.find_conversation(argument.lower())


def _resume(state: ReplState, argument: str) -> None:
    if state.application is None:
        state.say("Conversation history is not available.")
        return
    if not argument:
        state.say("Usage: /resume <number|id>. Type /history to list conversations.")
        return
    conversation = _resolve(state, argument)
    if conversation is None:
        state.say(f"No conversation matches {argument}. Type /history to list conversations.")
        return
    state.history = state.application.conversation_messages(conversation.id)
    state.conversation_id = conversation.id
    refresh_context(state)
    exchanges = len(state.history) // 2
    state.say(f"Resumed: {_title(conversation.title)} ({exchanges} exchanges)")
    directory = Path(conversation.working_directory)
    if directory.is_dir() and os.access(directory, os.R_OK | os.X_OK):
        os.chdir(directory)
        state.say(f"Working directory: {directory}")
    else:
        state.say(f"Saved directory is unavailable; staying in {Path.cwd()}")


def _context(state: ReplState, argument: str) -> None:
    error = refresh_context(state)
    usage = state.context
    state.say("Context")
    state.say("")
    if usage is None:
        state.say(f"Used:      ~{state.estimated_tokens():,} tokens (estimated)")
        state.say("Maximum:    unknown" + (f" ({error})" if error else ""))
    else:
        width = len(f"{usage.maximum:,}")
        state.say(f"Used:       {usage.used:>{width},} tokens")
        state.say(f"Maximum:    {usage.maximum:>{width},} tokens")
        state.say(f"Available:  {usage.available:>{width},} tokens")
        state.say(f"Usage:      {f'{usage.percent}%':>{width}}")
    state.say("")
    state.say(f"Messages: {len(state.history)} ({len(state.history) // 2} exchanges)")
    state.say("Each request also adds the new prompt and any recalled memories.")
    state.say("Use /compact to summarize the conversation, or /new to start over.")
    if usage is not None and usage.percent >= CONTEXT_WARNING_PERCENT:
        state.say(f"Warning: {usage.percent}% of the context is used. Run /compact to free it.")


def _used_tokens(state: ReplState) -> str:
    if state.context is not None:
        return f"{state.context.used:,}"
    return f"~{state.estimated_tokens():,}"


async def _compact(state: ReplState, argument: str) -> None:
    if state.application is None:
        state.say("Compaction is not available.")
        return
    if not state.history:
        state.say("Nothing to compact yet.")
        return
    before = _used_tokens(state)
    token = CancellationToken()
    activity = (
        state.console.status("Compacting the conversation…", spinner="dots")
        if state.console.is_terminal
        else nullcontext()
    )
    try:
        with activity, interrupt_cancels(token):
            compacted = await state.application.compact(state.history, cancel=token)
    except TurnCancelled:
        state.say("Cancelled. The conversation was not compacted.")
        return
    except (JamieApplicationError, LLMError, ValueError) as exc:
        render_error(state.console, f"Compaction failed: {exc}")
        return
    messages = len(state.history)
    state.history = compacted
    refresh_context(state)
    state.say(
        f"Compacted {messages} messages into a summary ({before} → {_used_tokens(state)} tokens)."
    )


def _thinking_label(thinking: bool | None) -> str:
    return "server default" if thinking is None else ("on" if thinking else "off")


def _thinking(state: ReplState, argument: str) -> None:
    argument = argument.lower()
    if argument in THINKING_MODES:
        state.thinking = THINKING_MODES[argument]
    elif argument:
        state.say("Usage: /thinking on|off")
        return
    state.say(f"Thinking: {_thinking_label(state.thinking)}")


def _model(state: ReplState, argument: str) -> None:
    state.say(f"Model: {state.model or 'reported after the first reply'}")
    state.say("Server: llama.cpp")
    state.say(f"Thinking: {_thinking_label(state.thinking)}")
    state.say(f"Temperature: {state.temperature}  Max tokens: {state.max_tokens}")


def _tool_definitions() -> list[dict[str, object]]:
    # The backend is only named here, never called; creating it sends no request.
    with WebClient() as web:
        return create_agent_registry(Path.cwd(), web=web, backend=LlamaCppBackend()).definitions()


def _tools(state: ReplState, argument: str) -> None:
    state.say("Tools Jamie can use:")
    definitions = _tool_definitions()
    width = max(len(str(d["name"])) for d in definitions)
    for definition in definitions:
        summary = str(definition["description"]).split(". ")[0].rstrip(".")
        state.say(f"  {definition['name']:<{width}}  {definition['risk']:<11}  {summary}")


def _permissions(state: ReplState, argument: str) -> None:
    policy = PermissionPolicy()
    registered = {str(d["name"]) for d in _tool_definitions()}
    for level in PermissionLevel:
        tools = sorted(name for name in registered if policy.permission_for(name) is level)
        for action, rule in (
            (PolicyAction.ALLOW, "runs without asking"),
            (PolicyAction.CONFIRM, "asks for approval"),
        ):
            matching = [name for name in tools if policy.decide(name).action is action]
            if matching:
                state.say(f"  {level.value} ({rule}): {', '.join(matching)}")


def _cwd(state: ReplState, argument: str) -> None:
    if argument:
        target = Path(argument).expanduser()
        target = (Path.cwd() / target).resolve()
        if not target.is_dir():
            state.say(f"Not a directory: {target}")
            return
        if not os.access(target, os.R_OK | os.X_OK):
            state.say(f"Cannot open directory: {target}")
            return
        os.chdir(target)
    state.say(str(Path.cwd()))


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60)} min"


def _stats(state: ReplState, argument: str) -> None:
    rows = [("Turns", f"{state.turns:,}")]
    if state.failures:
        rows.append(("Failed turns", f"{state.failures:,}"))
    if state.cancelled:
        rows.append(("Cancelled turns", f"{state.cancelled:,}"))
    model_calls = getattr(state.application, "model_calls", None)
    if model_calls is not None:
        rows.append(("Model calls", f"{model_calls:,}"))
    rows += [
        ("Tool calls", f"{state.tool_calls:,}"),
        ("Generated tokens", f"{state.output_tokens:,}"),
    ]
    if state.first_token_seconds:
        average = sum(state.first_token_seconds) / len(state.first_token_seconds)
        rows.append(("Average first token", duration(average)))
    if state.turns:
        rows.append(("Average turn", duration(state.turn_seconds / state.turns)))
    if state.tool_calls:
        rows.append(("Average tool", duration(state.tool_seconds / state.tool_calls)))
    if state.turn_seconds and state.output_tokens:
        rows.append(("Tokens per second", f"{state.output_tokens / state.turn_seconds:.1f}"))
    if state.context is not None:
        rows.append(("Context", f"{state.context.used:,} / {state.context.maximum:,}"))
    if state.startup_seconds is not None:
        rows.append(("Startup", duration(state.startup_seconds)))
    rows.append(("Session age", duration(time.monotonic() - state.started)))
    label_width = max(len(label) for label, _ in rows) + 2
    value_width = max(len(value) for _, value in rows)
    state.say("Session statistics")
    state.say("")
    for label, value in rows:
        state.say(f"{label:<{label_width}}{value:>{value_width}}")


def create_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for command in (
        SlashCommand("/help", "Show commands.", _help(registry)),
        SlashCommand("/new", "Start a new conversation; the model stays loaded.", _new),
        SlashCommand("/clear", "Clear the screen; the conversation is kept.", _clear),
        SlashCommand(
            "/diff", "Show the latest changes in this workspace.", _workspace_command("/diff")
        ),
        SlashCommand(
            "/undo", "Undo the latest changed turn in this workspace.", _workspace_command("/undo")
        ),
        SlashCommand("/history", "List recent saved conversations.", _history),
        SlashCommand(
            "/resume", "Continue a saved conversation; the model stays loaded.", _resume, "<n|id>"
        ),
        SlashCommand("/context", "Show the size of this conversation.", _context),
        SlashCommand("/compact", "Summarize the conversation to free context.", _compact),
        SlashCommand("/model", "Show the model and generation settings.", _model),
        SlashCommand(
            "/thinking",
            "Turn model reasoning on or off.",
            _thinking,
            "[on|off]",
            completions=tuple(THINKING_MODES),
        ),
        SlashCommand("/tools", "List agent tools and their permission levels.", _tools),
        SlashCommand("/permissions", "Show which tools ask for approval.", _permissions),
        SlashCommand("/cwd", "Show or change the working directory.", _cwd, "[path]"),
        SlashCommand("/stats", "Show turn, token, and timing totals.", _stats),
        SlashCommand("/quit", "Leave Jamie (also Ctrl+D).", _quit, aliases=("/exit",)),
    ):
        registry.register(command)
    return registry
