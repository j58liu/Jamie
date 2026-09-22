"""Local slash commands for the interactive terminal; they never reach the model."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.completion import WordCompleter
from rich.console import Console

from assistant.llm.base import Message
from assistant.security.policy import PermissionLevel, PermissionPolicy, PolicyAction
from assistant.tools.registry import create_agent_registry
from assistant.tools.web import WebClient

# ponytail: rough English average; use llama.cpp /tokenize if exact counts matter.
CHARACTERS_PER_TOKEN = 4
THINKING_MODES = {"on": True, "off": False}


@dataclass
class ReplState:
    """Session-scoped interactive state; the application and model stay loaded."""

    console: Console
    history: list[Message] = field(default_factory=list)
    thinking: bool | None = None
    temperature: float = 0.6
    max_tokens: int = 1024
    model: str | None = None
    turns: int = 0
    failures: int = 0
    output_tokens: int = 0
    model_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic)
    running: bool = True

    def estimated_tokens(self) -> int:
        return sum(len(m.content) for m in self.history) // CHARACTERS_PER_TOKEN

    def say(self, text: str) -> None:
        self.console.print(text, markup=False, highlight=False)


CommandHandler = Callable[[ReplState, str], None]


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
            if name in self._lookup:
                raise ValueError(f"Command {name} is already registered.")
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

    def dispatch(self, state: ReplState, line: str) -> bool:
        """Run a slash command; return False when the line is a prompt for the model."""
        name, _, argument = line.partition(" ")
        # A second slash means a path such as /Users/me/notes.txt, not a command.
        if not name.startswith("/") or name.count("/") > 1:
            return False
        command = self._lookup.get(name.lower())
        if command is None:
            state.say(f"Unknown command {name}. Type /help for commands.")
        else:
            command.handler(state, argument.strip())
        return True


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
    state.say("Started a new conversation.")


def _clear(state: ReplState, argument: str) -> None:
    state.console.clear()


def _history(state: ReplState, argument: str) -> None:
    prompts = [m.content for m in state.history if m.role == "user"]
    if not prompts:
        state.say("No prompts in this conversation yet.")
    for number, prompt in enumerate(prompts, start=1):
        state.say(f"  {number}. {prompt}")


def _context(state: ReplState, argument: str) -> None:
    characters = sum(len(m.content) for m in state.history)
    state.say(f"Messages: {len(state.history)} ({len(state.history) // 2} exchanges)")
    state.say(f"Estimated tokens: ~{state.estimated_tokens()} ({characters} characters)")
    state.say("Relevant memories are added to each request and are not counted here.")


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
    with WebClient() as web:
        return create_agent_registry(Path.cwd(), web=web).definitions()


def _tools(state: ReplState, argument: str) -> None:
    state.say("Tools available to jamie agent (interactive chat does not call tools yet):")
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
        if not tools:
            continue
        action = policy.decide(tools[0]).action
        rule = "runs without asking" if action is PolicyAction.ALLOW else "asks for approval"
        state.say(f"  {level.value} ({rule}): {', '.join(tools)}")


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


def _stats(state: ReplState, argument: str) -> None:
    state.say(f"Turns: {state.turns} completed, {state.failures} failed")
    state.say(f"Output tokens: {state.output_tokens}")
    state.say(f"Model time: {state.model_seconds:.1f}s")
    if state.model_seconds and state.output_tokens:
        state.say(f"Speed: {state.output_tokens / state.model_seconds:.1f} tokens/s")
    state.say(f"Session time: {time.monotonic() - state.started:.0f}s")


def create_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for command in (
        SlashCommand("/help", "Show commands.", _help(registry)),
        SlashCommand("/new", "Start a new conversation; the model stays loaded.", _new),
        SlashCommand("/clear", "Clear the screen; the conversation is kept.", _clear),
        SlashCommand("/history", "List prompts in this conversation.", _history),
        SlashCommand("/context", "Show the size of this conversation.", _context),
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
