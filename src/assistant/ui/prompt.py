"""Shell-style input behavior for Jamie's scrolling terminal."""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings

from assistant.ui.commands import CommandRegistry


class JamieCompleter(Completer):
    """Complete slash commands and filesystem paths where they are useful."""

    def __init__(self, commands: CommandRegistry) -> None:
        self._commands = commands.completer()
        self._directories = PathCompleter(only_directories=True, expanduser=True)
        self._paths = PathCompleter(expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        before = document.current_line_before_cursor
        if not before:
            return
        first_line = "\n" not in document.text_before_cursor
        if first_line and before.lower().startswith("/cwd "):
            argument = before[len("/cwd ") :]
            yield from self._directories.get_completions(Document(argument), complete_event)
            return
        command = before.split(" ", 1)[0]
        if first_line and command.startswith("/") and command.count("/") == 1:
            yield from self._commands.get_completions(document, complete_event)
            return
        if before.endswith((" ", "\t")):
            return
        candidate = before.rsplit(maxsplit=1)[-1]
        if candidate.startswith(("./", "../", "~/")) or (
            candidate.startswith("/") and candidate.count("/") > 1
        ):
            yield from self._paths.get_completions(Document(candidate), complete_event)


def command_preview(commands: CommandRegistry, text: str) -> str:
    """Describe one matching slash command without treating file paths as commands."""
    if "\n" in text or not text.startswith("/"):
        return ""
    name = text.split(" ", 1)[0].lower()
    if name.count("/") != 1:
        return ""
    matches = [
        command
        for command in commands
        if command.name.startswith(name) or any(alias.startswith(name) for alias in command.aliases)
    ]
    if len(matches) != 1:
        return ""
    command = matches[0]
    return f"{command.name} {command.usage} · {command.description}".replace("  ·", " ·")


def input_bindings() -> KeyBindings:
    """Enter sends; Escape then Enter inserts a line break for longer prompts."""
    bindings = KeyBindings()

    @bindings.add("enter")
    def send(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def continuation(width: int, line_number: int, is_soft_wrap: bool) -> str:
    """Keep multiline input aligned with the first prompt."""
    return " " * width if is_soft_wrap else "… ".rjust(width)
