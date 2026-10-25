"""Render completed answers and turn progress for terminal commands."""

import json
from collections.abc import Callable
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from assistant.agent.events import (
    AgentEvent,
    AssistantDelta,
    AssistantFinished,
    PlanCreated,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnStarted,
)

MAX_SUBJECT_CHARACTERS = 60
MAX_ERROR_LINES = 3
TOKEN_LIMIT_NOTICE = "The response reached the token limit."


def _plain(value: str) -> str:
    return value


def _quoted(value: str) -> str:
    return f'"{value}"'


def _host(value: str) -> str:
    parts = urlsplit(value if "//" in value else f"//{value}")
    return (parts.netloc or value).removeprefix("www.")


class _Label(NamedTuple):
    """How one tool names its start, its success, and the thing it acts on."""

    icon: str
    present: str
    past: str
    field: str | None = None
    subject: Callable[[str], str] = _plain


_LABELS: dict[str, _Label] = {
    "read_file": _Label("▸", "Reading", "Read", "path"),
    "list_directory": _Label("▸", "Listing", "Listed", "path"),
    "find_files": _Label("◦", "Finding files matching", "Found files matching", "pattern", _quoted),
    "search_files": _Label("◦", "Searching files for", "Searched files for", "query", _quoted),
    "write_file": _Label("▸", "Writing", "Wrote", "path"),
    "edit_file": _Label("▸", "Editing", "Edited", "path"),
    "move_file": _Label("▸", "Moving", "Moved", "source"),
    "create_directory": _Label("▸", "Creating", "Created", "path"),
    "trash_file": _Label("▸", "Trashing", "Trashed", "path"),
    "web_search": _Label("◦", "Searching web for", "Searched web for", "query", _quoted),
    "fetch_url": _Label("▸", "Reading", "Read", "url", _host),
    "extract_page": _Label("▸", "Reading", "Read", "url", _host),
    "research": _Label("◦", "Researching", "Researched", "question", _quoted),
    "open_application": _Label("▸", "Opening", "Opened", "name"),
    "reveal_in_finder": _Label("▸", "Revealing", "Revealed", "path"),
    "clipboard_read": _Label("▸", "Reading the clipboard", "Read the clipboard"),
    "clipboard_write": _Label("▸", "Writing the clipboard", "Wrote the clipboard"),
    "run_shortcut": _Label("▸", "Running shortcut", "Ran shortcut", "name"),
    "run_applescript": _Label("▸", "Running AppleScript", "Ran AppleScript"),
}

_COUNTED: dict[str, tuple[str, str]] = {
    "read_file": ("line", "lines"),
    "list_directory": ("entry", "entries"),
    "find_files": ("file", "files"),
    "search_files": ("match", "matches"),
}


# What the file tools return when they find nothing; one line, but no results.
_EMPTY_OUTPUTS = {"No files found.", "No matches found.", "Directory is empty."}


def _label(tool: str) -> _Label:
    return _LABELS.get(tool, _Label("▸", f"Running {tool}", f"Ran {tool}"))


def _phrase(label: _Label, verb: str, args: dict[str, Any] | None) -> str:
    """Name the action without exposing the protocol arguments."""
    value = (args or {}).get(label.field) if label.field else None
    if not isinstance(value, str | int | float) or not str(value).strip():
        return verb
    subject = label.subject(str(value).strip())
    if len(subject) > MAX_SUBJECT_CHARACTERS:
        subject = f"{subject[: MAX_SUBJECT_CHARACTERS - 1]}…"
    return f"{verb} {subject}"


def _counted(count: int, units: tuple[str, str]) -> str:
    return f"{count} {units[0] if count == 1 else units[1]}"


def _detail(tool: str, output: str) -> str | None:
    """Summarize a successful result without printing the result itself."""
    if tool == "web_search":
        try:
            results = json.loads(output)["results"]
        except (TypeError, KeyError, ValueError):
            return None
        return _counted(len(results), ("result", "results"))
    units = _COUNTED.get(tool)
    if units is None or not output:
        return None
    if output in _EMPTY_OUTPUTS:
        return _counted(0, units)
    return _counted(len(output.splitlines()), units)


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} min {int(seconds % 60)} s"


def _error_lines(message: str) -> list[str]:
    lines = [line.strip() for line in message.replace("\r", " ").splitlines() if line.strip()]
    if not lines:
        return ["Unknown error"]
    if len(lines) > MAX_ERROR_LINES:
        return [*lines[: MAX_ERROR_LINES - 1], "…"]
    return lines


def render_error(console: Console, message: str) -> None:
    """Show a bounded error and keep the next terminal prompt readable."""
    first, *rest = _error_lines(message)
    line = Text("  ✗ Error: ", style="red")
    line.append(first)
    console.print(line)
    for detail in rest:
        console.print(Text(f"    {detail}", style="red"))


class OneShotRenderer:
    """Print the final answer once; command handlers retain error exit handling."""

    def __init__(self, console: Console, errors: Console) -> None:
        self._console = console
        self._errors = errors
        self._streaming = False
        self._streamed = ""

    def handle(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self._streaming = False
            self._streamed = ""
        if isinstance(event, AssistantDelta):
            self._console.out(event.text, end="", highlight=False)
            self._streaming = True
            self._streamed += event.text
        if isinstance(event, TurnFailed) and self._streaming:
            self._console.out("")
            self._streaming = False
        if not isinstance(event, AssistantFinished):
            return
        if self._streaming:
            suffix = (
                event.text[len(self._streamed) :] if event.text.startswith(self._streamed) else ""
            )
            self._console.out(suffix, highlight=False)
            self._streaming = False
        else:
            self._console.print(event.text, markup=False, highlight=False)
        if event.finish_reason == "length":
            self._errors.print(
                "The response reached the token limit; use --max-tokens to allow a longer answer.",
                markup=False,
                highlight=False,
            )


class EventRenderer:
    """Show tool progress and answers; never show protocol JSON or tool output."""

    def __init__(
        self,
        console: Console,
        errors: Console | None = None,
        *,
        answer_prefix: str = "",
        markdown: bool = False,
    ) -> None:
        self._console = console
        self._errors = errors if errors is not None else console
        self._prefix = answer_prefix
        self._markdown = markdown
        self._started: tuple[str, dict[str, Any]] | None = None
        self._streaming = False

    def handle(self, event: AgentEvent) -> None:
        match event:
            case TurnStarted():
                self._started = None
                self._streaming = False
            case TurnFailed():
                if self._streaming:
                    self._console.out("")
                    self._streaming = False
            case ToolStarted(tool=tool, args=args):
                self._started = (tool, args)
                label = _label(tool)
                self._status(label.icon, _phrase(label, label.present, args), "cyan")
            case ToolFinished(tool=tool, result=result, duration=duration):
                label = _label(tool)
                line = _phrase(label, label.past, self._args(tool))
                detail = _detail(tool, result.output)
                parts = [line, *([detail] if detail else []), _elapsed(duration)]
                self._status("✓", " · ".join(parts), "green")
            case ToolFailed(tool=tool, error=error, duration=duration):
                label = _label(tool)
                failed = _phrase(label, label.present, self._args(tool))
                self._status(
                    "✗",
                    f"{failed} failed · {_elapsed(duration)}",
                    "red",
                )
                for detail in _error_lines(error):
                    self._console.print(Text(f"    {detail}", style="red"))
            case PlanCreated(plan=plan):
                self._console.print(Text(f"  Plan: {plan.goal}", style="bold dim"))
                for number, step in enumerate(plan.steps, 1):
                    after = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
                    self._console.print(
                        Text(f"    {number}. {step.description} · {step.tool}{after}", style="dim")
                    )
            case AssistantDelta(text=text):
                if not self._streaming:
                    self._console.out(self._prefix, end="", highlight=False)
                    self._streaming = True
                self._console.out(text, end="", highlight=False)
            case AssistantFinished(text=text, finish_reason=finish_reason):
                if self._streaming:
                    self._console.out("")
                    self._streaming = False
                elif self._markdown:
                    self._console.print(Text(self._prefix, style="cyan"), end="")
                    self._console.print(Markdown(text), soft_wrap=True)
                else:
                    self._console.print(f"{self._prefix}{text}", markup=False, highlight=False)
                if finish_reason == "length":
                    self._errors.print(TOKEN_LIMIT_NOTICE, markup=False, highlight=False)

    def _args(self, tool: str) -> dict[str, Any] | None:
        if self._started is None or self._started[0] != tool:
            return None
        return self._started[1]

    def _status(self, icon: str, message: str, style: str) -> None:
        line = Text("  ")
        line.append(f"{icon} ", style=style)
        line.append(message, style="dim")
        self._console.print(line)
