"""Terminal approval prompt; it only collects the user's answer for a policy decision."""

from __future__ import annotations

import json
import shlex
import unicodedata
from collections.abc import Callable

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel

# Levels where one "yes" may cover the rest of the session. Sensitive and destructive
# actions are always confirmed one at a time.
SESSION_APPROVABLE = frozenset({PermissionLevel.SAFE_WRITE, PermissionLevel.EXTERNAL})

RISKS = {
    PermissionLevel.SAFE_WRITE: "changes local files or apps",
    PermissionLevel.SENSITIVE: "sensitive; can read private data or run automation",
    PermissionLevel.DESTRUCTIVE: "destructive; may not be recoverable",
    PermissionLevel.EXTERNAL: "contacts the network",
}

# tool: (what Jamie wants to do, argument shown as the headline)
ACTIONS: dict[str, tuple[str, str | None]] = {
    "write_file": ("write", "path"),
    "edit_file": ("edit", "path"),
    "move_file": ("move", "source"),
    "create_directory": ("create the directory", "path"),
    "trash_file": ("move to the Trash", "path"),
    "open_application": ("open", "name"),
    "reveal_in_finder": ("reveal in Finder", "path"),
    "clipboard_write": ("replace the clipboard with", "text"),
    "clipboard_read": ("read the clipboard", None),
    "web_search": ("search the web for", "query"),
    "fetch_url": ("fetch", "url"),
}

# tool: (argument holding the command, the command line the tool runs with it).
# Keep in step with the argument arrays in assistant.tools.macos.
COMMANDS: dict[str, tuple[str, Callable[[str], str]]] = {
    "run_shortcut": ("name", lambda name: shlex.join(["shortcuts", "run", name])),
    "run_applescript": ("script", lambda script: f"osascript -e\n{script}"),
}

# Unicode bidi controls can make a displayed path read differently from the real one.
_BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F}


def visible(value: object) -> str:
    """Show control characters as escapes so model text cannot rewrite the prompt."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return "".join(
        character
        if character in "\n\t"
        or (unicodedata.category(character) != "Cc" and ord(character) not in _BIDI)
        else character.encode("unicode_escape").decode("ascii")
        for character in text
    )


def approval_panel(approval: ApprovalRequest) -> Panel:
    """Render every normalized argument; nothing the tool receives is hidden."""
    arguments = dict(approval.arguments)
    command = COMMANDS.get(approval.tool)
    if command is not None and isinstance(arguments.get(command[0]), str):
        field, command_line = command
        lines = [Text("Command"), Text(visible(command_line(arguments.pop(field))), style="bold")]
        footer = Text(f"Classification: {approval.permission.value.lower()}")
    else:
        verb, field = ACTIONS.get(approval.tool, (f"use {approval.tool}", None))
        lines = [Text(f"Jamie wants to {verb}" + (":" if field in arguments else "."))]
        if field in arguments:
            lines.append(Text(visible(arguments.pop(field)), style="bold"))
        risk = RISKS.get(approval.permission, approval.permission.value.lower())
        footer = Text(f"Risk: {approval.permission.value} — {risk}")
    for name, value in arguments.items():
        lines.append(Text(f"{name}: ", style="dim").append(visible(value), style=""))
    lines += [Text(""), Text(f"Tool: {approval.tool}", style="dim"), footer]
    return Panel(
        Group(*lines),
        title="Approval required",
        title_align="left",
        border_style="yellow",
        expand=False,
    )


class TerminalApprovalProvider:
    """Ask on the terminal; blank input, EOF, and Ctrl+C all deny."""

    def __init__(self, console: Console, ask: Callable[[str], str] | None = None) -> None:
        self._console = console
        self._ask = ask or console.input
        self._session_tools: set[str] = set()

    def request(self, approval: ApprovalRequest) -> bool:
        if approval.tool in self._session_tools:
            self._console.print(Text(f"  ✓ {approval.tool} allowed for this session", style="dim"))
            return True

        self._console.print(approval_panel(approval))
        session = approval.permission in SESSION_APPROVABLE
        choices = "[y] allow once  " + ("[a] allow session  " if session else "") + "[n] deny"
        self._console.print(choices, markup=False, highlight=False)
        while True:
            try:
                answer = self._ask("› ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._console.print("Denied.", markup=False, highlight=False)
                return False
            if answer in {"y", "yes"}:
                return True
            if answer in {"a", "always"} and session:
                self._session_tools.add(approval.tool)
                return True
            if answer in {"", "n", "no"}:
                return False
            self._console.print(f"Choose {choices}", markup=False, highlight=False)
