"""User notifications for unattended task events."""

from __future__ import annotations

import sys
from typing import Protocol

from assistant.tools.base import ToolError
from assistant.tools.macos import CommandRunner, SubprocessCommandRunner

MAX_TEXT = 200


class Notifier(Protocol):
    def notify(self, title: str, message: str) -> None: ...


def _applescript_string(value: str) -> str:
    cleaned = " ".join(value.split())[:MAX_TEXT]
    return '"' + cleaned.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MacNotifier:
    """Show a macOS notification with a fixed osascript command; failures are ignored."""

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def notify(self, title: str, message: str) -> None:
        script = (
            f"display notification {_applescript_string(message)} "
            f"with title {_applescript_string(title)}"
        )
        try:
            self._runner.run(["osascript", "-e", script], timeout=10)
        except ToolError:
            pass  # a missed banner must never fail a task


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.messages.append((title, message))


class NullNotifier:
    def notify(self, title: str, message: str) -> None:
        return


def default_notifier() -> Notifier:
    return MacNotifier() if sys.platform == "darwin" else NullNotifier()
