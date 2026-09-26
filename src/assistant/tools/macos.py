"""Approved macOS tools implemented with fixed executable argument arrays."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from assistant.cancellation import TurnCancelled, current_token
from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError
from assistant.tools.filesystem import _resolve, _workspace_root

MAX_NAME_CHARACTERS = 200
MAX_TEXT_CHARACTERS = 2_000_000
MAX_COMMAND_OUTPUT = 200_000
CANCEL_POLL_SECONDS = 0.1
TERMINATE_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        timeout: float,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run one allowlisted macOS executable without shell parsing."""

    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        timeout: float,
    ) -> CommandResult:
        if sys.platform != "darwin":
            raise ToolError("This tool is available only on macOS.")
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"macOS command is unavailable: {arguments[0]}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot run macOS command: {arguments[0]}") from exc

        token = current_token()
        deadline = time.monotonic() + timeout
        pending_input = input_text
        while True:
            try:
                stdout, stderr = process.communicate(
                    pending_input, timeout=min(CANCEL_POLL_SECONDS, timeout)
                )
                break
            except subprocess.TimeoutExpired:
                pending_input = None  # already written; later calls only collect output
                if token is not None and token.cancelled:
                    _stop(process)
                    raise TurnCancelled from None
                if time.monotonic() >= deadline:
                    _stop(process)
                    raise ToolError(f"macOS command timed out: {arguments[0]}") from None
        completed = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
        if completed.returncode != 0:
            detail = completed.stderr.strip()[:500]
            suffix = f": {detail}" if detail else ""
            raise ToolError(f"macOS command failed ({completed.returncode}){suffix}")
        stdout = completed.stdout
        if len(stdout) > MAX_COMMAND_OUTPUT:
            stdout = f"{stdout[:MAX_COMMAND_OUTPUT]}\n[truncated]"
        return CommandResult(stdout=stdout, stderr=completed.stderr)


def _stop(process: subprocess.Popen) -> None:
    """Ask the process to exit, then kill it if it ignores SIGTERM."""
    process.terminate()
    try:
        process.communicate(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validated_name(value: str) -> str:
    if not value.strip():
        raise ValueError("Name must not be blank.")
    if value.startswith("-"):
        raise ValueError("Name must not begin with a hyphen.")
    if "\x00" in value or any(character in value for character in "\r\n"):
        raise ValueError("Name must be one line without null characters.")
    return value


def _output(message: str, result: CommandResult) -> str:
    output = result.stdout.strip()
    return f"{message}\n{output}" if output else message


class OpenApplicationTool:
    name = "open_application"
    description = (
        "Launch a named macOS application hidden in the background. You cannot see or "
        "control its windows; use the file tools to read files."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        name: str = Field(max_length=MAX_NAME_CHARACTERS)

        @field_validator("name")
        @classmethod
        def validate_name(cls, value: str) -> str:
            return _validated_name(value)

    input_schema = Input

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        # -g keeps it behind the user's current app; -j launches it hidden. An app that is
        # already running stays as the user left it rather than being hidden.
        result = self._runner.run(["open", "-g", "-j", "-a", arguments.name], timeout=30)
        return _output(f"Opened application {arguments.name} in the background.", result)


class RevealInFinderTool:
    name = "reveal_in_finder"
    description = "Reveal an existing workspace path in Finder."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: Path

    input_schema = Input

    def __init__(self, root: Path, runner: CommandRunner | None = None) -> None:
        self._root = _workspace_root(root)
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        path = _resolve(self._root, arguments.path)
        result = self._runner.run(["open", "-R", str(path)], timeout=30)
        return _output(f"Revealed {arguments.path.as_posix()} in Finder.", result)


class ClipboardReadTool:
    name = "clipboard_read"
    description = "Read text from the macOS clipboard."
    risk = PermissionLevel.SENSITIVE

    class Input(_Input):
        pass

    input_schema = Input

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        result = self._runner.run(["pbpaste"], timeout=10)
        return result.stdout if result.stdout else "Clipboard is empty."


class ClipboardWriteTool:
    name = "clipboard_write"
    description = "Replace the macOS clipboard with text."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        text: str = Field(max_length=MAX_TEXT_CHARACTERS)

    input_schema = Input

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        self._runner.run(["pbcopy"], input_text=arguments.text, timeout=10)
        return f"Copied {len(arguments.text)} characters to the clipboard."


class RunShortcutTool:
    name = "run_shortcut"
    description = "Run a named shortcut with the macOS Shortcuts command."
    risk = PermissionLevel.SENSITIVE

    class Input(_Input):
        name: str = Field(max_length=MAX_NAME_CHARACTERS)

        @field_validator("name")
        @classmethod
        def validate_name(cls, value: str) -> str:
            return _validated_name(value)

    input_schema = Input

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        result = self._runner.run(["shortcuts", "run", arguments.name], timeout=300)
        return _output(f"Ran shortcut {arguments.name}.", result)


class RunAppleScriptTool:
    name = "run_applescript"
    description = "Run approved AppleScript source with osascript."
    risk = PermissionLevel.SENSITIVE

    class Input(_Input):
        script: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)

        @field_validator("script")
        @classmethod
        def validate_script(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("AppleScript must not be blank.")
            if "\x00" in value:
                raise ValueError("AppleScript must not contain null characters.")
            return value

    input_schema = Input

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    def execute(self, arguments: Input) -> str:
        result = self._runner.run(["osascript", "-e", arguments.script], timeout=60)
        return _output("AppleScript completed.", result)
