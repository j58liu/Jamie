"""Apple Reminders through a native EventKit helper with a fixed JSON interface.

The helper is invoked as ``reminders-helper <command>`` with one JSON object on stdin
and answers with ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": {...}}``.
No AppleScript is generated.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, tzinfo
from pathlib import Path

from pydantic import BaseModel, ValidationError

from assistant.connectors.base import Capability, ConnectionStatus, ConnectorInfo
from assistant.connectors.errors import (
    ConnectorError,
    NotFoundError,
    ProviderError,
    UnavailableError,
)
from assistant.connectors.timeutil import require_aware
from assistant.tools.base import ToolError
from assistant.tools.macos import CommandRunner, SubprocessCommandRunner

TIMEOUT_SECONDS = 30
MAX_NOTES_CHARACTERS = 500
COMMANDS = ("status", "list", "create", "complete")
OPERATIONS = ("reminders_list", "reminders_create", "reminders_complete")
BUILD_HINT = "Build it with scripts/build-reminders-helper.sh."


def default_helper_path() -> Path:
    return Path("~/Library/Application Support/Jamie/bin/reminders-helper").expanduser()


class Reminder(BaseModel):
    id: str
    title: str
    list: str | None = None
    due: str | None = None
    completed: bool | None = None
    notes: str | None = None


class RemindersConnector:
    name = "apple_reminders"

    def __init__(
        self,
        helper_path: Path | None = None,
        *,
        runner: CommandRunner | None = None,
        tz: tzinfo | None = None,
        platform: str | None = None,
    ) -> None:
        self._helper = Path(helper_path) if helper_path is not None else default_helper_path()
        self._runner = runner if runner is not None else SubprocessCommandRunner()
        self._tz = tz
        self._platform = platform if platform is not None else sys.platform

    def _unavailable_reason(self) -> str | None:
        if self._platform != "darwin":
            return "Apple Reminders is available only on macOS."
        if not self._helper.is_file():
            return f"Reminders helper not found at {self._helper}. {BUILD_HINT}"
        return None

    def info(self) -> ConnectorInfo:
        reason = self._unavailable_reason()
        if reason is not None:
            return ConnectorInfo(
                self.name, ConnectionStatus.UNAVAILABLE, operations=OPERATIONS, detail=reason
            )
        try:
            access = self._call("status", {}).get("access", "unknown")
        except ConnectorError as exc:
            return ConnectorInfo(
                self.name, ConnectionStatus.UNAVAILABLE, operations=OPERATIONS, detail=str(exc)
            )
        if access in {"denied", "restricted"}:
            return ConnectorInfo(
                self.name,
                ConnectionStatus.NOT_CONNECTED,
                operations=OPERATIONS,
                detail="Reminders access is denied in System Settings.",
            )
        return ConnectorInfo(
            self.name,
            ConnectionStatus.CONNECTED,
            frozenset({Capability.READ, Capability.WRITE}),
            OPERATIONS,
            detail="" if access == "granted" else "access will be requested on first use",
        )

    def _call(self, command: str, request: dict) -> dict:
        if command not in COMMANDS:
            raise ConnectorError(f"Unsupported Reminders command {command!r}.")
        reason = self._unavailable_reason()
        if reason is not None:
            raise UnavailableError(reason)
        try:
            result = self._runner.run(
                [str(self._helper), command],
                input_text=json.dumps(request),
                timeout=TIMEOUT_SECONDS,
            )
        except ToolError as exc:
            raise ProviderError(f"Reminders helper failed: {exc}") from exc
        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise ProviderError("Reminders helper returned invalid JSON.") from exc
        if not isinstance(response, dict):
            raise ProviderError("Reminders helper returned an unexpected response.")
        if response.get("ok") is True:
            payload = response.get("result", {})
            return payload if isinstance(payload, dict) else {"items": payload}
        error = response.get("error") or {}
        code, message = error.get("code", "error"), error.get("message", "Reminders failed.")
        if code == "permission_denied":
            raise UnavailableError(message)
        if code == "not_found":
            raise NotFoundError(message)
        raise ProviderError(message)

    @staticmethod
    def _reminders(items: object) -> list[Reminder]:
        try:
            reminders = [Reminder.model_validate(item) for item in items or []]
        except ValidationError as exc:
            raise ProviderError("Reminders helper returned malformed reminders.") from exc
        for reminder in reminders:
            if reminder.notes and len(reminder.notes) > MAX_NOTES_CHARACTERS:
                reminder.notes = reminder.notes[:MAX_NOTES_CHARACTERS]
            if reminder.completed is False:
                reminder.completed = None
        return reminders

    def list(
        self, list_name: str | None = None, include_completed: bool = False, limit: int = 25
    ) -> list[Reminder]:
        result = self._call(
            "list",
            {"list": list_name, "include_completed": include_completed, "limit": limit},
        )
        return self._reminders(result.get("items"))

    def create(
        self,
        title: str,
        *,
        due: datetime | date | None = None,
        list_name: str | None = None,
        notes: str | None = None,
    ) -> Reminder:
        request: dict[str, object] = {"title": title, "list": list_name, "notes": notes}
        if isinstance(due, datetime):
            require_aware(due, "due")
            request["due"] = due.isoformat()
            request["all_day"] = False
        elif isinstance(due, date):
            request["due"] = due.isoformat()
            request["all_day"] = True
        result = self._call("create", request)
        return self._reminders([result])[0]

    def find(self, title: str) -> Reminder | None:
        """An existing reminder with exactly this title, completed or not."""
        for reminder in self.list(include_completed=True, limit=100):
            if reminder.title.strip().casefold() == title.strip().casefold():
                return reminder
        return None

    def complete(self, reminder_id: str) -> Reminder:
        result = self._call("complete", {"id": reminder_id})
        reminder = self._reminders([result])[0]
        reminder.completed = True
        return reminder
