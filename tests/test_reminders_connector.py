import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import (
    ConnectionStatus,
    NotFoundError,
    ProviderError,
    UnavailableError,
)
from assistant.connectors.apple.reminders import RemindersConnector
from assistant.connectors.timeutil import TimeResolutionError
from assistant.tools import CommandResult, ToolError

TZ = ZoneInfo("America/Toronto")


class HelperRunner:
    def __init__(self, *responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append((arguments, json.loads(input_text), timeout))
        if self.error:
            raise ToolError(self.error)
        response = self.responses.pop(0)
        return CommandResult(
            stdout=response if isinstance(response, str) else json.dumps(response), stderr=""
        )


@pytest.fixture
def helper(tmp_path):
    path = tmp_path / "reminders-helper"
    path.write_text("#!/bin/sh\n")
    return path


def connector(helper, runner):
    return RemindersConnector(helper, runner=runner, tz=TZ, platform="darwin")


def test_list_uses_fixed_command_and_returns_compact_items(helper):
    runner = HelperRunner(
        {
            "ok": True,
            "result": {
                "items": [
                    {
                        "id": "R1",
                        "title": "Call Alex",
                        "list": "Inbox",
                        "due": "2026-10-26",
                        "completed": False,
                        "notes": "n" * 900,
                    },
                ]
            },
        }
    )
    [item] = connector(helper, runner).list("Inbox")
    arguments, request, _ = runner.calls[0]
    assert arguments == [str(helper), "list"]
    assert request == {"list": "Inbox", "include_completed": False, "limit": 25}
    assert item.completed is None and len(item.notes) == 500


def test_create_sends_timed_or_all_day_due(helper):
    runner = HelperRunner(
        {"ok": True, "result": {"id": "R2", "title": "Call Alex"}},
        {"ok": True, "result": {"id": "R3", "title": "Pay rent"}},
    )
    reminders = connector(helper, runner)
    reminders.create("Call Alex", due=datetime(2026, 10, 26, 10, tzinfo=TZ))
    reminders.create("Pay rent", due=date(2026, 11, 1))
    assert runner.calls[0][1]["due"] == "2026-10-26T10:00:00-04:00"
    assert runner.calls[0][1]["all_day"] is False
    assert runner.calls[1][1] == {
        "title": "Pay rent",
        "list": None,
        "notes": None,
        "due": "2026-11-01",
        "all_day": True,
    }


def test_naive_due_time_is_rejected(helper):
    with pytest.raises(TimeResolutionError):
        connector(helper, HelperRunner()).create("x", due=datetime(2026, 10, 26, 10))


def test_complete_marks_completed(helper):
    runner = HelperRunner({"ok": True, "result": {"id": "R1", "title": "Call Alex"}})
    assert connector(helper, runner).complete("R1").completed is True
    assert runner.calls[0][0][-1] == "complete"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            {"ok": False, "error": {"code": "permission_denied", "message": "denied"}},
            UnavailableError,
        ),
        ({"ok": False, "error": {"code": "not_found", "message": "gone"}}, NotFoundError),
        ("not json", ProviderError),
        ({"ok": True, "result": {"items": [{"title": "missing id"}]}}, ProviderError),
    ],
)
def test_helper_errors_are_typed(helper, response, error):
    with pytest.raises(error):
        connector(helper, HelperRunner(response)).list()


def test_helper_timeout_is_a_provider_error(helper):
    runner = HelperRunner(error="macOS command timed out: reminders-helper")
    with pytest.raises(ProviderError, match="timed out"):
        connector(helper, runner).list()


def test_status_without_helper_or_off_macos(tmp_path, helper):
    missing = RemindersConnector(tmp_path / "nope", runner=HelperRunner(), platform="darwin")
    info = missing.info()
    assert info.status is ConnectionStatus.UNAVAILABLE and "build-reminders-helper" in info.detail
    linux = RemindersConnector(helper, runner=HelperRunner(), platform="linux")
    assert linux.info().status is ConnectionStatus.UNAVAILABLE
    with pytest.raises(UnavailableError):
        linux.list()


def test_status_reports_permission(helper):
    denied = connector(helper, HelperRunner({"ok": True, "result": {"access": "denied"}}))
    assert denied.info().status is ConnectionStatus.NOT_CONNECTED
    granted = connector(helper, HelperRunner({"ok": True, "result": {"access": "granted"}}))
    assert granted.info().status is ConnectionStatus.CONNECTED
