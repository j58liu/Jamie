import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors import ConnectorError, Connectors, PermissionScopeError
from assistant.connectors.google.auth import (
    CALENDAR_EVENTS,
    CALENDAR_EVENTS_READONLY,
    CALENDAR_FREEBUSY,
)
from assistant.connectors.google.calendar import CalendarConnector, merge_spans
from assistant.security import CallbackApprovalProvider
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools

TZ = ZoneInfo("America/Toronto")
EVENTS = "/calendar/v3/calendars/primary/events"
FREEBUSY = "/calendar/v3/freeBusy"


def at(*args):
    return datetime(*args, tzinfo=TZ)


@pytest.fixture
def calendar(google_api):
    session = google_api.session([CALENDAR_EVENTS, CALENDAR_FREEBUSY])
    return google_api, CalendarConnector(session, TZ)


def test_event_across_dst_fall_back_keeps_correct_offsets(calendar):
    api, connector = calendar
    api.on(
        "GET",
        EVENTS,
        {
            "items": [
                {
                    "id": "e1",
                    "summary": "Late call",
                    "start": {"dateTime": "2026-11-01T05:30:00Z"},
                    "end": {"dateTime": "2026-11-01T07:30:00Z"},
                }
            ]
        },
    )
    [event] = connector.list_events(at(2026, 11, 1), at(2026, 11, 2))
    assert event.start == "2026-11-01T01:30-04:00"
    assert event.end == "2026-11-01T02:30-05:00"
    params = api.sent("GET", EVENTS)[0].url.params
    assert params["timeMin"] == "2026-11-01T04:00:00Z"
    assert params["timeMax"] == "2026-11-02T05:00:00Z"  # the day is 25 hours long
    assert params["singleEvents"] == "true"


def test_all_day_events_show_inclusive_last_day(calendar):
    api, connector = calendar
    api.on(
        "GET",
        EVENTS,
        {
            "items": [
                {
                    "id": "a",
                    "summary": "Holiday",
                    "start": {"date": "2026-10-30"},
                    "end": {"date": "2026-10-31"},
                },
                {
                    "id": "b",
                    "summary": "Trip",
                    "start": {"date": "2026-10-30"},
                    "end": {"date": "2026-11-02"},
                },
                {
                    "id": "c",
                    "summary": "Gone",
                    "status": "cancelled",
                    "start": {"date": "2026-10-30"},
                    "end": {"date": "2026-10-31"},
                },
            ]
        },
    )
    one_day, trip = connector.list_events(at(2026, 10, 30), at(2026, 11, 3))
    assert (one_day.start, one_day.end, one_day.all_day) == ("2026-10-30", "2026-10-30", True)
    assert (trip.start, trip.end) == ("2026-10-30", "2026-11-01")


def test_overlapping_and_back_to_back_meetings_merge_into_busy_blocks(calendar):
    api, connector = calendar
    api.on(
        "POST",
        FREEBUSY,
        {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": "2026-10-27T13:00:00Z", "end": "2026-10-27T14:00:00Z"},
                        {"start": "2026-10-27T13:30:00Z", "end": "2026-10-27T15:00:00Z"},
                        {"start": "2026-10-27T15:00:00Z", "end": "2026-10-27T15:30:00Z"},
                    ]
                }
            }
        },
    )
    result = connector.check_availability(
        at(2026, 10, 27, 9), at(2026, 10, 27, 17), duration_minutes=60
    )
    assert [(b.start, b.end) for b in result.busy] == [
        ("2026-10-27T09:00-04:00", "2026-10-27T11:30-04:00")
    ]
    assert [(s.start, s.end) for s in result.free_slots] == [
        ("2026-10-27T11:30-04:00", "2026-10-27T17:00-04:00")
    ]
    assert result.free is True
    body = json.loads(api.sent("POST", FREEBUSY)[0].content)
    assert body["timeMin"] == "2026-10-27T13:00:00Z"
    assert "summary" not in json.dumps(result.model_dump())


def test_fully_booked_window_is_not_free(calendar):
    api, connector = calendar
    api.on(
        "POST",
        FREEBUSY,
        {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": "2026-10-27T18:00:00Z", "end": "2026-10-27T18:45:00Z"},
                    ]
                }
            }
        },
    )
    result = connector.check_availability(
        at(2026, 10, 27, 14), at(2026, 10, 27, 15), duration_minutes=30
    )
    assert result.free is False and result.free_slots == []


def test_unavailable_attendees_are_reported_not_treated_as_free(calendar):
    api, connector = calendar
    api.on(
        "POST",
        FREEBUSY,
        {
            "calendars": {
                "primary": {"busy": []},
                "sam@example.com": {
                    "busy": [{"start": "2026-10-27T14:00:00Z", "end": "2026-10-27T15:00:00Z"}]
                },
                "bob@example.com": {"errors": [{"reason": "notFound"}]},
            }
        },
    )
    result = connector.check_availability(
        at(2026, 10, 27, 9),
        at(2026, 10, 27, 12),
        attendees=["sam@example.com", "bob@example.com"],
    )
    assert result.unavailable_attendees == ["bob@example.com"]
    assert [(b.start, b.end) for b in result.busy] == [
        ("2026-10-27T10:00-04:00", "2026-10-27T11:00-04:00")
    ]


def test_own_calendar_errors_fail_loudly(calendar):
    api, connector = calendar
    api.on("POST", FREEBUSY, {"calendars": {"primary": {"errors": [{"reason": "backendError"}]}}})
    with pytest.raises(ConnectorError):
        connector.check_availability(at(2026, 10, 27, 9), at(2026, 10, 27, 12))


def test_duplicate_scheduling_returns_the_existing_event(calendar):
    api, connector = calendar
    api.on(
        "GET",
        EVENTS,
        {
            "items": [
                {
                    "id": "existing",
                    "summary": "Sync with Alex",
                    "start": {"dateTime": "2026-10-27T14:00:00-04:00"},
                    "end": {"dateTime": "2026-10-27T14:30:00-04:00"},
                }
            ]
        },
    )
    event = connector.create_event(
        "sync with alex", at(2026, 10, 27, 14, 15), at(2026, 10, 27, 14, 45)
    )
    assert event.id == "existing" and event.duplicate is True
    assert not api.sent("POST", EVENTS)


def test_new_event_is_sent_with_offset_and_zone(calendar):
    api, connector = calendar
    api.on("GET", EVENTS, {"items": []})
    api.on("POST", EVENTS, lambda request: {**json.loads(request.content), "id": "new"})
    event = connector.create_event(
        "Sync with Alex",
        at(2026, 11, 1, 1, 30),
        at(2026, 11, 1, 2, 30),
        attendees=["alex@example.com"],
    )
    sent = json.loads(api.sent("POST", EVENTS)[0].content)
    assert sent["start"] == {"dateTime": "2026-11-01T01:30:00-04:00", "timeZone": "America/Toronto"}
    assert api.sent("POST", EVENTS)[0].url.params["sendUpdates"] == "all"
    assert event.id == "new" and event.duplicate is None


def test_writes_need_the_write_scope(google_api):
    connector = CalendarConnector(
        google_api.session([CALENDAR_EVENTS_READONLY, CALENDAR_FREEBUSY]), TZ
    )
    with pytest.raises(PermissionScopeError, match="calendar-write"):
        connector.create_event("x", at(2026, 10, 27, 9), at(2026, 10, 27, 10))


def test_merge_spans_handles_containment():
    a, b, c = at(2026, 1, 1, 9), at(2026, 1, 1, 10), at(2026, 1, 1, 12)
    assert merge_spans([(a, c), (a, b)]) == [(a, c)]


def _registry(connector, requests):
    registry = ToolRegistry(
        approvals=CallbackApprovalProvider(lambda r: requests.append(r) or True)
    )
    return register_personal_tools(
        registry, Connectors(tz=TZ, calendar=connector), now=lambda: at(2026, 10, 25, 10)
    )


def test_tools_reject_naive_datetimes_and_reversed_ranges(calendar):
    api, connector = calendar
    registry = _registry(connector, [])
    naive = registry.execute(
        "calendar_list_events", {"start": "2026-10-27T09:00:00", "end": "2026-10-27T17:00:00"}
    )
    assert not naive.success and "timezone" in naive.output.lower()
    reversed_range = registry.execute(
        "calendar_check_availability",
        {"start": "2026-10-27T17:00:00-04:00", "end": "2026-10-27T09:00:00-04:00"},
    )
    assert not reversed_range.success
    partial = registry.execute(
        "calendar_update_event", {"event_id": "e1", "start": "2026-10-27T09:00:00-04:00"}
    )
    assert not partial.success
    assert not api.requests


def test_creating_an_event_asks_for_approval(calendar):
    api, connector = calendar
    api.on("GET", EVENTS, {"items": []})
    api.on(
        "POST",
        EVENTS,
        {
            "id": "new",
            "summary": "Sync",
            "start": {"dateTime": "2026-10-27T14:00:00-04:00"},
            "end": {"dateTime": "2026-10-27T14:30:00-04:00"},
        },
    )
    requests = []
    result = _registry(connector, requests).execute(
        "calendar_create_event",
        {"title": "Sync", "start": "2026-10-27T14:00:00-04:00", "end": "2026-10-27T14:30:00-04:00"},
    )
    assert result.success, result.output
    assert [r.tool for r in requests] == ["calendar_create_event"]


def test_prompt_includes_current_local_time(calendar):
    _, connector = calendar
    assert (
        "Current local time: Sunday 2026-10-25T10:00-04:00"
        in _registry(connector, []).instructions()
    )


def test_free_slot_lengths_are_elapsed_time_across_fall_back(calendar):
    api, connector = calendar
    # 00:00-04:00 local on 2026-11-01 is five real hours (04:00Z-09:00Z); busy 05:00-06:00Z.
    api.on(
        "POST",
        FREEBUSY,
        {
            "calendars": {
                "primary": {
                    "busy": [{"start": "2026-11-01T05:00:00Z", "end": "2026-11-01T06:00:00Z"}]
                }
            }
        },
    )
    result = connector.check_availability(
        at(2026, 11, 1, 0), at(2026, 11, 1, 4), duration_minutes=150
    )
    # 04:00Z-05:00Z is one hour, 06:00Z-09:00Z is three hours; only the second fits 2.5h.
    assert [(s.start, s.end) for s in result.free_slots] == [
        ("2026-11-01T01:00-05:00", "2026-11-01T04:00-05:00")
    ]
