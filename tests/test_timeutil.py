from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.connectors.timeutil import (
    TimeResolutionError,
    add_minutes,
    describe_now,
    iso_utc,
    parse_provider_time,
    resolve_relative,
)

TORONTO = ZoneInfo("America/Toronto")
SUNDAY = datetime(2026, 10, 25, 10, 0, tzinfo=TORONTO)
WEDNESDAY = datetime(2026, 10, 21, 10, 0, tzinfo=TORONTO)


@pytest.mark.parametrize(
    ("now", "phrase", "expected"),
    [
        (WEDNESDAY, "friday", date(2026, 10, 23)),
        (WEDNESDAY, "next Friday", date(2026, 10, 30)),
        (WEDNESDAY, "this wednesday", date(2026, 10, 21)),
        (SUNDAY, "friday", date(2026, 10, 30)),
        (SUNDAY, "next friday", date(2026, 10, 30)),
        (SUNDAY, "next monday", date(2026, 10, 26)),
        (SUNDAY, "tomorrow", date(2026, 10, 26)),
        (SUNDAY, "in 3 days", date(2026, 10, 28)),
    ],
)
def test_relative_days(now, phrase, expected):
    resolved = resolve_relative(phrase, now)
    assert resolved.start.date() == expected
    assert not resolved.has_time


def test_clock_time_is_attached_to_the_day():
    resolved = resolve_relative("tomorrow at 3pm", SUNDAY)
    assert resolved.start == datetime(2026, 10, 26, 15, 0, tzinfo=TORONTO)
    assert resolved.has_time
    assert resolve_relative("next tuesday 14:30", SUNDAY).start.hour == 14


def test_next_week_spans_monday_to_monday_across_dst():
    resolved = resolve_relative("next week", SUNDAY)
    assert resolved.start.isoformat() == "2026-10-26T00:00:00-04:00"
    assert resolved.end.isoformat() == "2026-11-02T00:00:00-05:00"


def test_unknown_phrase_and_naive_now_are_rejected():
    with pytest.raises(TimeResolutionError):
        resolve_relative("sometime soon", SUNDAY)
    with pytest.raises(TimeResolutionError):
        resolve_relative("tomorrow", datetime(2026, 10, 25))


def test_elapsed_minutes_cross_dst_transitions():
    fall = datetime(2026, 11, 1, 0, 30, tzinfo=TORONTO)
    assert add_minutes(fall, 120).isoformat() == "2026-11-01T01:30:00-05:00"
    spring = datetime(2027, 3, 14, 1, 30, tzinfo=TORONTO)
    assert add_minutes(spring, 60).isoformat() == "2027-03-14T03:30:00-04:00"


def test_provider_times_are_normalized():
    timed = parse_provider_time({"dateTime": "2026-11-01T05:30:00Z"}, TORONTO)
    assert timed.value.isoformat() == "2026-11-01T01:30:00-04:00" and not timed.all_day
    local = parse_provider_time({"dateTime": "2026-11-01T09:00:00", "timeZone": "UTC"}, TORONTO)
    assert iso_utc(local.value) == "2026-11-01T09:00:00Z"
    all_day = parse_provider_time({"date": "2026-11-01"}, TORONTO)
    assert all_day.all_day and all_day.value.isoformat() == "2026-11-01T00:00:00-04:00"


def test_prompt_line_names_weekday_and_zone():
    line = describe_now(SUNDAY)
    assert "Sunday 2026-10-25T10:00-04:00" in line and "America/Toronto" in line
