"""Timezone-aware time handling at the connector boundary.

Every datetime that enters or leaves a connector is timezone-aware. Provider values are
normalized here, and relative phrases ("next Friday") are resolved before any API call.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class TimeResolutionError(ValueError):
    """A phrase or value could not be turned into an unambiguous aware time."""


def local_zone(name: str | None = None) -> tzinfo:
    """Return the configured IANA zone, else the system zone, else UTC."""
    candidates = [name, os.environ.get("TZ")]
    try:
        link = Path("/etc/localtime").resolve()
        if "zoneinfo" in link.parts:
            candidates.append("/".join(link.parts[link.parts.index("zoneinfo") + 1 :]))
    except OSError:
        pass
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate.lstrip(":"))
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return UTC


def require_aware(value: datetime, field: str = "datetime") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TimeResolutionError(f"{field} must include a timezone offset.")
    return value


def to_utc(value: datetime) -> datetime:
    return require_aware(value).astimezone(UTC)


def iso_utc(value: datetime) -> str:
    return to_utc(value).isoformat().replace("+00:00", "Z")


def add_minutes(value: datetime, minutes: float) -> datetime:
    """Add elapsed time (not wall-clock time), so DST transitions are handled correctly."""
    tz = require_aware(value).tzinfo
    return (to_utc(value) + timedelta(minutes=minutes)).astimezone(tz)


def start_of_day(day: date, tz: tzinfo) -> datetime:
    return datetime.combine(day, time(0), tzinfo=tz)


@dataclass(frozen=True)
class ProviderTime:
    value: datetime
    all_day: bool


def parse_provider_time(raw: dict, tz: tzinfo) -> ProviderTime:
    """Parse a Google-style ``{"dateTime": ...}`` or ``{"date": ...}`` object."""
    if raw.get("dateTime"):
        parsed = datetime.fromisoformat(raw["dateTime"].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            zone = ZoneInfo(raw["timeZone"]) if raw.get("timeZone") else tz
            parsed = parsed.replace(tzinfo=zone)
        return ProviderTime(parsed.astimezone(tz), all_day=False)
    if raw.get("date"):
        return ProviderTime(start_of_day(date.fromisoformat(raw["date"]), tz), all_day=True)
    raise TimeResolutionError("Provider time has neither dateTime nor date.")


@dataclass(frozen=True)
class ResolvedRange:
    start: datetime
    end: datetime
    has_time: bool


_TIME = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b")


def _parse_clock(text: str) -> time | None:
    for match in _TIME.finditer(text):
        hour, minute, meridiem = int(match.group(1)), int(match.group(2) or 0), match.group(3)
        if meridiem is None and match.group(2) is None:
            continue  # a bare number such as "in 3 days" is not a time
        if meridiem:
            if not 1 <= hour <= 12:
                continue
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        if hour < 24 and minute < 60:
            return time(hour, minute)
    return None


def resolve_relative(text: str, now: datetime) -> ResolvedRange:
    """Resolve a small, deterministic set of relative phrases against ``now``.

    Rules: "today", "tomorrow", "yesterday", "in N days", "<weekday>" or
    "this <weekday>" (the next occurrence, today included), "next <weekday>" (that day
    in the following Monday-based week), "this week", "next week", plus an optional
    clock time like "3pm" or "15:30". Day-only phrases cover whole local days.
    """
    require_aware(now, "now")
    tz = now.tzinfo
    phrase = text.strip().lower()
    today = now.date()
    monday = today - timedelta(days=today.weekday())
    span_days = 1
    day: date | None = None

    if "next week" in phrase:
        day, span_days = monday + timedelta(days=7), 7
    elif "this week" in phrase:
        day, span_days = today, 7 - today.weekday()
    elif "day after tomorrow" in phrase:
        day = today + timedelta(days=2)
    elif "tomorrow" in phrase:
        day = today + timedelta(days=1)
    elif "yesterday" in phrase:
        day = today - timedelta(days=1)
    elif "today" in phrase or "tonight" in phrase:
        day = today
    elif match := re.search(r"\bin\s+(\d{1,3})\s+days?\b", phrase):
        day = today + timedelta(days=int(match.group(1)))
    else:
        for index, name in enumerate(WEEKDAYS):
            if re.search(rf"\bnext\s+{name}\b", phrase):
                day = monday + timedelta(days=7 + index)
                break
            if re.search(rf"\b{name}\b", phrase):
                day = today + timedelta(days=(index - today.weekday()) % 7)
                break
    if day is None:
        raise TimeResolutionError(f"Cannot resolve the time phrase {text!r}.")

    clock = _parse_clock(phrase) if span_days == 1 else None
    if clock is not None:
        start = datetime.combine(day, clock, tzinfo=tz)
        return ResolvedRange(start, start, has_time=True)
    start = start_of_day(day, tz)
    return ResolvedRange(start, start_of_day(day + timedelta(days=span_days), tz), has_time=False)


def describe_now(now: datetime) -> str:
    """One line for the system prompt so the model can resolve dates itself."""
    require_aware(now, "now")
    zone = getattr(now.tzinfo, "key", None) or now.tzname() or "UTC"
    return (
        f"Current local time: {now.strftime('%A')} {now.isoformat(timespec='minutes')} "
        f"({zone}). Always pass datetimes with a timezone offset."
    )
