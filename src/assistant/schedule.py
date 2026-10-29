"""When scheduled prompts run: daily, weekly, every so often, or once."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from datetime import time as clock

USAGE = (
    "Usage: /schedule daily HH:MM <prompt>, weekly <day> HH:MM <prompt>, "
    "every <N>m|h|d <prompt>, at HH:MM <prompt>, or on YYYY-MM-DD HH:MM <prompt>"
)
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MISSED = ("run", "skip", "ask")
# A task found this long after its time was missed (the Mac slept or Jamie was closed).
LATE = timedelta(minutes=10)
SHORTEST_INTERVAL = timedelta(minutes=5)  # each run starts the model; keep it from looping

_TIME = r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
_PROMPT = r"\s+(?P<prompt>.+)"
_DATE = r"(?P<date>\d{4}-\d{2}-\d{2})"
_FORMS = {
    form: re.compile(pattern + _PROMPT, re.IGNORECASE | re.DOTALL)
    for form, pattern in {
        "daily": rf"daily\s+{_TIME}",
        "at": rf"at\s+{_TIME}",
        "on": rf"on\s+{_DATE}\s+{_TIME}",
        "weekly": rf"weekly\s+(?P<day>[a-z]+)\s+{_TIME}",
        "every": r"every\s+(?P<count>\d+)(?P<unit>[mhd])",
    }.items()
}
_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def _hhmm(match: re.Match) -> str:
    hour, minute = int(match["hour"]), int(match["minute"])
    if hour > 23 or minute > 59:
        raise ValueError("Use a 24-hour time from 00:00 to 23:59.")
    return f"{hour:02d}:{minute:02d}"


def parse(argument: str) -> tuple[str, str, str]:
    """Read a schedule request as (repeat, time, prompt).

    ``time`` is what the repeat needs: ``08:00`` (daily, once), ``mon 09:00``
    (weekly), ``2h`` (every), or ``2026-11-03 15:30`` (once on a date).
    """
    text = argument.strip()
    form = text.split(maxsplit=1)[0].lower() if text else ""
    match = _FORMS[form].fullmatch(text) if form in _FORMS else None
    if match is None:
        raise ValueError(USAGE)
    prompt = match["prompt"].strip()
    if form == "every":
        count = int(match["count"])
        if timedelta(**{_UNITS[match["unit"].lower()]: count}) < SHORTEST_INTERVAL:
            raise ValueError("Repeat at most every 5 minutes.")
        return "every", f"{count}{match['unit'].lower()}", prompt
    time = _hhmm(match)
    if form == "weekly":
        word = match["day"].lower()
        if word not in DAYS and word not in _DAY_NAMES:
            raise ValueError("Name a weekday, such as mon or monday.")
        return "weekly", f"{word[:3]} {time}", prompt
    if form == "on":
        try:
            datetime.fromisoformat(f"{match['date']} {time}")
        except ValueError:
            raise ValueError("Use a real date, such as 2026-11-03.") from None
        return "once", f"{match['date']} {time}", prompt
    return ("daily" if form == "daily" else "once"), time, prompt


def next_run(repeat: str, time: str, after: datetime) -> datetime:
    """The first run later than ``after``, in local wall-clock time across DST changes."""
    if repeat == "every":
        count, unit = int(time[:-1]), time[-1]
        return after + timedelta(**{_UNITS[unit]: count})
    if repeat == "once" and "-" in time:  # a date was given; it may already have passed
        return datetime.fromisoformat(time).astimezone()
    weekday = None
    if repeat == "weekly":
        day, time = time.split()
        weekday = DAYS.index(day)
    hour, minute = (int(part) for part in time.split(":"))
    local = after.astimezone()
    day = local.date()
    while True:
        candidate = datetime.combine(day, clock(hour, minute)).astimezone()
        if candidate > local and weekday in (None, day.weekday()):
            return candidate
        day += timedelta(days=1)


def after_run(repeat: str, time: str, now: datetime) -> datetime | None:
    """A repeating task moves to its next time, skipping any it missed; a one-off ends."""
    return None if repeat == "once" else next_run(repeat, time, now)


def describe(repeat: str, time: str) -> str:
    if repeat == "daily":
        return f"every day at {time}"
    if repeat == "weekly":
        day, time = time.split()
        return f"every {day.capitalize()} at {time}"
    if repeat == "every":
        return f"every {time}"
    return f"once on {time}" if "-" in time else f"once at {time}"
