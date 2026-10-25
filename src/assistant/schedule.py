"""When scheduled prompts run: daily at a local time, or once at the next such time."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from datetime import time as clock

USAGE = "Usage: /schedule daily HH:MM <prompt> or /schedule at HH:MM <prompt>"
_REQUEST = re.compile(
    r"(?P<repeat>daily|at)\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+(?P<prompt>.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse(argument: str) -> tuple[str, str, str]:
    """Read ``daily 08:00 <prompt>`` or ``at 15:30 <prompt>`` as (repeat, HH:MM, prompt)."""
    match = _REQUEST.fullmatch(argument.strip())
    if match is None:
        raise ValueError(USAGE)
    hour, minute = int(match["hour"]), int(match["minute"])
    if hour > 23 or minute > 59:
        raise ValueError("Use a 24-hour time from 00:00 to 23:59.")
    repeat = "daily" if match["repeat"].lower() == "daily" else "once"
    return repeat, f"{hour:02d}:{minute:02d}", match["prompt"].strip()


def next_occurrence(time: str, after: datetime) -> datetime:
    """The first local wall-clock ``time`` later than ``after``, across DST changes."""
    hour, minute = (int(part) for part in time.split(":"))
    local = after.astimezone()
    day = local.date()
    while True:
        candidate = datetime.combine(day, clock(hour, minute)).astimezone()
        if candidate > local:
            return candidate
        day += timedelta(days=1)


def after_run(repeat: str, time: str, now: datetime) -> datetime | None:
    """A daily task moves to its next time, skipping any it missed; a one-off ends."""
    return next_occurrence(time, now) if repeat == "daily" else None
