"""Lightweight scheduling: decides when tasks are due without loading the model.

Schedules are computed in the schedule's IANA timezone, so "daily at 08:00" stays
08:00 local time across daylight-saving changes. When the computer was asleep, the
missed-run policy decides whether to run late, skip, or ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from assistant.connectors.timeutil import WEEKDAYS, add_minutes, resolve_relative
from assistant.tasks.approvals import request_decision
from assistant.tasks.models import (
    ApprovalKind,
    MissedPolicy,
    Schedule,
    ScheduleKind,
    TaskState,
)
from assistant.tasks.store import TaskStore

MISSED_GRACE = timedelta(minutes=15)
EVENT_POLL = timedelta(minutes=5)
MAX_EVENT_TASKS = 5


class ScheduleError(ValueError):
    """A schedule specification could not be understood."""


_UNITS = {"m": 1, "min": 1, "h": 60, "d": 1440}


def parse_every(text: str) -> tuple[ScheduleKind, dict]:
    """Parse ``daily@08:00``, ``weekly:sun@09:30``, ``interval:2h``, or ``2h``."""
    value = text.strip().lower()
    if match := re.fullmatch(r"daily@(\d{1,2}):(\d{2})", value):
        return ScheduleKind.DAILY, {"time": f"{int(match[1]):02d}:{match[2]}"}
    if match := re.fullmatch(r"weekly:([a-z]{3,9})@(\d{1,2}):(\d{2})", value):
        day = next((d for d in WEEKDAYS if d.startswith(match[1])), None)
        if day is None:
            raise ScheduleError(f"Unknown weekday {match[1]!r}.")
        return ScheduleKind.WEEKLY, {"weekday": WEEKDAYS.index(day),
                                     "time": f"{int(match[2]):02d}:{match[3]}"}  # fmt: skip
    if match := re.fullmatch(r"(?:interval:)?(\d+)\s*(m|min|h|d)", value):
        minutes = int(match[1]) * _UNITS[match[2]]
        if minutes < 5:
            raise ScheduleError("The shortest interval is 5 minutes.")
        return ScheduleKind.INTERVAL, {"minutes": minutes}
    raise ScheduleError(
        f"Cannot understand {text!r}. Use daily@08:00, weekly:sun@09:00, or interval:2h."
    )


def _clock(spec: dict) -> time:
    hour, minute = (int(part) for part in spec["time"].split(":"))
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ScheduleError(f"Invalid time {spec['time']!r}.")
    return time(hour, minute)


def next_run(kind: ScheduleKind, spec: dict, timezone: str, after: datetime) -> datetime | None:
    """The first run strictly after ``after``; None when a one-time schedule is done."""
    tz = ZoneInfo(timezone)
    local = after.astimezone(tz)
    if kind is ScheduleKind.ONCE:
        at = datetime.fromisoformat(spec["at"])
        return at if at > after else None
    if kind is ScheduleKind.INTERVAL:
        return add_minutes(after, spec["minutes"])  # elapsed time, DST-safe
    if kind is ScheduleKind.EVENT:
        return add_minutes(after, spec.get("poll_minutes", 5))
    clock = _clock(spec)
    day = local.date()
    for offset in range(8):
        candidate_day = day + timedelta(days=offset)
        if kind is ScheduleKind.WEEKLY and candidate_day.weekday() != spec["weekday"]:
            continue
        # Wall-clock time in the zone; zoneinfo resolves the UTC offset for that date.
        candidate = datetime.combine(candidate_day, clock, tzinfo=tz)
        if candidate > after:
            return candidate
    raise ScheduleError("No next run could be computed.")  # pragma: no cover


def once_at(text: str, now: datetime) -> datetime:
    """Resolve "tomorrow 8am" or an ISO datetime to an aware time."""
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        resolved = resolve_relative(text, now)
        if not resolved.has_time:
            raise ScheduleError(f"{text!r} needs a time, for example 'tomorrow 8am'.") from None
        return resolved.start
    if value.tzinfo is None:
        value = value.replace(tzinfo=now.tzinfo)
    return value


@dataclass
class TickReport:
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)


def _instance_title(schedule: Schedule, due: datetime) -> str:
    local = due.astimezone(ZoneInfo(schedule.timezone))
    return f"{schedule.title} ({local.strftime('%a %b %d %H:%M')})"


def tick(store: TaskStore, now: datetime, connectors=None) -> TickReport:
    """Create task instances for every due schedule and advance it, without the model."""
    report = TickReport()
    for schedule in store.due_schedules(now):
        if schedule.kind is ScheduleKind.EVENT:
            _poll_event(store, schedule, now, connectors, report)
            continue
        due = schedule.next_run_at
        following = next_run(schedule.kind, schedule.spec, schedule.timezone, now)
        late = due is not None and now.astimezone(UTC) - due.astimezone(UTC) > MISSED_GRACE
        with store.transaction() as connection:
            if late and schedule.missed_policy is MissedPolicy.SKIP:
                report.skipped.append(schedule.id)
            elif late and schedule.missed_policy is MissedPolicy.ASK:
                task = store.create_task(
                    _instance_title(schedule, due), schedule.instruction, scope=schedule.scope,
                    workspace=schedule.workspace, schedule_id=schedule.id,
                    state=TaskState.WAITING_APPROVAL, connection=connection,
                )  # fmt: skip
                local = due.astimezone(ZoneInfo(schedule.timezone))
                request_decision(
                    store, task, kind=ApprovalKind.RUN_MISSED, tool="run_missed_task",
                    display={
                        "tool": "run_missed_task",
                        "arguments": {"task": schedule.title,
                                      "missed": local.isoformat(timespec="minutes")},
                        "note": "The computer was asleep or off when this was due.",
                    },
                    fingerprint_value=f"run_missed:{task.id}", connection=connection,
                )  # fmt: skip
                report.asked.append(task.id)
            else:  # on time, or run_late: missed runs collapse into one run now
                task = store.create_task(
                    _instance_title(schedule, due or now), schedule.instruction,
                    scope=schedule.scope, workspace=schedule.workspace,
                    schedule_id=schedule.id, connection=connection,
                )  # fmt: skip
                report.created.append(task.id)
            store.update_schedule(
                schedule.id, connection=connection, next_run_at=following,
                enabled=following is not None,
            )  # fmt: skip
    return report


def _poll_event(store: TaskStore, schedule: Schedule, now: datetime, connectors,
                report: TickReport) -> None:  # fmt: skip
    """Check a connector cheaply; create a task only for new matching items."""
    source = schedule.spec.get("source")
    following = next_run(schedule.kind, schedule.spec, schedule.timezone, now)
    new_ids: list[str] = []
    cursor = schedule.cursor
    if source == "gmail" and connectors is not None and connectors.gmail is not None:
        seen = set((schedule.cursor or "").split(","))
        found = connectors.gmail.search_messages(schedule.spec["query"], 10)
        new_ids = [message.id for message in found if message.id not in seen]
        cursor = ",".join([message.id for message in found][:50]) or schedule.cursor
    with store.transaction() as connection:
        if new_ids and schedule.cursor is not None:  # the first poll only sets the baseline
            ids = ", ".join(new_ids[:MAX_EVENT_TASKS])
            task = store.create_task(
                f"{schedule.title} ({len(new_ids)} new)",
                f"{schedule.instruction}\n\nNew Gmail message ids: {ids}",
                scope=schedule.scope, workspace=schedule.workspace, schedule_id=schedule.id,
                connection=connection,
            )  # fmt: skip
            report.created.append(task.id)
        store.update_schedule(schedule.id, connection=connection, next_run_at=following,
                              cursor=cursor or "")  # fmt: skip
