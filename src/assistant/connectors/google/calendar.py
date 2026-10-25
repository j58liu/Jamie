"""Google Calendar connector with timezone-aware inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo

from pydantic import BaseModel

from assistant.connectors.base import Capability, ConnectionStatus, ConnectorInfo
from assistant.connectors.errors import ConnectorError
from assistant.connectors.google.auth import (
    CALENDAR_EVENTS,
    CALENDAR_EVENTS_READONLY,
    CALENDAR_FREEBUSY,
    GoogleSession,
)
from assistant.connectors.timeutil import (
    add_minutes,
    iso_utc,
    parse_provider_time,
    require_aware,
)

API = "https://www.googleapis.com/calendar/v3"
MAX_EVENTS = 25
MAX_DESCRIPTION_CHARACTERS = 1000
MAX_RANGE_DAYS = 62

OPERATIONS = (
    "calendar_list_events",
    "calendar_check_availability",
    "calendar_get_event",
    "calendar_create_event",
    "calendar_update_event",
)


class EventSummary(BaseModel):
    id: str
    title: str
    start: str
    end: str
    all_day: bool | None = None
    attendees: int | None = None


class Attendee(BaseModel):
    email: str
    response: str | None = None


class EventDetail(EventSummary):
    location: str | None = None
    description: str | None = None
    organizer: str | None = None
    attendee_list: list[Attendee] | None = None
    duplicate: bool | None = None


class TimeSpan(BaseModel):
    start: str
    end: str


class Availability(BaseModel):
    start: str
    end: str
    free: bool
    busy: list[TimeSpan]
    free_slots: list[TimeSpan]
    unavailable_attendees: list[str] | None = None


def _check_range(start: datetime, end: datetime) -> None:
    require_aware(start, "start")
    require_aware(end, "end")
    if end <= start:
        raise ConnectorError("end must be after start.")
    if end - start > timedelta(days=MAX_RANGE_DAYS):
        raise ConnectorError(f"The range must be at most {MAX_RANGE_DAYS} days.")


def merge_spans(spans: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping or touching busy blocks."""
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def free_slots(
    start: datetime, end: datetime, busy: list[tuple[datetime, datetime]], minimum_minutes: float
) -> list[tuple[datetime, datetime]]:
    slots = []
    cursor = start
    for busy_start, busy_end in merge_spans(busy):
        if busy_end <= start or busy_start >= end:
            continue
        if busy_start > cursor:
            slots.append((cursor, min(busy_start, end)))
        cursor = max(cursor, busy_end)
    if cursor < end:
        slots.append((cursor, end))
    minimum = timedelta(minutes=minimum_minutes)
    return [(a, b) for a, b in slots if b - a >= minimum]


class CalendarConnector:
    """Read, check, create, and update events on the primary Google calendar."""

    name = "google_calendar"

    def __init__(self, session: GoogleSession, tz: tzinfo, calendar_id: str = "primary") -> None:
        self._session = session
        self._tz = tz
        self._calendar = calendar_id

    def info(self) -> ConnectorInfo:
        status = self._session.status()
        capabilities: set[Capability] = set()
        if status is ConnectionStatus.CONNECTED:
            if self._session.has_scope(CALENDAR_EVENTS_READONLY):
                capabilities.add(Capability.READ)
            if self._session.has_scope(CALENDAR_FREEBUSY):
                capabilities.add(Capability.FREE_BUSY)
            if self._session.has_scope(CALENDAR_EVENTS):
                capabilities.add(Capability.WRITE)
            if not capabilities:
                status = ConnectionStatus.INSUFFICIENT_SCOPE
        return ConnectorInfo(self.name, status, frozenset(capabilities), OPERATIONS)

    # -- conversion ---------------------------------------------------------------

    def _local(self, value: datetime) -> str:
        return value.astimezone(self._tz).isoformat(timespec="minutes")

    def _provider_time(self, value: datetime) -> dict[str, str]:
        require_aware(value)
        zone = getattr(self._tz, "key", None)
        local = value.astimezone(self._tz)
        return {"dateTime": local.isoformat(), **({"timeZone": zone} if zone else {})}

    def _span(self, event: dict) -> tuple[datetime, datetime, bool]:
        start = parse_provider_time(event.get("start", {}), self._tz)
        end = parse_provider_time(event.get("end", event.get("start", {})), self._tz)
        return start.value, end.value, start.all_day

    def _summary_fields(self, event: dict) -> dict:
        start, end, all_day = self._span(event)
        if all_day:
            # Google's all-day end date is exclusive; show the last day it covers.
            last_day = (end - timedelta(days=1)).date() if end > start else start.date()
            start_text, end_text = start.date().isoformat(), last_day.isoformat()
        else:
            start_text, end_text = self._local(start), self._local(end)
        attendees = event.get("attendees") or []
        return {
            "id": event["id"],
            "title": event.get("summary", "(no title)"),
            "start": start_text,
            "end": end_text,
            "all_day": True if all_day else None,
            "attendees": len(attendees) or None,
        }

    def _detail(self, event: dict, duplicate: bool = False) -> EventDetail:
        attendees = [
            Attendee(email=a["email"], response=a.get("responseStatus"))
            for a in event.get("attendees") or []
            if a.get("email")
        ]
        description = (event.get("description") or "")[:MAX_DESCRIPTION_CHARACTERS] or None
        return EventDetail(
            **self._summary_fields(event),
            location=event.get("location"),
            description=description,
            organizer=(event.get("organizer") or {}).get("email"),
            attendee_list=attendees or None,
            duplicate=True if duplicate else None,
        )

    # -- operations ---------------------------------------------------------------

    def _raw_events(self, start: datetime, end: datetime, max_results: int) -> list[dict]:
        result = self._session.request(
            "GET",
            f"{API}/calendars/{self._calendar}/events",
            params={
                "timeMin": iso_utc(start),
                "timeMax": iso_utc(end),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": min(max_results, MAX_EVENTS),
            },
        )
        return [e for e in result.get("items", []) if e.get("status") != "cancelled"]

    def list_events(
        self, start: datetime, end: datetime, max_results: int = 10
    ) -> list[EventSummary]:
        self._session.require_scope(CALENDAR_EVENTS_READONLY, "read")
        _check_range(start, end)
        return [
            EventSummary(**self._summary_fields(e))
            for e in self._raw_events(start, end, max_results)
        ]

    def get_event(self, event_id: str) -> EventDetail:
        self._session.require_scope(CALENDAR_EVENTS_READONLY, "read")
        return self._detail(
            self._session.request("GET", f"{API}/calendars/{self._calendar}/events/{event_id}")
        )

    def check_availability(
        self,
        start: datetime,
        end: datetime,
        *,
        duration_minutes: int | None = None,
        attendees: list[str] | None = None,
    ) -> Availability:
        self._session.require_scope(CALENDAR_FREEBUSY, "read")
        _check_range(start, end)
        calendars = [self._calendar, *(attendees or [])]
        result = self._session.request(
            "POST",
            f"{API}/freeBusy",
            json={
                "timeMin": iso_utc(start),
                "timeMax": iso_utc(end),
                "items": [{"id": calendar} for calendar in calendars],
            },
        )
        busy: list[tuple[datetime, datetime]] = []
        unavailable: list[str] = []
        for calendar_id, data in (result.get("calendars") or {}).items():
            if data.get("errors"):
                if calendar_id != self._calendar:
                    unavailable.append(calendar_id)
                    continue
                raise ConnectorError("Your calendar's free/busy data is unavailable.")
            for block in data.get("busy", []):
                busy.append(
                    (
                        parse_provider_time({"dateTime": block["start"]}, self._tz).value,
                        parse_provider_time({"dateTime": block["end"]}, self._tz).value,
                    )
                )
        merged = merge_spans(busy)
        length = duration_minutes or (end - start).total_seconds() / 60
        slots = free_slots(start, end, merged, length)
        return Availability(
            start=self._local(start),
            end=self._local(end),
            free=bool(slots),
            busy=[TimeSpan(start=self._local(a), end=self._local(b)) for a, b in merged],
            free_slots=[TimeSpan(start=self._local(a), end=self._local(b)) for a, b in slots],
            unavailable_attendees=sorted(unavailable) or None,
        )

    def find_duplicate(self, title: str, start: datetime, end: datetime) -> dict | None:
        """An existing event with the same title that overlaps the requested time."""
        window_start = add_minutes(start, -1)
        window_end = add_minutes(end, 1)
        for event in self._raw_events(window_start, window_end, MAX_EVENTS):
            existing_start, existing_end, _ = self._span(event)
            same_title = event.get("summary", "").strip().casefold() == title.strip().casefold()
            if same_title and existing_start < end and existing_end > start:
                return event
        return None

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        *,
        attendees: list[str] | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> EventDetail:
        self._session.require_scope(CALENDAR_EVENTS, "calendar-write")
        _check_range(start, end)
        existing = self.find_duplicate(title, start, end)
        if existing is not None:
            return self._detail(existing, duplicate=True)
        body: dict[str, object] = {
            "summary": title,
            "start": self._provider_time(start),
            "end": self._provider_time(end),
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            body["attendees"] = [{"email": email} for email in attendees]
        created = self._session.request(
            "POST",
            f"{API}/calendars/{self._calendar}/events",
            params={"sendUpdates": "all" if attendees else "none"},
            json=body,
        )
        return self._detail(created)

    def update_event(
        self,
        event_id: str,
        *,
        title: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
    ) -> EventDetail:
        self._session.require_scope(CALENDAR_EVENTS, "calendar-write")
        if start is not None and end is not None:
            _check_range(start, end)
        body: dict[str, object] = {}
        if title is not None:
            body["summary"] = title
        if start is not None:
            body["start"] = self._provider_time(start)
        if end is not None:
            body["end"] = self._provider_time(end)
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if attendees is not None:
            body["attendees"] = [{"email": email} for email in attendees]
        if not body:
            raise ConnectorError("Nothing to update.")
        updated = self._session.request(
            "PATCH",
            f"{API}/calendars/{self._calendar}/events/{event_id}",
            params={"sendUpdates": "all" if attendees else "none"},
            json=body,
        )
        return self._detail(updated)
