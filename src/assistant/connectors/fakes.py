"""In-memory Gmail and Calendar connectors for evaluations and demos.

They return the same models as the real connectors, so the agent sees identical data
shapes without contacting Google.
"""

from __future__ import annotations

import re
from datetime import datetime, tzinfo

from assistant.connectors.base import Capability, ConnectionStatus, ConnectorInfo
from assistant.connectors.errors import ConnectorError, NotFoundError
from assistant.connectors.google import calendar as calendar_module
from assistant.connectors.google import gmail as gmail_module
from assistant.connectors.google.calendar import (
    Availability,
    EventDetail,
    EventSummary,
    TimeSpan,
    free_slots,
    merge_spans,
)
from assistant.connectors.google.gmail import (
    Draft,
    MessageDetail,
    MessageSummary,
    SentMessage,
    ThreadSummary,
)
from assistant.connectors.timeutil import to_utc


class FakeGmail:
    name = "gmail"

    def __init__(self, messages: list[dict]) -> None:
        self.messages = [MessageDetail.model_validate(m) for m in messages]
        self.drafts: dict[str, Draft] = {}
        self.drafts_by_operation: dict[str, Draft] = {}
        self.sent: list[str] = []

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            self.name,
            ConnectionStatus.CONNECTED,
            frozenset({Capability.READ, Capability.DRAFT, Capability.SEND}),
            gmail_module.OPERATIONS,
        )

    def _matches(self, message: MessageDetail, query: str) -> bool:
        haystack = f"{message.sender} {message.subject} {message.snippet} {message.body}".lower()
        for term in query.lower().split():
            key, _, value = term.partition(":")
            if value and key in {"newer_than", "older_than", "in", "is", "label"}:
                continue
            if value and key == "from":
                if value not in message.sender.lower():
                    return False
            elif value and key == "subject":
                if value not in message.subject.lower():
                    return False
            elif re.sub(r"[\"']", "", term) not in haystack:
                return False
        return True

    def search_messages(self, query: str, max_results: int = 5) -> list[MessageSummary]:
        found = [m for m in self.messages if self._matches(m, query)]
        return [
            MessageSummary(**m.model_dump(include=set(MessageSummary.model_fields)))
            for m in found[:max_results]
        ]

    def get_message(self, message_id: str) -> MessageDetail:
        for message in self.messages:
            if message.id == message_id:
                return message
        raise NotFoundError("Not found in the Google account.")

    def get_thread(self, thread_id: str) -> ThreadSummary:
        thread = [m for m in self.messages if m.thread_id == thread_id]
        if not thread:
            raise NotFoundError("Not found in the Google account.")
        summaries = [
            MessageSummary(**m.model_dump(include=set(MessageSummary.model_fields))) for m in thread
        ]
        return ThreadSummary(
            id=thread_id, message_count=len(thread), messages=summaries, latest=thread[-1]
        )

    def find_draft_by_operation(self, operation_key: str) -> Draft | None:
        return self.drafts_by_operation.get(operation_key)

    def create_draft(
        self, to, subject, body, *, cc=None, reply_to_message_id=None, operation_key=None
    ) -> Draft:
        if operation_key in self.drafts_by_operation:
            return self.drafts_by_operation[operation_key]
        thread_id = None
        if reply_to_message_id:
            original = self.get_message(reply_to_message_id)
            thread_id = original.thread_id
            subject = subject or original.subject
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
        draft = Draft(
            id=f"draft-{len(self.drafts) + 1}",
            thread_id=thread_id,
            to=list(to),
            subject=subject,
            body=body,
            in_reply_to=reply_to_message_id,
        )
        self.drafts[draft.id] = draft
        if operation_key is not None:
            self.drafts_by_operation[operation_key] = draft
        return draft

    def get_draft(self, draft_id: str) -> Draft:
        if draft_id not in self.drafts:
            raise NotFoundError("Not found in the Google account.")
        return self.drafts[draft_id]

    def send_draft(self, draft_id: str) -> SentMessage:
        draft = self.get_draft(draft_id)
        self.sent.append(draft_id)
        return SentMessage(id=f"sent-{draft_id}", thread_id=draft.thread_id)


class FakeCalendar:
    name = "google_calendar"

    def __init__(self, events: list[dict], tz: tzinfo) -> None:
        self.tz = tz
        self.events = [
            {
                **e,
                "start": datetime.fromisoformat(e["start"]),
                "end": datetime.fromisoformat(e["end"]),
            }
            for e in events
        ]

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            self.name,
            ConnectionStatus.CONNECTED,
            frozenset({Capability.READ, Capability.FREE_BUSY, Capability.WRITE}),
            calendar_module.OPERATIONS,
        )

    def _local(self, value: datetime) -> str:
        return value.astimezone(self.tz).isoformat(timespec="minutes")

    def _summary(self, event: dict) -> dict:
        return {
            "id": event["id"],
            "title": event["title"],
            "start": self._local(event["start"]),
            "end": self._local(event["end"]),
        }

    def list_events(self, start, end, max_results=10) -> list[EventSummary]:
        found = [e for e in self.events if e["start"] < end and e["end"] > start]
        return [
            EventSummary(**self._summary(e))
            for e in sorted(found, key=lambda e: e["start"])[:max_results]
        ]

    def get_event(self, event_id: str) -> EventDetail:
        for event in self.events:
            if event["id"] == event_id:
                return EventDetail(**self._summary(event))
        raise NotFoundError("Not found in the Google account.")

    def check_availability(
        self, start, end, *, duration_minutes=None, attendees=None
    ) -> Availability:
        if end <= start:
            raise ConnectorError("end must be after start.")
        busy = merge_spans(
            [(e["start"], e["end"]) for e in self.events if e["start"] < end and e["end"] > start]
        )
        length = duration_minutes or (to_utc(end) - to_utc(start)).total_seconds() / 60
        slots = free_slots(start, end, busy, length)
        return Availability(
            start=self._local(start),
            end=self._local(end),
            free=bool(slots),
            busy=[TimeSpan(start=self._local(a), end=self._local(b)) for a, b in busy],
            free_slots=[TimeSpan(start=self._local(a), end=self._local(b)) for a, b in slots],
            unavailable_attendees=list(attendees) if attendees else None,
        )

    def find_by_operation(self, operation_key: str) -> EventDetail | None:
        for event in self.events:
            if event.get("operation_key") == operation_key:
                return EventDetail(**self._summary(event))
        return None

    def create_event(
        self,
        title,
        start,
        end,
        *,
        attendees=None,
        description=None,
        location=None,
        operation_key=None,
    ) -> EventDetail:
        existing = self.find_by_operation(operation_key) if operation_key else None
        if existing is not None:
            return existing.model_copy(update={"duplicate": True})
        for event in self.events:
            if (
                event["title"].casefold() == title.casefold()
                and event["start"] < end
                and event["end"] > start
            ):
                return EventDetail(**self._summary(event), duplicate=True)
        event = {
            "id": f"event-{len(self.events) + 1}",
            "title": title,
            "start": start,
            "end": end,
            "operation_key": operation_key,
        }
        self.events.append(event)
        return EventDetail(**self._summary(event))

    def update_event(self, event_id, **fields) -> EventDetail:
        for event in self.events:
            if event["id"] == event_id:
                for key in ("title", "start", "end"):
                    if fields.get(key) is not None:
                        event[key] = fields[key]
                return EventDetail(**self._summary(event))
        raise NotFoundError("Not found in the Google account.")
