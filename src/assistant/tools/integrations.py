"""Connector tools: one explicit, typed operation per tool.

Tools only adapt arguments and results. Connectors make the provider calls, the
registry validates arguments, and the security policy decides whether a call may run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime

from pydantic import AwareDatetime, Field, field_validator, model_validator

from assistant.connectors import Connectors
from assistant.connectors.base import ConnectorInput, ConnectorTool, to_json
from assistant.connectors.timeutil import describe_now
from assistant.security.policy import PermissionLevel
from assistant.tasks.context import current_operation_key

_ID = re.compile(r"^[A-Za-z0-9_:.\-]{1,256}$")
_EMAIL = re.compile(r"^[^@\s,<>]+@[^@\s,<>]+\.[^@\s,<>]+$")
MAX_RECIPIENTS = 20
MAX_BODY_CHARACTERS = 20_000


def _identifier(value: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError("Use the id exactly as returned by an earlier tool result.")
    return value


def _emails(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    cleaned = [value.strip() for value in values]
    for value in cleaned:
        address = value[value.find("<") + 1 : value.rfind(">")] if "<" in value else value
        if not _EMAIL.fullmatch(address):
            raise ValueError(f"{value!r} is not an email address.")
    return cleaned


def _one_line(value: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError("Must be a single line.")
    return value.strip()


class _NowContext:
    """Tools that take datetimes add the current local time to the prompt."""

    def __init__(self, now: Callable[[], datetime]) -> None:
        self._now = now

    def prompt_context(self) -> str:
        return describe_now(self._now())


# -- connector status ---------------------------------------------------------------


class ConnectorsStatusTool:
    name = "connectors_status"
    description = "Show which personal apps (Gmail, Calendar, Reminders) are connected."
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        pass

    input_schema = Input

    def __init__(self, connectors: Connectors) -> None:
        self._connectors = connectors

    def execute(self, arguments: Input) -> str:
        return to_json([connector.info().as_dict() for connector in self._connectors.all()])


# -- Gmail ------------------------------------------------------------------------------


class GmailSearchMessagesTool(ConnectorTool):
    name = "gmail_search_messages"
    description = (
        "Search Gmail using Gmail search syntax, e.g. 'from:alex meeting newer_than:14d'. "
        "Returns id, thread_id, sender, subject, snippet, received_at. "
        "Read one result with gmail_get_message."
    )
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        query: str = Field(min_length=1, max_length=500)
        max_results: int = Field(default=5, ge=1, le=10)

        _line = field_validator("query")(_one_line)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.search_messages(arguments.query, arguments.max_results)


class GmailGetMessageTool(ConnectorTool):
    name = "gmail_get_message"
    description = (
        "Read one Gmail message by id: sender, recipients, subject, and the plain-text body "
        "without quoted history."
    )
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        message_id: str

        _id = field_validator("message_id")(_identifier)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.get_message(arguments.message_id)


class GmailGetThreadTool(ConnectorTool):
    name = "gmail_get_thread"
    description = (
        "Read a Gmail thread by thread_id: a summary of every message plus the full body of "
        "the latest one. Use gmail_get_message when one message is enough."
    )
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        thread_id: str

        _id = field_validator("thread_id")(_identifier)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.get_thread(arguments.thread_id)


class GmailCreateDraftTool(ConnectorTool):
    name = "gmail_create_draft"
    description = (
        "Save an email draft in Gmail; nothing is sent. To reply, pass reply_to_message_id so "
        "the draft joins the thread. Requires user approval."
    )
    risk = PermissionLevel.EXTERNAL

    class Input(ConnectorInput):
        to: list[str] = Field(min_length=1, max_length=MAX_RECIPIENTS)
        subject: str = Field(default="", max_length=300)
        body: str = Field(min_length=1, max_length=MAX_BODY_CHARACTERS)
        cc: list[str] | None = Field(default=None, max_length=MAX_RECIPIENTS)
        reply_to_message_id: str | None = None

        _addresses = field_validator("to", "cc")(_emails)
        _subject = field_validator("subject")(_one_line)

        @field_validator("reply_to_message_id")
        @classmethod
        def _reply_id(cls, value: str | None) -> str | None:
            return None if value is None else _identifier(value)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.create_draft(
            arguments.to,
            arguments.subject,
            arguments.body,
            cc=arguments.cc,
            reply_to_message_id=arguments.reply_to_message_id,
            operation_key=current_operation_key.get(),
        )


class GmailGetDraftTool(ConnectorTool):
    name = "gmail_get_draft"
    description = "Read a saved Gmail draft by draft id."
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        draft_id: str

        _id = field_validator("draft_id")(_identifier)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.get_draft(arguments.draft_id)


class GmailSendDraftTool(ConnectorTool):
    name = "gmail_send_draft"
    description = (
        "Send a saved Gmail draft by draft id. Only use this when the user explicitly asks to "
        "send. The user must approve every send."
    )
    risk = PermissionLevel.SEND

    class Input(ConnectorInput):
        draft_id: str

        _id = field_validator("draft_id")(_identifier)

    input_schema = Input

    def __init__(self, gmail) -> None:
        self._gmail = gmail

    def run(self, arguments: Input):
        return self._gmail.send_draft(arguments.draft_id)


# -- Calendar ---------------------------------------------------------------------------


class _Range(ConnectorInput):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _ordered(self):
        if self.end <= self.start:
            raise ValueError("end must be after start.")
        return self


class CalendarListEventsTool(_NowContext, ConnectorTool):
    name = "calendar_list_events"
    description = (
        "List calendar events between start and end (ISO 8601 with timezone offset). "
        "Returns id, title, start, end."
    )
    risk = PermissionLevel.READ

    class Input(_Range):
        max_results: int = Field(default=10, ge=1, le=25)

    input_schema = Input

    def __init__(self, calendar, now: Callable[[], datetime]) -> None:
        super().__init__(now)
        self._calendar = calendar

    def run(self, arguments: Input):
        return self._calendar.list_events(arguments.start, arguments.end, arguments.max_results)


class CalendarCheckAvailabilityTool(_NowContext, ConnectorTool):
    name = "calendar_check_availability"
    description = (
        "Check whether the user (and optional attendees) are free between start and end. "
        "Returns busy blocks and free slots at least duration_minutes long, without event "
        "details."
    )
    risk = PermissionLevel.READ

    class Input(_Range):
        duration_minutes: int | None = Field(default=None, ge=5, le=1440)
        attendees: list[str] | None = Field(default=None, max_length=10)

        _addresses = field_validator("attendees")(_emails)

    input_schema = Input

    def __init__(self, calendar, now: Callable[[], datetime]) -> None:
        super().__init__(now)
        self._calendar = calendar

    def run(self, arguments: Input):
        return self._calendar.check_availability(
            arguments.start,
            arguments.end,
            duration_minutes=arguments.duration_minutes,
            attendees=arguments.attendees,
        )


class CalendarGetEventTool(ConnectorTool):
    name = "calendar_get_event"
    description = "Read one calendar event by id, including location, description, attendees."
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        event_id: str

        _id = field_validator("event_id")(_identifier)

    input_schema = Input

    def __init__(self, calendar) -> None:
        self._calendar = calendar

    def run(self, arguments: Input):
        return self._calendar.get_event(arguments.event_id)


class CalendarCreateEventTool(_NowContext, ConnectorTool):
    name = "calendar_create_event"
    description = (
        "Create a calendar event. If an event with the same title already overlaps that "
        "time, the existing event is returned with duplicate=true. Attendees are invited. "
        "Requires user approval."
    )
    risk = PermissionLevel.EXTERNAL

    class Input(_Range):
        title: str = Field(min_length=1, max_length=300)
        attendees: list[str] | None = Field(default=None, max_length=MAX_RECIPIENTS)
        description: str | None = Field(default=None, max_length=5000)
        location: str | None = Field(default=None, max_length=300)

        _addresses = field_validator("attendees")(_emails)
        _title = field_validator("title")(_one_line)

    input_schema = Input

    def __init__(self, calendar, now: Callable[[], datetime]) -> None:
        super().__init__(now)
        self._calendar = calendar

    def run(self, arguments: Input):
        return self._calendar.create_event(
            arguments.title,
            arguments.start,
            arguments.end,
            attendees=arguments.attendees,
            description=arguments.description,
            location=arguments.location,
            operation_key=current_operation_key.get(),
        )


class CalendarUpdateEventTool(_NowContext, ConnectorTool):
    name = "calendar_update_event"
    description = (
        "Change fields of an existing calendar event by id. Pass start and end together. "
        "Requires user approval."
    )
    risk = PermissionLevel.EXTERNAL

    class Input(ConnectorInput):
        event_id: str
        title: str | None = Field(default=None, min_length=1, max_length=300)
        start: AwareDatetime | None = None
        end: AwareDatetime | None = None
        description: str | None = Field(default=None, max_length=5000)
        location: str | None = Field(default=None, max_length=300)
        attendees: list[str] | None = Field(default=None, max_length=MAX_RECIPIENTS)

        _id = field_validator("event_id")(_identifier)
        _addresses = field_validator("attendees")(_emails)

        @model_validator(mode="after")
        def _times(self):
            if (self.start is None) != (self.end is None):
                raise ValueError("Pass start and end together.")
            if self.start is not None and self.end is not None and self.end <= self.start:
                raise ValueError("end must be after start.")
            return self

    input_schema = Input

    def __init__(self, calendar, now: Callable[[], datetime]) -> None:
        super().__init__(now)
        self._calendar = calendar

    def run(self, arguments: Input):
        return self._calendar.update_event(
            arguments.event_id,
            title=arguments.title,
            start=arguments.start,
            end=arguments.end,
            description=arguments.description,
            location=arguments.location,
            attendees=arguments.attendees,
        )


# -- Reminders --------------------------------------------------------------------------


class RemindersListTool(ConnectorTool):
    name = "reminders_list"
    description = "List Apple Reminders, optionally from one list. Returns id, title, due."
    risk = PermissionLevel.READ

    class Input(ConnectorInput):
        list_name: str | None = Field(default=None, max_length=200)
        include_completed: bool = False
        limit: int = Field(default=25, ge=1, le=100)

    input_schema = Input

    def __init__(self, reminders) -> None:
        self._reminders = reminders

    def run(self, arguments: Input):
        return self._reminders.list(
            arguments.list_name, arguments.include_completed, arguments.limit
        )


class RemindersCreateTool(_NowContext, ConnectorTool):
    name = "reminders_create"
    description = (
        "Create an Apple Reminder. due is a date (YYYY-MM-DD) or a datetime with timezone offset."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(ConnectorInput):
        title: str = Field(min_length=1, max_length=300)
        due: AwareDatetime | date | None = None
        list_name: str | None = Field(default=None, max_length=200)
        notes: str | None = Field(default=None, max_length=2000)

        _title = field_validator("title")(_one_line)

    input_schema = Input

    def __init__(self, reminders, now: Callable[[], datetime]) -> None:
        super().__init__(now)
        self._reminders = reminders

    def run(self, arguments: Input):
        return self._reminders.create(
            arguments.title,
            due=arguments.due,
            list_name=arguments.list_name,
            notes=arguments.notes,
        )


class RemindersCompleteTool(ConnectorTool):
    name = "reminders_complete"
    description = "Mark an Apple Reminder as completed by id (from reminders_list)."
    risk = PermissionLevel.SAFE_WRITE

    class Input(ConnectorInput):
        reminder_id: str

        _id = field_validator("reminder_id")(_identifier)

    input_schema = Input

    def __init__(self, reminders) -> None:
        self._reminders = reminders

    def run(self, arguments: Input):
        return self._reminders.complete(arguments.reminder_id)


GMAIL_TOOLS = (
    "gmail_search_messages",
    "gmail_get_message",
    "gmail_get_thread",
    "gmail_create_draft",
    "gmail_get_draft",
    "gmail_send_draft",
)
CALENDAR_TOOLS = (
    "calendar_list_events",
    "calendar_check_availability",
    "calendar_get_event",
    "calendar_create_event",
    "calendar_update_event",
)
REMINDERS_TOOLS = ("reminders_list", "reminders_create", "reminders_complete")
PERSONAL_TOOLS = frozenset({"connectors_status", *GMAIL_TOOLS, *CALENDAR_TOOLS, *REMINDERS_TOOLS})


def register_personal_tools(
    registry, connectors: Connectors | None, now: Callable[[], datetime] | None = None
):
    """Register tools only for the connectors that are configured."""
    if connectors is None or not connectors.all():
        return registry
    clock = now if now is not None else (lambda: datetime.now(connectors.tz))
    registry.register(ConnectorsStatusTool(connectors))
    if connectors.gmail is not None:
        gmail = connectors.gmail
        for tool in (
            GmailSearchMessagesTool,
            GmailGetMessageTool,
            GmailGetThreadTool,
            GmailCreateDraftTool,
            GmailGetDraftTool,
            GmailSendDraftTool,
        ):
            registry.register(tool(gmail))
    if connectors.calendar is not None:
        calendar = connectors.calendar
        registry.register(CalendarListEventsTool(calendar, clock))
        registry.register(CalendarCheckAvailabilityTool(calendar, clock))
        registry.register(CalendarGetEventTool(calendar))
        registry.register(CalendarCreateEventTool(calendar, clock))
        registry.register(CalendarUpdateEventTool(calendar, clock))
    if connectors.reminders is not None:
        reminders = connectors.reminders
        registry.register(RemindersListTool(reminders))
        registry.register(RemindersCreateTool(reminders, clock))
        registry.register(RemindersCompleteTool(reminders))
    return registry
