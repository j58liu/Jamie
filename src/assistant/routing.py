"""Decide from the user's own words what Jamie should do with a message.

The user just talks; Jamie works out whether to answer, use tools now (and which ones),
start work in the background, schedule it, or apply to jobs. Each decision is one
forward pass of the local model reading a single answer letter (see
``option_probabilities``), so routing costs a fraction of a second and no generated text.
Explicit phrases ("in the background", "every morning") win over the model, and anything
uncertain falls back to doing it now with every tool, which is how Jamie behaved before.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from assistant.llm.base import Message, constrained_to

NOW = "now"
BACKGROUND = "background"
SCHEDULE = "schedule"

# What the request is about -> the tool scope that serves it.
SUBJECTS: dict[str, tuple[str, str]] = {
    "A": ("chat", "A question or conversation Jamie can answer without apps, files, or web"),
    "B": ("files", "Find, read, organize, or edit files and documents on this Mac"),
    "C": ("personal", "Email, calendar events, or reminders"),
    "D": ("web", "Look something up on the internet or research a topic"),
    "E": ("browser", "Do something on a website or online account (log in, fill a form, book)"),
    "F": ("macos", "Open or control Mac apps, the clipboard, or Shortcuts"),
    "G": ("jobs", "Apply to jobs or job postings"),
    "H": ("tasks", "Jamie's own background tasks, schedules, or applications in progress"),
}
TIMINGS: dict[str, tuple[str, str]] = {
    "A": (NOW, "Do it now while the user waits for the answer"),
    "B": (BACKGROUND, "Work on it in the background while the user does other things"),
    "C": (SCHEDULE, "Do it later at a set time, or repeatedly on a schedule"),
}

CHAT_CONFIDENCE = 0.6  # below this, a "just talk" answer still gets tools
SUBJECT_CONFIDENCE = 0.6  # below this, the runner-up's tools are added too
RUNNER_UP = 0.2
TIMING_CONFIDENCE = 0.55  # background and schedules need a clear signal

_BACKGROUND_CUES = re.compile(
    r"\b(in the background|while i(?:'m| am)?\b|meanwhile|take your time|"
    r"no rush|when you(?:'re| are)? done|let me know when|ping me when|"
    r"notify me when|get back to me)",
    re.IGNORECASE,
)
_SCHEDULE_CUES = re.compile(
    r"\b(every (?:day|morning|evening|night|week|weekday|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|\d+ ?(?:min|minutes|hours?|h|days?))|daily|weekly|"
    r"each (?:day|morning|week)|on a schedule)\b",
    re.IGNORECASE,
)
_REMINDER = re.compile(r"\bremind me\b", re.IGNORECASE)
_APPLY = re.compile(
    r"\bapply(?:ing)?\b.{0,40}\b(jobs?|positions?|roles?|postings?|openings?)\b|"
    r"\b(jobs?|positions?|roles?|postings?)\b.{0,40}\bapply\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_AUTO_SUBMIT = re.compile(
    r"\b(submit (?:them|it|everything|all)?\s*(?:yourself|for me|without asking|"
    r"automatically|directly)|auto[- ]?submit|don'?t (?:wait|ask) (?:for )?me|"
    r"go ahead and submit|you can submit)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Route:
    """What to do with one message."""

    timing: str = NOW
    scopes: tuple[str, ...] = ("all",)
    subject: str = "all"
    confidence: float = 0.0
    reason: str = ""
    urls: tuple[str, ...] = ()
    auto_submit: bool = False
    apply: bool = False  # the user asked to apply (queue applications, no agent turn)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def chat(self) -> bool:
        return self.timing == NOW and self.scopes == ("chat",)

    @property
    def jobs(self) -> bool:
        return self.subject == "jobs"


def _question(message: str, prompt: str, options: dict[str, tuple[str, str]]) -> list[Message]:
    # Options after the message scored best on Qwen3.5-4B (see knowledge.extraction).
    listed = "\n".join(f"{letter}. {text}" for letter, (_, text) in options.items())
    return [
        Message(
            "user",
            f"Message: {json.dumps(message, ensure_ascii=False)}\n\n{prompt}\n{listed}\n\n"
            "Answer with the letter only.",
        )
    ]


def _choose(backend, message: str, prompt: str, options) -> dict[str, float]:
    choose = getattr(backend, "option_probabilities", None)
    if choose is None:
        return {}
    try:
        return choose(_question(message, prompt, options), "".join(options))
    except Exception:  # noqa: BLE001 - routing is an optimization; never fail the turn
        return {}


def find_urls(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for url in _URL.findall(text):
        url = url.rstrip(".,;:!?")
        if url not in seen:
            seen.append(url)
    return tuple(seen)


def route(backend, message: str, *, available: set[str] | None = None) -> Route:
    """Route one message; ``available`` names the tool scopes this front end can offer."""
    text = message.strip()
    urls = find_urls(text)
    auto_submit = bool(_AUTO_SUBMIT.search(text))
    if _APPLY.search(text):
        timing = SCHEDULE if _SCHEDULE_CUES.search(text) else BACKGROUND
        return Route(
            timing, ("jobs",), "jobs", 1.0, "asked to apply", urls, auto_submit, apply=True
        )

    subjects = _choose(backend, text, "What does this message ask Jamie to work with?", SUBJECTS)
    if subjects:
        ranked = sorted(subjects.items(), key=lambda item: item[1], reverse=True)
        (best, best_p), (second, second_p) = ranked[0], ranked[1]
        subject = SUBJECTS[best][0]
        scopes = [subject]
        if subject == "chat" and best_p < CHAT_CONFIDENCE:
            scopes = ["all"]
        elif best_p < SUBJECT_CONFIDENCE and second_p >= RUNNER_UP:
            scopes.append(SUBJECTS[second][0])
            scopes = [s for s in scopes if s != "chat"] or ["all"]
        confidence = best_p
    else:
        subject, scopes, confidence = "all", ["all"], 0.0

    if available is not None:
        scopes = [s for s in scopes if s in available or s == "chat"] or ["all"]

    timing, reason = NOW, "default"
    if _BACKGROUND_CUES.search(text):
        timing, reason = BACKGROUND, "asked for background work"
    elif _SCHEDULE_CUES.search(text) and not _REMINDER.search(text):
        timing, reason = SCHEDULE, "asked for a repeating task"
    elif subject != "chat" and subjects:
        timings = _choose(backend, text, "When should Jamie do this?", TIMINGS)
        if timings:
            best = max(timings, key=timings.get)
            kind = TIMINGS[best][0]
            if kind != NOW and timings[best] >= TIMING_CONFIDENCE:
                timing, reason = kind, f"model ({timings[best]:.2f})"
    if timing != NOW and scopes == ["chat"]:
        scopes = ["all"]
    return Route(
        timing,
        tuple(scopes),
        subject,
        confidence,
        reason,
        urls,
        auto_submit,
        scores=dict(subjects),
    )


# -- turning a request into a schedule ------------------------------------------------

SCHEDULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "repeat": {"enum": ["daily", "weekly", "every", "once"]},
        "day": {"enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun", ""]},
        "time": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
        "date": {"type": "string", "pattern": "^(\\d{4}-\\d{2}-\\d{2})?$"},
        "interval": {"type": "string", "pattern": "^(\\d{1,3}[mhd])?$"},
        "task": {"type": "string", "minLength": 1, "maxLength": 1000},
    },
    "required": ["repeat", "day", "time", "date", "interval", "task"],
    "additionalProperties": False,
}

SCHEDULE_INSTRUCTIONS = """\
Turn the user's request into a task Jamie runs by itself later.
Now: {now}.
- repeat: "daily", "weekly" (set day), "every" (set interval such as "2h", "30m", "1d"),
  or "once" (set date, YYYY-MM-DD, and time).
- time: 24-hour HH:MM. Use 08:00 for "morning", 12:00 for "noon", 18:00 for "evening"
  when no time is given.
- task: what Jamie should do each time, written as an instruction, without the timing
  words. For example "every morning, summarize my unread email" -> "Summarize my unread
  email."
Leave fields that do not apply as empty strings, but always set time (use 09:00 if
none applies).\
"""


@dataclass(frozen=True)
class ScheduleRequest:
    repeat: str
    time: str
    prompt: str

    def command(self) -> str:
        """The /schedule form, so the same parser validates both paths."""
        if self.repeat == "every":
            return f"every {self.time} {self.prompt}"
        if self.repeat == "once" and "-" in self.time:
            return f"on {self.time} {self.prompt}"
        if self.repeat == "once":
            return f"at {self.time} {self.prompt}"
        return f"{self.repeat} {self.time} {self.prompt}"


def schedule_request(backend, message: str, now: datetime | None = None) -> ScheduleRequest:
    """Ask the model to fill a constrained form, then validate it like /schedule."""
    from assistant import schedule

    moment = (now or datetime.now(UTC)).astimezone()
    instructions = SCHEDULE_INSTRUCTIONS.format(now=moment.strftime("%A %Y-%m-%d %H:%M"))
    with constrained_to(SCHEDULE_SCHEMA):
        response = backend.chat(
            [Message("system", instructions), Message("user", message)],
            temperature=0.0,
            max_tokens=300,
        )
    from assistant.agent.protocol import _remove_thinking

    try:
        data = json.loads(_remove_thinking(response.content))
    except (ValueError, TypeError) as exc:
        raise ValueError("I could not work out when to run that.") from exc
    repeat, task = data.get("repeat"), str(data.get("task", "")).strip()
    hhmm = data.get("time") or "09:00"
    if repeat == "weekly":
        time = f"{data.get('day') or 'mon'} {hhmm}"
    elif repeat == "every":
        time = data.get("interval") or "1d"
    elif repeat == "once" and data.get("date"):
        time = f"{data['date']} {hhmm}"
    else:
        time = hhmm
    request = ScheduleRequest(repeat or "daily", time, task or message)
    parsed = schedule.parse(request.command())  # raises ValueError with the usage text
    return ScheduleRequest(*parsed)
