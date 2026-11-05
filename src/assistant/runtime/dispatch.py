"""Act on a routing decision that needs no agent turn: queue, schedule, or apply."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from assistant import schedule
from assistant.routing import BACKGROUND, SCHEDULE, Route, schedule_request

FIND_JOBS = (
    "Find up to 10 current job postings that match this request: {request}\n"
    "Use web search. For each, give the title, company, location, and the direct link to "
    "the application page. Say which application sites they use (Greenhouse, Lever, "
    "Ashby, Workday, LinkedIn, Indeed) when you can tell."
)


def _when(moment: datetime) -> str:
    return moment.astimezone().strftime("%a %d %b %H:%M")


async def dispatch(application, store, prompt: str, decision: Route, *, source: str) -> str | None:
    """Return Jamie's reply when the request was handled here; None to run it now."""
    if decision.jobs:
        if decision.urls:
            from assistant.jobs import queue_applications

            ids = queue_applications(
                store, decision.urls, auto_submit=decision.auto_submit, source=source,
                note=prompt,
            )  # fmt: skip
            count = len(ids)
            plural = "application" if count == 1 else "applications"
            if decision.auto_submit:
                finish = "I'll submit each one myself and show you the confirmation pages."
            else:
                finish = (
                    "When each one is filled in, I'll show you the final page so you can "
                    "approve the submission."
                )
            return (
                f"Working on {count} job {plural} in the background (tasks "
                f"{', '.join(map(str, ids))}). Keep doing your thing. {finish}"
            )
        task_id = store.add_task(
            FIND_JOBS.format(request=prompt),
            source=source,
            payload={"scopes": ["web", "browser"]},
        )
        return (
            "I'll look for matching postings in the background (task "
            f"{task_id}) and list them with their links. Tell me which ones to apply to, "
            "or paste links to postings you already picked."
        )
    if decision.timing == BACKGROUND:
        task_id = store.add_task(prompt, source=source, payload={"scopes": list(decision.scopes)})
        return (
            f"On it in the background (task {task_id}). Keep working; I'll notify you when "
            "it's done, and ask here first if anything needs your OK."
        )
    if decision.timing == SCHEDULE:
        backend = application.backend
        try:
            request = await asyncio.to_thread(schedule_request, backend, prompt)
        except ValueError as exc:
            return f"{exc} Try saying it like: every weekday at 8:00, summarize my email."
        now = datetime.now(UTC)
        first = schedule.next_run(request.repeat, request.time, now)
        if first <= now:
            return "That time has already passed. When should I do it instead?"
        task_id = application.memory.add_schedule(
            request.prompt, request.repeat, request.time, first
        )
        how = schedule.describe(request.repeat, request.time)
        return (
            f"Scheduled: “{request.prompt}” {how} (task {task_id}), first on {_when(first)}. "
            "I'll send a notification each time. To stop it, just tell me, for example "
            f"“remove scheduled task {task_id}”."
        )
    return None
