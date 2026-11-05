"""Run one job application as a background task.

Open the posting, fill every page from the applicant profile, screenshot the final
page, and wait for the user's OK, unless they chose to let Jamie submit this batch by
itself. After submitting, save the confirmation page. Anything only the user can answer
stops the task with their questions; once answered, a retry picks up from the start
with the new answers saved.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from assistant.background import Grant, TaskNeedsInput
from assistant.background.worker import TaskContext
from assistant.jobs import AUTO, JOB_KIND
from assistant.jobs.filler import SUCCESS, ApplicationFiller, Drafter, NeedsUser
from assistant.jobs.profile import ProfileStore
from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel

SUBMIT_TOOL = "job_submit"
CONFIRMATION_WAIT_SECONDS = 15.0


def grants_for_task(task) -> tuple[Grant, ...]:
    """The user said to submit this batch without asking: allow job_submit for it only."""
    if task.kind == JOB_KIND and task.payload.get("submit") == AUTO:
        return (Grant(SUBMIT_TOOL),)
    return ()


def _resume_text(path: str) -> str:
    if not path:
        return ""
    try:
        from assistant.tools.mac_files import file_text

        return file_text(Path(path).expanduser())
    except Exception:  # noqa: BLE001 - the résumé helps answers; it is not required
        return ""


def _submitted_today(store) -> int:
    today = datetime.now(UTC).date().isoformat()
    return sum(
        1
        for task in store.tasks(limit=500)
        if task.kind == JOB_KIND
        and task.payload.get("submitted")
        and (task.finished_at or "").startswith(today)
    )


def job_application_runner(application, settings, browser, *, profiles=None, pause=0.15):
    profiles = profiles or ProfileStore()

    async def run(task, context: TaskContext) -> str:
        return await asyncio.to_thread(_apply, task, context)

    def _apply(task, context: TaskContext) -> str:
        url = task.payload.get("url", "")
        profile = profiles.load()
        missing = profile.missing()
        if missing:
            raise TaskNeedsInput(
                "Your applicant profile is missing some details. Tell Jamie, for example "
                "“for job applications, my phone is …”, or “use my résumé to set up job "
                "applications”.",
                [name.replace("_", " ") for name in missing],
            )
        limit = settings.jobs.max_per_day
        if _submitted_today(context.store) >= limit:
            raise TaskNeedsInput(
                f"Jamie already submitted {limit} applications today, the daily limit in "
                "settings ([jobs] max_per_day). Retry tomorrow, or raise the limit."
            )
        if browser is None:
            raise RuntimeError("Jamie's browser is turned off ([browser] enabled = false).")
        tab = browser.tab()
        try:
            return _fill_and_submit(task, context, tab, profile, url)
        except NeedsUser as exc:
            raise TaskNeedsInput(str(exc), exc.questions) from exc
        except InterruptedError:
            from assistant.cancellation import TurnCancelled

            raise TurnCancelled from None
        finally:
            tab.close()

    def _fill_and_submit(task, context, tab, profile, url) -> str:
        context.progress("Opening the posting")
        filler = ApplicationFiller(
            tab,
            profile,
            progress=context.progress,
            cancelled=lambda: context.cancel.cancelled,
            pause=pause,
        )
        state, snapshot = filler.open(url)
        title = snapshot.title or url
        filler.drafter = Drafter(
            application.backend, profile, _resume_text(profile.resume_path), state.job_text
        )
        snapshot = filler.fill_until_final(state, snapshot)
        if SUCCESS.search(snapshot.text):
            shot = tab.screenshot(f"confirmation-{task.id}")
            context.store.update_payload(task.id, confirmation_screenshot=str(shot), submitted=True)
            return f"Applied: {title}. The site confirmed it; screenshot saved."
        final = filler.final_button(snapshot)
        review = tab.screenshot(f"review-{task.id}")
        drafted = [f"{f.label}: {f.value}" for f in state.filled if f.how == "drafted"]
        context.store.update_payload(
            task.id,
            review_screenshot=str(review),
            job=title,
            filled=[f"{f.label}: {f.value}" for f in state.filled],
            drafted=drafted,
        )
        request = ApprovalRequest(
            tool=SUBMIT_TOOL,
            permission=PermissionLevel.SEND,
            arguments={
                "job": title,
                "page": snapshot.url,
                "button": final.label,
                "answers": state.summary(),
                **({"written_by_jamie": "\n".join(drafted)} if drafted else {}),
            },
        )
        context.progress("Filled in. Waiting for your OK to submit")
        approvals = context.approvals(attachment=lambda: str(review))
        if not approvals.request(request):
            return (
                f"Not submitted: {title}. You chose not to send it; the filled form is in "
                "the review screenshot."
            )
        context.progress("Submitting")
        current = tab.snapshot()
        again = filler.final_button(current)
        if again is None or again.label != final.label:
            raise RuntimeError(
                "The application page changed while it waited for you, so nothing was "
                "submitted. Retry it to fill it again."
            )
        result = tab.click(again.ref)
        deadline = time.monotonic() + CONFIRMATION_WAIT_SECONDS
        while not SUCCESS.search(result.text) and time.monotonic() < deadline:
            time.sleep(1.0)
            result = tab.snapshot()
        confirmation = tab.screenshot(f"confirmation-{task.id}")
        confirmed = bool(SUCCESS.search(result.text))
        context.store.update_payload(
            task.id,
            confirmation_screenshot=str(confirmation),
            submitted=True,
            confirmed=confirmed,
        )
        if confirmed:
            return f"Submitted: {title}. The confirmation page is saved."
        return (
            f"Clicked “{final.label}” for {title}, but no confirmation message appeared. "
            "Check the saved screenshot."
        )

    return run
