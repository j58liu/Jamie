"""Job applications Jamie fills in the background and submits with the user's OK."""

from __future__ import annotations

from urllib.parse import urlparse

JOB_KIND = "job_application"
REVIEW = "review"  # stop at the final page and wait for the user
AUTO = "auto"  # submit without asking; the user chose this for the batch


def job_title(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return f"Apply: {parsed.netloc}{'/' + path if path else ''}"


def queue_applications(
    store, urls, *, auto_submit: bool = False, source: str = "gui", note: str = ""
) -> list[int]:
    """One background task per posting, so each can be reviewed, retried, or cancelled."""
    ids = []
    for url in dict.fromkeys(urls):
        ids.append(
            store.add_task(
                job_title(url),
                kind=JOB_KIND,
                payload={
                    "url": url,
                    "submit": AUTO if auto_submit else REVIEW,
                    "note": note[:2000],
                },
                source=source,
            )
        )
    return ids
