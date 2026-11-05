"""`jamie bg`: hand Jamie work, keep working, and answer its questions later."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assistant.background import BackgroundStore, BackgroundStoreError
from assistant.background.store import ACTIVE
from assistant.config import ConfigurationError, load_config

bg_app = typer.Typer(help="Run tasks in the background and answer their approvals.")


def _store() -> BackgroundStore:
    try:
        return BackgroundStore()
    except BackgroundStoreError as exc:
        _fail(str(exc))
        raise  # unreachable; _fail exits


def _fail(message: str) -> None:
    Console(stderr=True).print(f"Error: {message}", markup=False, highlight=False)
    raise typer.Exit(code=1)


def _say(text: str) -> None:
    Console().print(text, markup=False, highlight=False)


@bg_app.command("add")
def add(
    prompt: Annotated[str, typer.Argument(help="What Jamie should do.")],
) -> None:
    """Queue a task; Jamie.app or `jamie serve` runs it while you keep working."""
    if not prompt.strip():
        raise typer.BadParameter("Prompt must not be empty.", param_hint="PROMPT")
    task_id = _store().add_task(prompt, source="cli")
    _say(f"Queued task {task_id}. Check on it with: jamie bg list")


@bg_app.command("list")
def list_tasks(
    all_tasks: Annotated[bool, typer.Option("--all", help="Include finished tasks.")] = False,
) -> None:
    """Show queued, running, and waiting tasks (or all recent ones)."""
    store = _store()
    tasks = store.tasks(active_only=not all_tasks, limit=30)
    if not tasks:
        _say("No background tasks." if all_tasks else "Nothing running. Use --all for history.")
        return
    table = Table(box=None, pad_edge=False)
    for column in ("id", "status", "task", "progress"):
        table.add_column(column)
    for task in tasks:
        table.add_row(str(task.id), task.status, task.prompt[:60], task.progress[:40])
    Console().print(table)
    pending = store.pending_approvals()
    if pending:
        _say(f"\n{len(pending)} approval(s) waiting. See them with: jamie bg approvals")


@bg_app.command("show")
def show(task_id: Annotated[int, typer.Argument()]) -> None:
    """Show one task's result or error."""
    task = _store().task(task_id)
    if task is None:
        _fail(f"No background task {task_id}.")
        return
    _say(f"Task {task.id} [{task.status}] ({task.kind}, from {task.source})")
    _say(task.prompt)
    if task.progress:
        _say(f"Progress: {task.progress}")
    if task.result:
        _say(f"\n{task.result}")
    if task.error:
        _say(f"\nError: {task.error}")
    for key in ("review_screenshot", "confirmation_screenshot"):
        if task.payload.get(key):
            _say(f"{key.replace('_', ' ').capitalize()}: {task.payload[key]}")


@bg_app.command("cancel")
def cancel(task_id: Annotated[int, typer.Argument()]) -> None:
    """Stop a queued or running task."""
    if _store().request_cancel(task_id):
        _say(f"Cancelling task {task_id}.")
    else:
        _fail(f"Task {task_id} is not queued or running.")


@bg_app.command("retry")
def retry(task_id: Annotated[int, typer.Argument()]) -> None:
    """Queue a finished, failed, or interrupted task again."""
    if _store().retry(task_id):
        _say(f"Queued task {task_id} again.")
    else:
        _fail(f"Task {task_id} has not finished, or does not exist.")


@bg_app.command("approvals")
def approvals() -> None:
    """List what background tasks are waiting for you to approve."""
    pending = _store().pending_approvals()
    if not pending:
        _say("Nothing is waiting for approval.")
        return
    for approval in pending:
        _say(f"── {approval.id}  (task {approval.task_id}, {approval.tool})")
        _say(approval.summary)
        if approval.attachment:
            _say(f"Screenshot: {approval.attachment}")
        _say(f"Approve: jamie bg approve {approval.id}   Deny: jamie bg deny {approval.id}\n")


@bg_app.command("approve")
def approve(approval_id: Annotated[str, typer.Argument()]) -> None:
    """Allow one waiting action, exactly as it was shown."""
    if _store().answer(approval_id, True, by="cli"):
        _say("Approved.")
    else:
        _fail("That approval is not waiting (answered, expired, or unknown).")


@bg_app.command("deny")
def deny(approval_id: Annotated[str, typer.Argument()]) -> None:
    """Refuse one waiting action; the task stops instead of working around it."""
    if _store().answer(approval_id, False, by="cli"):
        _say("Denied.")
    else:
        _fail("That approval is not waiting (answered, expired, or unknown).")


def active_count(store: BackgroundStore) -> int:
    return sum(1 for task in store.tasks(limit=200) if task.status in ACTIVE)


def serve_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Run background tasks without the window (Jamie.app does this too)."""
    from assistant.runtime.daemon import run_serve

    try:
        settings = load_config(config)
    except ConfigurationError as exc:
        _fail(str(exc))
        return
    run_serve(settings)
