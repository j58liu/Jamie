"""`jamie tasks` and `jamie worker` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from assistant.config import ConfigurationError, ProjectConfig, load_config
from assistant.tasks.approvals import ApprovalError, approve, reject
from assistant.tasks.models import MissedPolicy, ScheduleKind, TaskState
from assistant.tasks.store import StaleTaskError, TaskStore, TaskStoreError
from assistant.ui.tasks import ACTIVE_STATES, approval_view, pending_approvals, task_rows

tasks_app = typer.Typer(help="Create, inspect, and approve persistent tasks.")
worker_app = typer.Typer(help="Run scheduled and pending tasks in the background.")

ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="Read a specific TOML configuration file.")
]


def _fail(message: str) -> None:
    Console(stderr=True).print(f"Error: {message}", markup=False, highlight=False)
    raise typer.Exit(code=1)


def _open(config: Path | None) -> tuple[ProjectConfig, TaskStore]:
    try:
        settings = load_config(config)
        return settings, TaskStore(settings.resolved_tasks_path())
    except (ConfigurationError, TaskStoreError) as exc:
        _fail(str(exc))
        raise  # unreachable


def _zone(settings: ProjectConfig) -> str:
    from assistant.connectors.timeutil import local_zone

    zone = local_zone(settings.timezone)
    return getattr(zone, "key", "UTC")


def _say(text: str) -> None:
    Console().print(text, markup=False, highlight=False)


def _find_task(store: TaskStore, prefix: str):
    task = store.find_task(prefix)
    if task is None:
        _fail(f"No single task matches {prefix!r}.")
    return task


@tasks_app.command("add")
def add(
    instruction: Annotated[str, typer.Argument(help="What Jamie should do.")],
    title: Annotated[
        str | None, typer.Option(help="Short name; defaults to the instruction.")
    ] = None,
    at: Annotated[str | None, typer.Option(help="Run once, e.g. 'tomorrow 8am'.")] = None,
    every: Annotated[
        str | None, typer.Option(help="Repeat: daily@08:00, weekly:sun@09:00, interval:2h.")
    ] = None,
    on_email: Annotated[
        str | None, typer.Option(help="Run when new Gmail messages match this search.")
    ] = None,
    scope: Annotated[str, typer.Option(help="Tools available to the task.")] = "personal",
    missed: Annotated[
        MissedPolicy, typer.Option(help="If the computer was asleep at run time.")
    ] = MissedPolicy.RUN_LATE,
    workspace: Annotated[Path | None, typer.Option(help="Folder for file tools.")] = None,
    config: ConfigOption = None,
) -> None:
    """Add a task now, at a time, on a repeating schedule, or on new email."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from assistant.tasks.scheduler import ScheduleError, next_run, once_at, parse_every
    from assistant.tools.scopes import ToolScope

    if sum(option is not None for option in (at, every, on_email)) > 1:
        raise typer.BadParameter("Use only one of --at, --every, or --on-email.")
    try:
        ToolScope(scope)
    except ValueError as exc:
        raise typer.BadParameter(f"Unknown scope {scope!r}.", param_hint="--scope") from exc
    settings, store = _open(config)
    name = title or (instruction if len(instruction) <= 60 else instruction[:57] + "...")
    folder = str((workspace or Path.cwd()).resolve())
    zone = _zone(settings)
    now = datetime.now(ZoneInfo(zone))
    try:
        if at is not None:
            when = once_at(at, now)
            kind, spec, first = ScheduleKind.ONCE, {"at": when.isoformat()}, when
        elif every is not None:
            kind, spec = parse_every(every)
            first = next_run(kind, spec, zone, now)
        elif on_email is not None:
            kind, spec, first = ScheduleKind.EVENT, {"source": "gmail", "query": on_email}, now
        else:
            task = store.create_task(name, instruction, scope=scope, workspace=folder)
            _say(f"Added task {task.id[:8]}: it runs on the next worker pass.")
            return
    except ScheduleError as exc:
        raise typer.BadParameter(str(exc)) from exc
    schedule = store.create_schedule(
        title=name, instruction=instruction, kind=kind, spec=spec, timezone=zone,
        next_run_at=first, scope=scope, workspace=folder, missed_policy=missed,
    )  # fmt: skip
    when_text = first.astimezone(ZoneInfo(zone)).strftime("%a %b %d %H:%M") if first else "-"
    _say(f"Scheduled {schedule.id[:8]} ({kind.value}); next run {when_text}.")


@tasks_app.command("list")
def list_tasks(
    all_tasks: Annotated[bool, typer.Option("--all", help="Include finished tasks.")] = False,
    config: ConfigOption = None,
) -> None:
    """List tasks and schedules."""
    _, store = _open(config)
    tasks = store.list_tasks(None if all_tasks else ACTIVE_STATES)
    for line in task_rows(tasks):
        _say(line)
    schedules = [schedule for schedule in store.list_schedules() if schedule.enabled]
    if schedules:
        _say("")
        _say("Schedules:")
        for schedule in schedules:
            next_text = (
                schedule.next_run_at.astimezone().strftime("%b %d %H:%M")
                if schedule.next_run_at
                else "-"
            )
            _say(f"{schedule.id[:8]}  {schedule.kind.value:<8}  next {next_text}  "
                 f"{schedule.title}")  # fmt: skip
    waiting = len(pending_approvals(store))
    if waiting:
        _say(f"\n{waiting} waiting for approval. Run: jamie tasks approvals")


@tasks_app.command("show")
def show(task_id: str, config: ConfigOption = None) -> None:
    """Show a task's state, steps, and approvals."""
    _, store = _open(config)
    task = _find_task(store, task_id)
    _say(f"{task.title}\n{task.instruction}\n")
    _say(f"State: {task.state.value} · revision {task.revision} · attempts {task.attempt_count}"
         f" · scope {task.scope}")  # fmt: skip
    if task.last_error:
        _say(f"Last error: {task.last_error}")
    if task.result_summary:
        _say(f"Result: {task.result_summary}")
    for step in store.steps(task.id):
        _say(f"  {step.step_index:>2}. {step.tool_name:<28} {step.status.value:<10} "
             f"{step.summary[:80]}")  # fmt: skip
    for approval in store.approvals(task_id=task.id):
        _say(f"  approval {approval.id[:8]} {approval.tool} {approval.status.value}")


@tasks_app.command("cancel")
def cancel(task_id: str, config: ConfigOption = None) -> None:
    """Cancel a task that is not currently running."""
    _, store = _open(config)
    task = _find_task(store, task_id)
    if task.state is TaskState.RUNNING:
        _fail("The task is running; cancel it after this run finishes.")
    try:
        store.transition(task.id, TaskState.CANCELLED, expected=task.state)
    except (StaleTaskError, ValueError) as exc:
        _fail(str(exc))
    _say(f"Cancelled {task.id[:8]}.")


@tasks_app.command("start")
def start(task_id: str, config: ConfigOption = None) -> None:
    """Queue a scheduled task (such as a remembered commitment) to run now."""
    _, store = _open(config)
    task = _find_task(store, task_id)
    try:
        store.transition(task.id, TaskState.READY, expected=TaskState.SCHEDULED)
    except (StaleTaskError, ValueError) as exc:
        _fail(str(exc))
    _say(f"Queued {task.id[:8]}; it runs on the next worker pass.")


@tasks_app.command("unschedule")
def unschedule(schedule_id: str, config: ConfigOption = None) -> None:
    """Stop a repeating or one-time schedule."""
    _, store = _open(config)
    matches = [s for s in store.list_schedules() if s.id.startswith(schedule_id)]
    if len(matches) != 1:
        _fail(f"No single schedule matches {schedule_id!r}.")
    store.disable_schedule(matches[0].id)
    _say(f"Stopped schedule {matches[0].id[:8]}.")


@tasks_app.command("edit")
def edit(
    task_id: str,
    instruction: Annotated[str, typer.Option(help="The new instruction.")],
    config: ConfigOption = None,
) -> None:
    """Change a task's instruction; outstanding approvals no longer apply."""
    _, store = _open(config)
    task = _find_task(store, task_id)
    try:
        updated = store.revise(task.id, instruction)
    except StaleTaskError as exc:
        _fail(str(exc))
    _say(f"Updated {task.id[:8]} to revision {updated.revision}.")


@tasks_app.command("approvals")
def approvals(config: ConfigOption = None) -> None:
    """Show actions waiting for your approval."""
    _, store = _open(config)
    pending = pending_approvals(store)
    console = Console()
    if not pending:
        _say("Nothing is waiting for approval.")
        return
    for approval in pending:
        console.print(approval_view(store, approval))
    _say("Approve with: jamie tasks approve ID   Reject with: jamie tasks reject ID")


def _approval_id(store: TaskStore, prefix: str) -> str:
    approval = store.find_approval_by_prefix(prefix)
    if approval is None:
        _fail(f"No single approval matches {prefix!r}.")
    return approval.id


@tasks_app.command("approve")
def approve_command(
    approval_id: str,
    run: Annotated[bool, typer.Option("--run", help="Run the task now.")] = False,
    config: ConfigOption = None,
) -> None:
    """Approve exactly the action shown in `jamie tasks approvals`."""
    _, store = _open(config)
    try:
        approval = approve(store, _approval_id(store, approval_id))
    except ApprovalError as exc:
        _fail(str(exc))
    _say(f"Approved. Task {approval.task_id[:8]} will continue on the next worker pass.")
    if run:
        run_worker(once=True, config=config)


@tasks_app.command("reject")
def reject_command(approval_id: str, config: ConfigOption = None) -> None:
    """Reject the action; the task is cancelled and the action never runs."""
    _, store = _open(config)
    try:
        approval = reject(store, _approval_id(store, approval_id))
    except ApprovalError as exc:
        _fail(str(exc))
    _say(f"Rejected. Task {approval.task_id[:8]} was cancelled.")


# -- worker -----------------------------------------------------------------------------


def build_worker(settings: ProjectConfig, store: TaskStore):
    """Wire the worker to the real model, tools, connectors, and notifications."""
    from assistant.connectors import build_connectors
    from assistant.llm.base import LLMError
    from assistant.llm.llamacpp import LlamaCppBackend
    from assistant.tasks.engine import TaskEngine
    from assistant.tasks.notifications import default_notifier
    from assistant.tasks.worker import Worker
    from assistant.tools import WebClient, create_agent_registry
    from assistant.tools.scopes import select_tools

    connectors = build_connectors(settings)
    notifier = default_notifier()

    def model_available() -> bool:
        backend = LlamaCppBackend()
        try:
            backend.server_info()
            return True
        except LLMError:
            return False
        finally:
            backend.close()

    def engine_factory():
        backend = LlamaCppBackend()
        web = WebClient()

        def registry_factory(task, approvals):
            root = Path(task.workspace) if task.workspace else Path.home()
            registry = create_agent_registry(
                root, approvals=approvals, web=web, backend=backend, connectors=connectors
            )
            return select_tools(registry, task.scope)

        return TaskEngine(store, backend_factory=lambda: backend,
                          registry_factory=registry_factory, notifier=notifier)  # fmt: skip

    return Worker(store, engine_factory=engine_factory, model_available=model_available,
                  connectors=connectors, notifier=notifier)  # fmt: skip


@worker_app.command("run")
def run_worker(
    once: Annotated[bool, typer.Option("--once", help="Do one pass and exit.")] = False,
    config: ConfigOption = None,
) -> None:
    """Check schedules, recover interrupted work, and run ready tasks."""
    import time

    settings, store = _open(config)
    worker = build_worker(settings, store)
    while True:
        report = worker.run_once()
        if report.ran or report.tick.created or report.recovery.ambiguous or once:
            _say(
                f"created {len(report.tick.created)} · ran {len(report.ran)} · "
                f"waiting {len(pending_approvals(store))}"
                + (" · model not running" if report.model_unavailable else "")
            )
        if once:
            return
        time.sleep(60)


@worker_app.command("install")
def install(config: ConfigOption = None) -> None:
    """Run the worker every 5 minutes in the background with launchd (macOS)."""
    import sys

    from assistant.tasks.worker import install_launch_agent
    from assistant.tools.base import ToolError

    if sys.platform != "darwin":
        _fail("The background worker uses launchd and needs macOS.")
    settings, store = _open(config)
    config_path = (config or Path("assistant.toml")).resolve()
    try:
        path = install_launch_agent(
            working_directory=Path.cwd(),
            log_directory=store.path.parent / "logs",
            config=config_path if config_path.exists() else None,
        )
    except ToolError as exc:
        _fail(str(exc))
    _say(f"Installed {path}. Jamie checks for due tasks every 5 minutes.")


@worker_app.command("uninstall")
def uninstall() -> None:
    """Stop and remove the background worker."""
    from assistant.tasks.worker import uninstall_launch_agent

    removed = uninstall_launch_agent()
    _say("Removed the background worker." if removed else "The worker was not installed.")


def task_summary(store: TaskStore) -> Text | None:
    waiting = len(pending_approvals(store))
    if not waiting:
        return None
    noun = "task is" if waiting == 1 else "tasks are"
    return Text(f"{waiting} {noun} waiting for your approval · /approvals", style="yellow")
