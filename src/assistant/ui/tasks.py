"""Terminal rendering for tasks and deferred approvals (CLI and REPL)."""

from __future__ import annotations

from datetime import datetime

from rich.panel import Panel
from rich.text import Text

from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel
from assistant.tasks.approvals import approval_display
from assistant.tasks.models import ApprovalStatus, Task, TaskApproval, TaskState
from assistant.tasks.store import TaskStore
from assistant.ui.approvals import approval_panel, visible

ACTIVE_STATES = (TaskState.READY, TaskState.RUNNING, TaskState.WAITING_APPROVAL,
                 TaskState.SCHEDULED)  # fmt: skip


def _when(value: datetime | None) -> str:
    return value.astimezone().strftime("%b %d %H:%M") if value else "-"


def task_rows(tasks: list[Task]) -> list[str]:
    if not tasks:
        return ["No tasks."]
    width = max(len(task.state.value) for task in tasks)
    return [
        f"{task.id[:8]}  {task.state.value:<{width}}  {_when(task.updated_at)}  {task.title}"
        for task in tasks
    ]


def approval_view(store: TaskStore, approval: TaskApproval) -> Panel:
    """The standard approval panel, plus what the approval is bound to."""
    display = approval_display(store, approval)
    task = store.get_task(approval.task_id)
    try:
        permission = PermissionLevel(display.get("permission", ""))
    except ValueError:
        permission = PermissionLevel.SAFE_WRITE
    arguments = dict(display.get("arguments") or {})
    if display.get("account"):
        arguments["account"] = display["account"]
    request = ApprovalRequest(
        id=approval.id, tool=approval.tool, permission=permission, arguments=arguments
    )
    panel = approval_panel(request)
    extra = [Text("")]
    if display.get("note"):
        extra.append(Text(visible(display["note"]), style="italic"))
    extra.append(
        Text(
            f"Task: {visible(task.title)} (revision {approval.task_revision}) · "
            f"approval {approval.id[:8]} · expires {_when(approval.expires_at)}",
            style="dim",
        )
    )
    from rich.console import Group

    return Panel(Group(panel.renderable, *extra), title="Approval required", title_align="left",
                 border_style="yellow", expand=False)  # fmt: skip


def pending_approvals(store: TaskStore) -> list[TaskApproval]:
    return store.approvals(status=ApprovalStatus.PENDING)
