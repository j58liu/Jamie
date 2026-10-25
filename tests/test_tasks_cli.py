import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.security import ApprovalRequest, PermissionLevel
from assistant.tasks import TaskState
from assistant.tasks.approvals import DeferredApprovalProvider
from assistant.tasks.store import TaskStore

runner = CliRunner()


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "tasks.sqlite3"
    monkeypatch.setenv("JAMIE_TASKS_PATH", str(path))
    monkeypatch.chdir(tmp_path)
    return lambda: TaskStore(path)


def waiting(store):
    task = store.create_task("Reply to Alex", "reply")
    store.transition(task.id, TaskState.RUNNING)
    provider = DeferredApprovalProvider(store, task)
    provider.request(
        ApprovalRequest(
            tool="gmail_send_draft", permission=PermissionLevel.SEND, arguments={"draft_id": "r-1"}
        )
    )
    store.transition(task.id, TaskState.WAITING_APPROVAL)
    return task, provider.deferred


def test_add_now_and_scheduled(store):
    result = runner.invoke(cli.app, ["tasks", "add", "Summarize my inbox"])
    assert result.exit_code == 0, result.output
    assert "Added task" in result.output
    result = runner.invoke(cli.app, ["tasks", "add", "Morning brief", "--every", "daily@08:00"])
    assert "Scheduled" in result.output and "(daily)" in result.output
    result = runner.invoke(cli.app, ["tasks", "add", "Call", "--at", "tomorrow 8am"])
    assert "(once)" in result.output
    listing = runner.invoke(cli.app, ["tasks", "list"])
    assert "Summarize my inbox" in listing.output and "Schedules:" in listing.output
    bad = runner.invoke(cli.app, ["tasks", "add", "x", "--every", "hourly"])
    assert bad.exit_code == 2


def test_approvals_show_bound_details_and_approve(store):
    task, approval = waiting(store())
    shown = runner.invoke(cli.app, ["tasks", "approvals"])
    assert "SEND the Gmail draft" in shown.output and "r-1" in shown.output
    assert "revision 1" in shown.output and "google:default" in shown.output
    result = runner.invoke(cli.app, ["tasks", "approve", approval.id[:8]])
    assert result.exit_code == 0, result.output
    assert store().get_task(task.id).state is TaskState.READY
    again = runner.invoke(cli.app, ["tasks", "approve", approval.id[:8]])
    assert again.exit_code == 1 and "already approved" in again.output


def test_reject_edit_and_cancel(store):
    task, approval = waiting(store())
    assert runner.invoke(cli.app, ["tasks", "reject", approval.id[:8]]).exit_code == 0
    assert store().get_task(task.id).state is TaskState.CANCELLED

    other = store().create_task("Other", "x")
    edited = runner.invoke(cli.app, ["tasks", "edit", other.id[:8], "--instruction", "y"])
    assert "revision 2" in edited.output
    assert runner.invoke(cli.app, ["tasks", "cancel", other.id[:8]]).exit_code == 0
    shown = runner.invoke(cli.app, ["tasks", "show", other.id[:8]])
    assert "State: cancelled" in shown.output
