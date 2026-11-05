import asyncio
import threading
from datetime import timedelta

import pytest

from assistant.background import (
    BackgroundStore,
    BackgroundWorker,
    DeferredApprovalProvider,
    Grant,
    acquire_worker_lock,
    arguments_digest,
)
from assistant.background.store import (
    CANCELLED,
    DONE,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    WAITING,
)
from assistant.cancellation import CancellationToken
from assistant.security.approvals import ApprovalRequest
from assistant.security.policy import PermissionLevel


@pytest.fixture
def store(tmp_path):
    return BackgroundStore(tmp_path / "background.sqlite3")


def request(tool="gmail_send_draft", **arguments):
    return ApprovalRequest(
        tool=tool, permission=PermissionLevel.SEND, arguments=arguments or {"draft_id": "r-1"}
    )


def test_tasks_are_claimed_once_in_order(store):
    first = store.add_task("one")
    second = store.add_task("two")
    assert store.claim_next().id == first
    assert store.claim_next().id == second
    assert store.claim_next() is None
    assert store.task(first).status == RUNNING


def test_claim_respects_kinds(store):
    store.add_task("apply", kind="job_application")
    assert store.claim_next(frozenset({"agent"})) is None
    assert store.claim_next(frozenset({"job_application"})).kind == "job_application"


def test_cancel_queued_task_and_retry(store):
    task_id = store.add_task("one")
    assert store.request_cancel(task_id)
    assert store.task(task_id).status == CANCELLED
    assert store.claim_next() is None
    assert store.retry(task_id)
    assert store.task(task_id).status == QUEUED


def test_interrupted_tasks_are_not_rerun(store):
    task_id = store.add_task("one")
    store.claim_next()
    assert store.recover_interrupted() == [task_id]
    assert store.task(task_id).status == INTERRUPTED
    assert store.claim_next() is None


def test_answers_are_recorded_once(store):
    approval = store.add_approval(
        task_id=None, tool="t", permission="SEND", digest="d", summary="s"
    )
    assert [p.id for p in store.pending_approvals()] == [approval]
    assert store.answer(approval, True)
    assert not store.answer(approval, False)
    assert store.approval(approval).status == "approved"
    assert store.pending_approvals() == []


def test_expired_approvals_cannot_be_answered(store):
    approval = store.add_approval(
        task_id=None,
        tool="t",
        permission="SEND",
        digest="d",
        summary="s",
        lifetime=timedelta(seconds=-1),
    )
    assert not store.answer(approval, True)
    assert store.approval(approval).status == "expired"


def answer_when_asked(store, approved, results):
    def run():
        for _ in range(500):
            pending = store.pending_approvals()
            if pending:
                results.append(pending[0])
                store.answer(pending[0].id, approved)
                return
            threading.Event().wait(0.01)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


@pytest.mark.parametrize("approved", [True, False])
def test_deferred_approval_waits_for_the_answer(store, approved):
    task_id = store.add_task("send it")
    store.claim_next()
    notified, seen = [], []
    provider = DeferredApprovalProvider(store, task_id, notify=notified.append, poll_seconds=0.01)
    thread = answer_when_asked(store, approved, seen)
    assert provider.request(request()) is approved
    thread.join()
    assert len(notified) == 1
    assert "SEND the Gmail draft" in seen[0].summary
    assert store.task(task_id).status == RUNNING


def test_task_waits_while_approval_is_pending(store):
    task_id = store.add_task("send it")
    store.claim_next()
    statuses = []

    def sleep(_):
        statuses.append(store.task(task_id).status)
        store.answer(store.pending_approvals()[0].id, True)

    provider = DeferredApprovalProvider(store, task_id, sleep=sleep)
    assert provider.request(request())
    assert statuses == [WAITING]


def test_grants_skip_the_question_only_for_the_granted_tool(store):
    exact = Grant("gmail_send_draft", arguments_digest("gmail_send_draft", {"draft_id": "r-1"}))
    provider = DeferredApprovalProvider(store, None, grants=(exact,), sleep=lambda _: None)
    assert provider.request(request())
    assert store.pending_approvals() == []

    any_args = DeferredApprovalProvider(store, None, grants=(Grant("job_submit"),))
    assert any_args.request(request("job_submit", job="1"))

    token = CancellationToken()
    token.cancel()
    other = DeferredApprovalProvider(store, None, grants=(exact,), cancel=token)
    assert not other.request(request(draft_id="r-2"))  # different arguments: asked, cancelled


def test_cancelling_the_task_denies_its_approval(store):
    task_id = store.add_task("send it")
    store.claim_next()

    def sleep(_):
        store.request_cancel(task_id)

    provider = DeferredApprovalProvider(store, task_id, sleep=sleep)
    assert not provider.request(request())


def test_worker_runs_tasks_in_parallel_and_records_results(store):
    store.add_task("a")
    store.add_task("b")
    store.add_task("boom")
    running = []
    peak = []

    async def runner(task, context):
        running.append(task.id)
        peak.append(len(running))
        await asyncio.sleep(0.01)
        running.remove(task.id)
        if task.prompt == "boom":
            raise RuntimeError("it broke")
        context.progress("halfway")
        return f"did {task.prompt}"

    sent = []

    class Notes:
        def send(self, title, body):
            sent.append(body)

    worker = BackgroundWorker(
        store, {"agent": runner}, notifier=Notes(), max_parallel=2, poll_seconds=0.01
    )
    asyncio.run(worker.drain())
    tasks = {task.prompt: task for task in store.tasks()}
    assert tasks["a"].status == DONE and tasks["a"].result == "did a"
    assert tasks["a"].progress == "halfway"
    assert tasks["boom"].status == FAILED and tasks["boom"].error == "it broke"
    assert max(peak) == 2
    assert any("Could not finish" in body for body in sent)


def test_worker_cancels_a_running_task(store):
    task_id = store.add_task("long")

    async def runner(task, context):
        store.request_cancel(task.id)
        while not context.cancel.cancelled:
            await asyncio.sleep(0.01)
            worker.check_cancellations()
        from assistant.cancellation import TurnCancelled

        raise TurnCancelled

    worker = BackgroundWorker(store, {"agent": runner}, poll_seconds=0.01)
    asyncio.run(worker.drain())
    assert store.task(task_id).status == CANCELLED


def test_only_one_process_holds_the_worker_lock(tmp_path):
    first = acquire_worker_lock(tmp_path / "background.lock")
    assert first is not None
    # flock locks belong to the open file, so a second open in this process conflicts too.
    assert acquire_worker_lock(tmp_path / "background.lock") is None
    first.close()
    assert acquire_worker_lock(tmp_path / "background.lock") is not None
