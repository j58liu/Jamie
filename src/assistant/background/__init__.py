"""Background tasks: queue work, keep working, approve what needs approval later."""

from assistant.background.approvals import DeferredApprovalProvider, Grant
from assistant.background.store import (
    BackgroundStore,
    BackgroundStoreError,
    BackgroundTask,
    PendingApproval,
    arguments_digest,
    default_background_path,
)
from assistant.background.worker import (
    BackgroundWorker,
    FanOutNotifier,
    PrintNotifier,
    TaskContext,
    TaskNeedsInput,
    acquire_worker_lock,
    agent_runner,
)

__all__ = [
    "BackgroundStore",
    "BackgroundStoreError",
    "BackgroundTask",
    "BackgroundWorker",
    "DeferredApprovalProvider",
    "FanOutNotifier",
    "Grant",
    "PendingApproval",
    "PrintNotifier",
    "TaskContext",
    "TaskNeedsInput",
    "acquire_worker_lock",
    "agent_runner",
    "arguments_digest",
    "default_background_path",
]
