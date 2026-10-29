"""The background worker and its launchd agent.

``jamie worker --once`` is cheap when nothing is due: it checks schedules, recovers
interrupted work, and expires stale approvals using SQLite only. It contacts the
local model only when a task is ready to run.
"""

from __future__ import annotations

import os
import plistlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from assistant.tasks.approvals import expire_due
from assistant.tasks.notifications import Notifier, NullNotifier
from assistant.tasks.recovery import RecoveryReport, recover
from assistant.tasks.scheduler import TickReport, tick
from assistant.tasks.store import TaskStore
from assistant.tools.base import ToolError
from assistant.tools.macos import CommandRunner, SubprocessCommandRunner

LABEL = "com.jamie.worker"
INTERVAL_SECONDS = 300
MODEL_DOWN_FLAG = "model_down_notified"


@dataclass
class WorkerReport:
    tick: TickReport = field(default_factory=TickReport)
    recovery: RecoveryReport = field(default_factory=RecoveryReport)
    expired: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    model_unavailable: bool = False


class Worker:
    def __init__(
        self,
        store: TaskStore,
        *,
        engine_factory: Callable[[], object],
        model_available: Callable[[], bool],
        connectors=None,
        notifier: Notifier | None = None,
    ) -> None:
        self._store = store
        self._engine_factory = engine_factory
        self._model_available = model_available
        self._connectors = connectors
        self._notifier = notifier if notifier is not None else NullNotifier()

    def run_once(self) -> WorkerReport:
        report = WorkerReport()
        report.tick = tick(self._store, self._store.now(), self._connectors)
        report.recovery = recover(self._store, self._connectors)
        report.expired = expire_due(self._store)
        if self._store.count_runnable() == 0:
            return report
        if not self._model_available():
            report.model_unavailable = True
            if self._store.get_flag(MODEL_DOWN_FLAG) is None:
                self._notifier.notify(
                    "Jamie has tasks waiting",
                    "The local model is not running. Start it with scripts/start-local-model.sh.",
                )
                self._store.set_flag(MODEL_DOWN_FLAG, "1")
            return report
        self._store.set_flag(MODEL_DOWN_FLAG, None)
        engine = self._engine_factory()
        report.ran = [task.id for task in engine.run_all()]
        return report


# -- launchd ----------------------------------------------------------------------------


def launch_agent_path() -> Path:
    return Path("~/Library/LaunchAgents").expanduser() / f"{LABEL}.plist"


def launch_agent_plist(
    *, working_directory: Path, log_directory: Path, config: Path | None = None
) -> bytes:
    arguments = [sys.executable, "-m", "assistant", "worker", "run", "--once"]
    if config is not None:
        arguments += ["--config", str(config)]
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": arguments,
            "WorkingDirectory": str(working_directory),
            "StartInterval": INTERVAL_SECONDS,
            "RunAtLoad": True,
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "StandardOutPath": str(log_directory / "worker.log"),
            "StandardErrorPath": str(log_directory / "worker.log"),
        }
    )


def install_launch_agent(
    *,
    working_directory: Path,
    log_directory: Path,
    config: Path | None = None,
    runner: CommandRunner | None = None,
    path: Path | None = None,
) -> Path:
    runner = runner if runner is not None else SubprocessCommandRunner()
    path = path if path is not None else launch_agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    try:  # replace an older copy if it is loaded
        runner.run(["launchctl", "bootout", domain, str(path)], timeout=15)
    except ToolError:
        pass
    path.write_bytes(
        launch_agent_plist(
            working_directory=working_directory, log_directory=log_directory, config=config
        )
    )
    runner.run(["launchctl", "bootstrap", domain, str(path)], timeout=15)
    return path


def uninstall_launch_agent(
    *, runner: CommandRunner | None = None, path: Path | None = None
) -> bool:
    runner = runner if runner is not None else SubprocessCommandRunner()
    path = path if path is not None else launch_agent_path()
    if not path.exists():
        return False
    try:
        runner.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], timeout=15)
    except ToolError:
        pass
    path.unlink()
    return True
