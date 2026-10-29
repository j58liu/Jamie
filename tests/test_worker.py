import os
import plistlib
from datetime import UTC, datetime
from pathlib import Path

from assistant.tasks.notifications import MacNotifier, RecordingNotifier
from assistant.tasks.store import TaskStore
from assistant.tasks.worker import (
    Worker,
    install_launch_agent,
    launch_agent_plist,
    uninstall_launch_agent,
)
from assistant.tools import CommandResult


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append(arguments)
        return CommandResult(stdout="", stderr="")


def make_worker(tmp_path, *, available=True):
    store = TaskStore(tmp_path / "tasks.sqlite3", clock=lambda: datetime.now(UTC))
    calls = {"engine": 0, "probe": 0}
    notifier = RecordingNotifier()

    class Engine:
        def run_all(self):
            return []

    def engine_factory():
        calls["engine"] += 1
        return Engine()

    def probe():
        calls["probe"] += 1
        return available

    worker = Worker(store, engine_factory=engine_factory, model_available=probe, notifier=notifier)
    return store, worker, calls, notifier


def test_idle_worker_never_touches_the_model(tmp_path):
    _, worker, calls, _ = make_worker(tmp_path)
    worker.run_once()
    assert calls == {"engine": 0, "probe": 0}


def test_ready_task_runs_when_model_is_up(tmp_path):
    store, worker, calls, _ = make_worker(tmp_path)
    store.create_task("t", "x")
    worker.run_once()
    assert calls == {"engine": 1, "probe": 1}


def test_model_down_notifies_once(tmp_path):
    store, worker, calls, notifier = make_worker(tmp_path, available=False)
    store.create_task("t", "x")
    assert worker.run_once().model_unavailable
    worker.run_once()
    assert len(notifier.messages) == 1 and calls["engine"] == 0


def test_launch_agent_plist_runs_one_pass_every_five_minutes(tmp_path):
    data = plistlib.loads(
        launch_agent_plist(
            working_directory=tmp_path,
            log_directory=tmp_path / "logs",
            config=tmp_path / "assistant.toml",
        )
    )
    assert data["Label"] == "com.jamie.worker"
    assert data["ProgramArguments"][-5:] == [
        "worker",
        "run",
        "--once",
        "--config",
        str(tmp_path / "assistant.toml"),
    ]
    assert data["StartInterval"] == 300 and data["RunAtLoad"] is True


def test_install_and_uninstall_use_launchctl(tmp_path):
    runner = FakeRunner()
    path = tmp_path / "agents" / "com.jamie.worker.plist"
    install_launch_agent(
        working_directory=tmp_path, log_directory=tmp_path / "logs", runner=runner, path=path
    )
    domain = f"gui/{os.getuid()}"
    assert runner.calls == [
        ["launchctl", "bootout", domain, str(path)],
        ["launchctl", "bootstrap", domain, str(path)],
    ]
    assert path.exists()
    assert uninstall_launch_agent(runner=runner, path=path)
    assert not path.exists() and not uninstall_launch_agent(runner=runner, path=Path(path))


def test_mac_notification_escapes_text():
    runner = FakeRunner()
    MacNotifier(runner).notify('Say "hi"', "line\none \\ two")
    [arguments] = runner.calls
    assert arguments[:2] == ["osascript", "-e"]
    assert '\\"hi\\"' in arguments[2] and "\n" not in arguments[2]
