import plistlib
import threading
from pathlib import Path

from typer.testing import CliRunner

from assistant import cli, doctor
from assistant.doctor import Status, check_desktop_app
from assistant.ui import login_item
from assistant.ui.gui.background import hide_instead_of_closing


def build_app(path: Path, **info) -> Path:
    contents = path / "Contents"
    contents.mkdir(parents=True)
    info = {"CFBundleIdentifier": login_item.BUNDLE_ID, **info}
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    return path


def test_only_a_jamie_bundle_counts_as_installed(isolated_app_install):
    app = login_item.APP_LOCATIONS[0]
    assert login_item.installed_app() is None

    build_app(app, CFBundleIdentifier="com.example.other")
    assert login_item.installed_app() is None

    (app / "Contents" / "Info.plist").unlink()
    (app / "Contents").rmdir()
    build_app(app)
    assert login_item.installed_app() == app


def test_login_item_opens_the_app_hidden_and_can_be_removed():
    app = build_app(login_item.APP_LOCATIONS[0])

    login_item.open_at_login(app)
    agent = plistlib.loads(login_item.LOGIN_AGENT.read_bytes())
    assert agent["ProgramArguments"] == [
        "/usr/bin/open",
        "-g",
        "-a",
        str(app),
        "--args",
        "--hidden",
    ]
    assert agent["RunAtLoad"] is True and login_item.opens_at_login()

    login_item.stop_opening_at_login()
    login_item.stop_opening_at_login()  # already off
    assert not login_item.opens_at_login()


def run_command(line: str) -> str:
    from io import StringIO

    from rich.console import Console

    from assistant.ui.commands import ReplState, create_command_registry

    output = StringIO()
    state = ReplState(Console(file=output, width=200))
    create_command_registry().dispatch(state, line)
    return output.getvalue()


def test_startup_command_needs_the_app_and_reports_the_setting():
    assert "Build Jamie.app first" in run_command("/startup on")
    assert not login_item.opens_at_login()

    build_app(login_item.APP_LOCATIONS[0])
    assert "Open at login: on" in run_command("/startup on")
    assert login_item.opens_at_login()
    assert "Open at login: on" in run_command("/startup")
    assert "Open at login: off" in run_command("/startup off")
    assert "Usage: /startup on|off" in run_command("/startup maybe")


def test_doctor_reports_a_missing_stale_or_current_app(monkeypatch):
    assert check_desktop_app().status is Status.SKIP

    app = build_app(login_item.APP_LOCATIONS[0])
    stale = check_desktop_app()
    assert stale.status is Status.WARN and "an older build" in stale.detail

    (app / "Contents" / "Info.plist").write_bytes(
        plistlib.dumps(
            {"CFBundleIdentifier": login_item.BUNDLE_ID, "JamiePython": "/moved/.venv/bin/python"}
        )
    )
    assert "runs /moved/.venv/bin/python" in check_desktop_app().detail

    monkeypatch.setattr(doctor, "ENVIRONMENT_PYTHON", "/moved/.venv/bin/python")
    current = check_desktop_app()
    assert current.status is Status.PASS and "not at login" in current.detail
    login_item.open_at_login(app)
    assert "opens at login" in check_desktop_app().detail


class FakeEvent:
    """Like pywebview's Event: ``+=`` adds a handler."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class FakeWindow:
    def __init__(self):
        self.hidden = 0
        self.events = type("Events", (), {"closing": FakeEvent()})()

    def hide(self):
        self.hidden += 1


def test_closing_hides_the_window_until_jamie_quits():
    window, quitting = FakeWindow(), threading.Event()
    hide_instead_of_closing(window, quitting)
    (closing,) = window.events.closing.handlers

    assert closing() is False and window.hidden == 1  # pywebview keeps the window
    quitting.set()
    assert closing() is True and window.hidden == 1


def test_gui_command_can_start_hidden(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_gui", lambda settings, **options: calls.append(options))

    assert CliRunner().invoke(cli.app, ["gui", "--hidden"]).exit_code == 0
    assert CliRunner().invoke(cli.app, ["gui"]).exit_code == 0
    assert calls == [{"hidden": True}, {"hidden": False}]
