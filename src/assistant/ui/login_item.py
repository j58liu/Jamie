"""Find the installed Jamie.app and open it at login with a user LaunchAgent."""

from __future__ import annotations

import plistlib
from pathlib import Path

BUNDLE_ID = "me.jameshliu.jamie"
APP_LOCATIONS = (Path.home() / "Applications" / "Jamie.app", Path("/Applications/Jamie.app"))
# Loaded by launchd at the next login; writing it starts nothing now.
LOGIN_AGENT = Path.home() / "Library" / "LaunchAgents" / f"{BUNDLE_ID}.plist"


def app_info(app: Path) -> dict | None:
    """The bundle's Info.plist when it is a Jamie build, else None."""
    try:
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    return info if info.get("CFBundleIdentifier") == BUNDLE_ID else None


def installed_app() -> Path | None:
    return next((app for app in APP_LOCATIONS if app_info(app) is not None), None)


def opens_at_login() -> bool:
    return LOGIN_AGENT.is_file()


def open_at_login(app: Path) -> None:
    """Open the app in the background, without its window, each time you log in."""
    LOGIN_AGENT.parent.mkdir(parents=True, exist_ok=True)
    agent = {
        "Label": BUNDLE_ID,
        "ProgramArguments": ["/usr/bin/open", "-g", "-a", str(app), "--args", "--hidden"],
        "RunAtLoad": True,
        "LimitLoadToSessionType": "Aqua",
    }
    LOGIN_AGENT.write_bytes(plistlib.dumps(agent))


def stop_opening_at_login() -> None:
    LOGIN_AGENT.unlink(missing_ok=True)
