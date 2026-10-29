"""Read-only prerequisite checks for `jamie doctor`; nothing here changes the system."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from assistant.config import ConfigurationError, ProjectConfig, load_config
from assistant.llm.base import LLMError, ServerInfo
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.ui import login_item
from assistant.ui.status import short_model

MINIMUM_PYTHON = (3, 12)
MACOS_COMMANDS = ("open", "osascript", "shortcuts", "pbcopy", "pbpaste")
# What macos/build-app.sh records in Jamie.app when built from this environment.
ENVIRONMENT_PYTHON = str(Path(sys.prefix) / "bin" / "python")


class Status(StrEnum):
    PASS = "pass"
    WARN = "warn"  # a feature is limited, but Jamie can run
    SKIP = "skip"  # not applicable, or blocked by an earlier failure
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _home(path: Path | str) -> str:
    text = str(path)
    home = str(Path.home())
    return f"~{text[len(home) :]}" if text == home or text.startswith(f"{home}/") else text


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1_000_000_000:.1f} GB"


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = tuple(sys.version_info[:2]) >= MINIMUM_PYTHON
    return Check("Python", Status.PASS if ok else Status.FAIL, f"{version} (requires 3.12+)")


def check_configuration(path: Path | None) -> tuple[Check, ProjectConfig | None]:
    try:
        settings = load_config(path)
    except ConfigurationError as exc:
        return Check("Jamie configuration", Status.FAIL, str(exc)), None
    source = path or Path("assistant.toml")
    detail = f"{source} is valid" if source.exists() else "built-in defaults (no assistant.toml)"
    return Check("Jamie configuration", Status.PASS, detail), settings


def check_backend(backend: LlamaCppBackend) -> tuple[Check, ServerInfo | None]:
    try:
        info = backend.server_info()
    except LLMError as exc:
        return Check("Inference backend", Status.FAIL, str(exc)), None
    detail = (
        f"llama.cpp at {backend.server_url} · {short_model(info.model)} · "
        f"{info.context:,}-token context"
    )
    return Check("Inference backend", Status.PASS, detail), info


def check_model_files(info: ServerInfo | None) -> Check:
    if info is None:
        return Check("Model files", Status.SKIP, "needs a running inference backend")
    if info.model_path is None:
        return Check("Model files", Status.WARN, "llama.cpp did not report a model path")
    path = Path(info.model_path)
    if not path.is_file():
        return Check(
            "Model files", Status.WARN, f"{info.model_path} is not visible here (remote server?)"
        )
    return Check("Model files", Status.PASS, f"{path.name} ({_size(path)})")


def check_data_directory(settings: ProjectConfig | None) -> Check:
    if settings is None:
        return Check("Application data directory", Status.SKIP, "needs a valid configuration")
    directory = settings.resolved_memory_path().parent
    existing = directory
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    writable = existing.is_dir() and os.access(existing, os.W_OK | os.X_OK)
    if not writable:
        return Check(
            "Application data directory", Status.FAIL, f"{_home(existing)} is not writable"
        )
    if existing != directory:
        return Check(
            "Application data directory", Status.PASS, f"{_home(directory)} (created on first use)"
        )
    return Check("Application data directory", Status.PASS, f"{_home(directory)} (writable)")


def check_browser_runtime() -> Check:
    return Check("Browser runtime", Status.SKIP, "not used; web tools fetch pages over HTTP")


def check_connectors(settings: ProjectConfig | None) -> list[Check]:
    """Report each configured personal-app connector; nothing is contacted."""
    if settings is None:
        return [Check("Personal connectors", Status.SKIP, "needs a valid configuration")]
    from assistant.connectors import ConnectionStatus, build_connectors

    connectors = build_connectors(settings).all()
    if not connectors:
        return [Check("Personal connectors", Status.SKIP, "none configured in assistant.toml")]
    checks = []
    for connector in connectors:
        try:
            info = connector.info()
        except Exception as exc:  # a status check must never break doctor
            checks.append(Check(f"Connector {connector.name}", Status.WARN, str(exc)))
            continue
        ok = info.status is ConnectionStatus.CONNECTED
        detail = info.status.value.replace("_", " ")
        if info.capabilities:
            detail += f" ({', '.join(sorted(c.value for c in info.capabilities))})"
        if info.detail:
            detail += f"; {info.detail}"
        checks.append(Check(f"Connector {info.name}", Status.PASS if ok else Status.WARN, detail))
    return checks


def check_macos() -> Check:
    if sys.platform != "darwin":
        return Check("macOS tools", Status.WARN, f"unavailable on {sys.platform}")
    missing = [command for command in MACOS_COMMANDS if shutil.which(command) is None]
    if missing:
        return Check("macOS tools", Status.WARN, f"missing: {', '.join(missing)}")
    return Check(
        "macOS tools",
        Status.PASS,
        "commands found; macOS asks for Automation access the first time a tool needs it",
    )


def check_desktop_app() -> Check:
    app = login_item.installed_app()
    if app is None:
        return Check("Desktop app", Status.SKIP, "not built; run macos/build-app.sh")
    built_for = (login_item.app_info(app) or {}).get("JamiePython")
    if built_for != ENVIRONMENT_PYTHON:
        source = built_for or "an older build"
        detail = f"{_home(app)} runs {source}; rebuild with macos/build-app.sh"
        return Check("Desktop app", Status.WARN, detail)
    login = "opens at login" if login_item.opens_at_login() else "not at login (/startup on)"
    return Check("Desktop app", Status.PASS, f"{_home(app)} · {login}")


def run_checks(config: Path | None, backend: LlamaCppBackend) -> list[Check]:
    configuration, settings = check_configuration(config)
    inference, info = check_backend(backend)
    return [
        check_python(),
        configuration,
        check_model_files(info),
        inference,
        check_data_directory(settings),
        check_browser_runtime(),
        check_macos(),
        check_desktop_app(),
        *check_connectors(settings),
    ]
