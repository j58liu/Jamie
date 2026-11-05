"""The background service: runs queued tasks and job applications for every Jamie front end.

Jamie.app runs it inside the window process; `jamie serve` runs it on its own (for
example from a LaunchAgent). Only one process at a time holds the worker lock.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant.background import (
    BackgroundStore,
    BackgroundWorker,
    acquire_worker_lock,
    agent_runner,
)
from assistant.background.worker import Notifier, PrintNotifier
from assistant.config import ProjectConfig


def worker_lock_path(store: BackgroundStore) -> Path:
    return store.path.with_name("background.lock")


def build_runners(
    application,
    settings: ProjectConfig,
    *,
    web,
    backend,
    workspace: Path,
    max_tokens: int,
    browser=None,
) -> dict[str, Any]:
    """Runners by task kind; each builds fresh tools whose approvals wait for the user."""
    from assistant.tools import create_agent_registry

    def registry(approvals):
        return create_agent_registry(
            workspace,
            approvals=approvals,
            web=web,
            backend=backend,
            connectors=application.connectors,
            files=settings.files.paths(),
            browser=browser,
        )

    return {"agent": agent_runner(application, registry, max_tokens=max_tokens)}


async def run_background_service(
    application,
    settings: ProjectConfig,
    *,
    web,
    backend,
    store: BackgroundStore | None = None,
    notifier: Notifier | None = None,
    workspace: Path | None = None,
    max_tokens: int = 1024,
    browser=None,
    on_ready: Callable[[bool], object] | None = None,
) -> None:
    """Serve until cancelled; without the worker lock it returns at once (another serves)."""
    store = store or BackgroundStore()
    lock = acquire_worker_lock(worker_lock_path(store))
    if on_ready is not None:
        on_ready(lock is not None)
    if lock is None:
        return
    try:
        runners = build_runners(
            application,
            settings,
            web=web,
            backend=backend,
            workspace=workspace or Path.home(),
            max_tokens=max_tokens,
            browser=browser,
        )
        worker = BackgroundWorker(
            store,
            runners,
            notifier=notifier or PrintNotifier(),
            max_parallel=settings.background.max_parallel,
        )
        await worker.run_forever()
    finally:
        with contextlib.suppress(Exception):
            lock.close()


async def serve(settings: ProjectConfig, *, notifier: Notifier | None = None) -> None:
    """`jamie serve`: own a model connection and run background work until interrupted."""
    from assistant.llm.llamacpp import LlamaCppBackend
    from assistant.runtime import JamieApplication
    from assistant.tools import WebClient

    backend = LlamaCppBackend()
    async with JamieApplication(
        settings,
        backend_factory=lambda: backend,
        extract_memories=False,
    ) as application:
        browser = settings.browser.session()
        with WebClient() as web:

            def ready(owned: bool) -> None:
                if not owned:
                    print("Another Jamie process is already running background tasks.")
                else:
                    print("Jamie is running background tasks. Press Ctrl+C to stop.")

            try:
                await run_background_service(
                    application,
                    settings,
                    web=web,
                    backend=backend,
                    notifier=notifier,
                    browser=browser,
                    on_ready=ready,
                )
            finally:
                if browser is not None:
                    browser.close()


def run_serve(settings: ProjectConfig) -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(settings))
