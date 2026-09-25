"""Cooperative turn cancellation shared by the REPL, runtime, agent, model, and tools."""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class TurnCancelled(Exception):
    """The user cancelled the current turn; the work stopped before completing."""

    def __init__(self) -> None:
        super().__init__("Cancelled.")


class CancellationToken:
    """A thread-safe flag set by the UI and checked at safe points by the work."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TurnCancelled


_current: ContextVar[CancellationToken | None] = ContextVar("jamie_cancellation", default=None)


def current_token() -> CancellationToken | None:
    """Return the active turn's token; None means the work cannot be cancelled."""
    return _current.get()


def check_cancelled() -> None:
    token = _current.get()
    if token is not None:
        token.raise_if_cancelled()


@contextmanager
def cancellation_scope(token: CancellationToken | None) -> Iterator[None]:
    """Expose a token to nested model and tool code (asyncio.to_thread copies it)."""
    reset = _current.set(token)
    try:
        yield
    finally:
        _current.reset(reset)


@contextmanager
def interrupt_cancels(token: CancellationToken) -> Iterator[None]:
    """While work runs on this event loop, Ctrl+C cancels it instead of stopping Jamie.

    At the prompt, prompt_toolkit reads Ctrl+C as a key, so no signal is involved.
    """
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    try:
        loop.add_signal_handler(signal.SIGINT, token.cancel)
    except (NotImplementedError, RuntimeError, ValueError):  # not the main thread
        yield
        return
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous)
