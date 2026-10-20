"""A desktop window over the shared runtime; the page only receives display events."""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Coroutine
from contextlib import ExitStack
from functools import cache
from pathlib import Path
from typing import Any

from assistant.agent.events import (
    AgentEvent,
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from assistant.agent.loop import ToolExecutor
from assistant.cancellation import CancellationToken, TurnCancelled
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, Message, ServerInfo
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.runtime import JamieApplication
from assistant.ui.renderer import error_lines, tool_summary
from assistant.ui.status import short_model

HERE = Path(__file__).parent
FONT_WEIGHTS = (400, 500, 600)
ICON = HERE / "icon.png"
# The window is launched from the repository checkout, which holds the start script.
MODEL_SCRIPT = HERE.parents[3] / "scripts" / "start-local-model.sh"
MODEL_LOG = Path.home() / "Library" / "Logs" / "Jamie" / "llama-server.log"
MODEL_START_SECONDS = 900.0  # a first, uncached load can take minutes
# Apps opened from Finder get a minimal PATH without Homebrew.
EXTRA_PATH = ("/opt/homebrew/bin", "/usr/local/bin")

type Payload = dict[str, Any]
type Submit = Callable[[Coroutine[Any, Any, None]], object]


class StartupFailed(RuntimeError):
    """The window cannot reach a model server; the message says what to do."""


def event_payload(event: AgentEvent, args: dict[str, Any] | None = None) -> Payload | None:
    """What the page may show: tool lines and answer text, never tool output or JSON."""
    match event:
        case ToolStarted(args=started):
            return {"type": "tool", "state": "running", "text": tool_summary(event, started)}
        case ToolFinished():
            return {"type": "tool", "state": "done", "text": tool_summary(event, args)}
        case ToolFailed(error=error):
            text = tool_summary(event, args)
            return {"type": "tool", "state": "failed", "text": text, "detail": error_lines(error)}
        case AssistantDelta(text=text):
            return {"type": "delta", "text": text}
        case AssistantFinished(text=text, finish_reason=reason):
            return {"type": "answer", "text": text, "truncated": reason == "length"}
    return None  # failures arrive as the turn's exception; approvals come in phase 2


class GuiSession:
    """One conversation in the window: one turn at a time, cancellable, kept in memory.

    Page calls arrive on pywebview threads; ``submit`` runs a turn on the runtime loop.
    """

    def __init__(
        self,
        application: JamieApplication,
        tools: ToolExecutor,
        push: Callable[[Payload], object],
        submit: Submit,
    ) -> None:
        self._application = application
        self._tools = tools
        self._push = push
        self._submit = submit
        self._lock = threading.Lock()
        self._token: CancellationToken | None = None
        self._started: tuple[str, dict[str, Any]] | None = None
        self.history: list[Message] = []
        self.conversation_id: str | None = None

    def send(self, prompt: str) -> bool:
        """Start a turn; False when the prompt is blank or a turn is already running."""
        prompt = prompt.strip()
        with self._lock:
            if not prompt or self._token is not None:
                return False
            token = self._token = CancellationToken()
        self._submit(self._turn(prompt, token))
        return True

    def cancel(self) -> None:
        with self._lock:
            if self._token is not None:
                self._token.cancel()

    def new_conversation(self) -> bool:
        with self._lock:
            if self._token is not None:
                return False
            self.history = []
            self.conversation_id = None
            return True

    async def _turn(self, prompt: str, token: CancellationToken) -> None:
        try:
            if self.conversation_id is None:
                self.conversation_id = self._application.start_conversation()
            result = await self._application.run_turn(
                prompt,
                history=self.history,
                thinking=False,
                cancel=token,
                conversation_id=self.conversation_id,
                event_sink=self._show,
                tools=self._tools,
            )
        except TurnCancelled:
            self._send({"type": "cancelled"})
        except Exception as exc:  # noqa: BLE001 - the window must return to its prompt
            self._send({"type": "error", "lines": error_lines(str(exc) or type(exc).__name__)})
        else:
            if result.compacted_history is not None:
                self.history = list(result.compacted_history)
            self.history += [Message("user", prompt), Message("assistant", result.message)]
        finally:
            with self._lock:
                self._token = None
            self._send({"type": "idle"})

    def _show(self, event: AgentEvent) -> None:
        if isinstance(event, ToolStarted):
            self._started = (event.tool, event.args)
        tool = getattr(event, "tool", None)
        args = self._started[1] if self._started and self._started[0] == tool else None
        payload = event_payload(event, args)
        if payload is not None:
            self._send(payload)

    def _send(self, payload: Payload) -> None:
        try:
            self._push(payload)
        except Exception:  # noqa: BLE001 - a closed window must not fail the turn
            pass


def start_model_server() -> subprocess.Popen:
    """Start llama.cpp in its own session so it keeps running after the window closes."""
    if not MODEL_SCRIPT.is_file():
        raise StartupFailed("Start the model server with scripts/start-local-model.sh.")
    MODEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    path = os.pathsep.join([*EXTRA_PATH, os.environ.get("PATH", "/usr/bin:/bin")])
    with MODEL_LOG.open("ab") as log:
        return subprocess.Popen(
            [str(MODEL_SCRIPT)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PATH": path},
            start_new_session=True,
        )


def _last_log_line() -> str:
    try:
        lines = MODEL_LOG.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1] if lines else ""


def ensure_server(
    backend: LlamaCppBackend,
    status: Callable[[str], object],
    *,
    start: Callable[[], subprocess.Popen] = start_model_server,
    timeout: float = MODEL_START_SECONDS,
    interval: float = 2.0,
) -> ServerInfo:
    """Use a running server, or start one and wait until its model has loaded."""
    try:
        return backend.server_info()
    except LLMError:
        pass
    status("Starting the local model…")
    process = start()
    deadline = time.monotonic() + timeout
    while True:
        try:
            return backend.server_info()
        except LLMError as exc:
            if process.poll() is not None:
                reason = _last_log_line() or "it exited during startup"
                raise StartupFailed(f"The model server stopped: {reason}\nSee {MODEL_LOG}") from exc
            if time.monotonic() >= deadline:
                raise StartupFailed(f"The model did not load in time. See {MODEL_LOG}") from exc
        time.sleep(interval)


def _home(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}".removesuffix("/.")
    except ValueError:
        return str(path)


@cache
def page_html(workspace: str = "") -> str:
    """The page with fonts inlined, so the window loads nothing from disk or network."""
    fonts = "".join(
        "@font-face{font-family:'DM Sans';font-style:normal;font-display:swap;"
        f"font-weight:{weight};src:url(data:font/woff2;base64,"
        f"{base64.b64encode((HERE / 'fonts' / f'dm-sans-{weight}.woff2').read_bytes()).decode()})"
        " format('woff2');}"
        for weight in FONT_WEIGHTS
    )
    return (
        (HERE / "index.html")
        .read_text(encoding="utf-8")
        .replace("/* fonts */", fonts)
        .replace("{{workspace}}", html.escape(workspace))
    )


class _RuntimeThread:
    """The asyncio loop that owns the application, beside the window's main thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, name="jamie-gui-runtime", daemon=True
        )
        self._thread.start()

    def submit(self, coroutine: Coroutine[Any, Any, Any]):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def run(self, coroutine: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        return self.submit(coroutine).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def run_gui(settings: ProjectConfig) -> None:
    """Show the window until it closes, working in the current directory."""
    from Foundation import NSBundle

    # Outside Jamie.app the process is Python; name the menu before AppKit reads it.
    NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "Jamie"
    import webview  # loaded only here, so the terminal never pays for AppKit

    from assistant.tools import WebClient, create_agent_registry

    runtime = _RuntimeThread()
    window = webview.create_window(
        "Jamie",
        html=page_html(_home(Path.cwd())),
        width=760,
        height=820,
        min_size=(420, 480),
        text_select=True,
    )
    closed = threading.Event()  # evaluate_js on a destroyed window can block the loop
    window.events.closed += closed.set

    def push(payload: Payload) -> None:
        if not closed.is_set():
            window.evaluate_js(f"jamie.receive({json.dumps(payload)})")

    sessions: list[GuiSession] = []  # filled once the model is reachable

    def send(prompt: str) -> bool:
        return bool(sessions) and sessions[0].send(prompt)

    def cancel() -> None:
        if sessions:
            sessions[0].cancel()

    def new_conversation() -> bool:
        return bool(sessions) and sessions[0].new_conversation()

    window.expose(send, cancel, new_conversation)
    resources = ExitStack()
    application: JamieApplication | None = None

    async def start() -> None:
        nonlocal application
        backend = LlamaCppBackend()
        resources.callback(backend.close)  # closing twice is safe

        def status(text: str) -> None:
            push({"type": "status", "text": text})

        info = await asyncio.to_thread(ensure_server, backend, status)
        application = JamieApplication(settings, backend_factory=lambda: backend)
        await application.start()
        web = resources.enter_context(WebClient())
        tools = create_agent_registry(Path.cwd(), web=web, backend=backend)
        resources.callback(tools.close)
        sessions.append(GuiSession(application, tools, push, runtime.submit))
        push({"type": "ready", "model": short_model(info.model)})

    def startup() -> None:
        window.events.loaded.wait()
        try:
            runtime.run(start())
        except Exception as exc:  # noqa: BLE001 - shown in the window, which stays open
            push({"type": "fatal", "lines": error_lines(str(exc) or type(exc).__name__)})

    try:
        webview.start(startup, icon=str(ICON))
    finally:
        closed.set()
        cancel()
        try:
            if application is not None:
                # Cancellation is cooperative: wait briefly for a running tool, then close.
                runtime.run(application.close(), timeout=10)
        finally:
            resources.close()
            runtime.stop()
