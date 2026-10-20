"""A desktop window over the shared runtime; the page only receives display events."""

from __future__ import annotations

import asyncio
import base64
import html
import inspect
import io
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

from rich.console import Console

from assistant.agent.events import (
    AgentEvent,
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from assistant.cancellation import CancellationToken, TurnCancelled, current_token
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, ServerInfo
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.runtime import JamieApplication
from assistant.runtime.workspaces import WorkspaceTools
from assistant.security.approvals import ApprovalRequest
from assistant.ui.approvals import summarize_approval
from assistant.ui.commands import ReplState, create_command_registry
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
COMMAND_WIDTH = 88  # command output is shown in a monospace block
COMMANDS = create_command_registry()

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
    return None  # failures arrive as the turn's exception; approvals ask separately


class GuiApprovalProvider:
    """Ask in the window. Cancelling the turn, closing the window, or a timeout all deny.

    Only a click on the window's own card answers; the model cannot, because its text
    reaches the page as text and the request ID is never sent to the model.
    """

    CHOICES = frozenset({"once", "session", "deny"})

    def __init__(self, push: Callable[[Payload], object]) -> None:
        self._push = push
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, list[str]]] = {}
        self._session_tools: set[str] = set()

    def request(self, approval: ApprovalRequest) -> bool:
        if approval.tool in self._session_tools:
            return True
        summary = summarize_approval(approval)
        answered, answer = threading.Event(), []
        with self._lock:
            self._pending[approval.id] = (answered, answer)
        token = current_token()  # the agent worker runs inside the turn's scope
        try:
            self._push(
                {
                    "type": "approval",
                    "id": approval.id,
                    "tool": approval.tool,
                    "heading": summary.heading,
                    "subject": summary.subject,
                    "details": summary.details,
                    "footer": summary.footer,
                    "session": summary.session_allowed,
                }
            )
            while not answered.wait(0.1):
                if token is not None and token.cancelled:
                    answer.append("deny")
                    break
        finally:
            with self._lock:
                self._pending.pop(approval.id, None)
        choice = answer[0]
        if choice == "session":
            self._session_tools.add(approval.tool)
        self._push({"type": "approval-closed", "id": approval.id, "choice": choice})
        return choice != "deny"

    def answer(self, approval_id: str, choice: str) -> bool:
        """Record the user's click; False for an unknown, finished, or invalid answer."""
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or choice not in self.CHOICES:
                return False
            answered, answer = pending
            if answered.is_set():
                return False
            answer.append(choice)
            answered.set()
            return True


class GuiSession:
    """One conversation in the window: one turn or command at a time, cancellable.

    It keeps the terminal's ReplState, so slash commands, statistics, and history work
    the same way. Page calls arrive on pywebview threads; ``submit`` runs work on the
    runtime loop.
    """

    def __init__(
        self,
        application: JamieApplication,
        workspace_tools: WorkspaceTools,
        push: Callable[[Payload], object],
        submit: Submit,
        *,
        model: str | None = None,
        on_quit: Callable[[], object] = lambda: None,
    ) -> None:
        self._push = push
        self._submit = submit
        self._on_quit = on_quit
        self._output = io.StringIO()  # what commands print, sent to the page as text
        console = Console(file=self._output, width=COMMAND_WIDTH, no_color=True)
        self.state = ReplState(
            console, application=application, model=model, workspace_tools=workspace_tools
        )
        self._lock = threading.Lock()
        self._token: CancellationToken | None = None
        self._started: tuple[str, dict[str, Any]] | None = None

    def send(self, prompt: str) -> bool:
        """Start a turn or command; False when blank or while one is already running."""
        prompt = prompt.strip()
        with self._lock:
            if not prompt or self._token is not None:
                return False
            token = self._token = CancellationToken()
        self._submit(self._run(prompt, token))
        return True

    def cancel(self) -> None:
        with self._lock:
            if self._token is not None:
                self._token.cancel()

    def new_conversation(self) -> bool:
        """The plus button: /new without its message, refused while work is running."""
        with self._lock:
            if self._token is not None:
                return False
            COMMANDS.dispatch(self.state, "/new")
            self._take_output()
            return True

    async def _run(self, prompt: str, token: CancellationToken) -> None:
        try:
            if "\n" not in prompt and prompt.startswith("/"):
                await self._command(prompt)
            else:
                await self._turn(prompt, token)
        except TurnCancelled:
            self.state.cancelled += 1
            self._send({"type": "cancelled"})
        except Exception as exc:  # noqa: BLE001 - the window must return to its prompt
            self.state.failures += 1
            self._send({"type": "error", "lines": error_lines(str(exc) or type(exc).__name__)})
        finally:
            with self._lock:
                self._token = None
            self._send({"type": "idle", "workspace": _home(Path.cwd())})

    async def _turn(self, prompt: str, token: CancellationToken) -> None:
        state = self.state
        if state.conversation_id is None:
            state.conversation_id = state.application.start_conversation()
        tools = state.workspace_tools.current(state.scope) if state.mode == "task" else None
        started = time.monotonic()
        result = await state.application.run_turn(
            prompt,
            history=state.history,
            temperature=state.temperature,
            max_tokens=state.max_tokens,
            thinking=state.thinking,
            cancel=token,
            conversation_id=state.conversation_id,
            event_sink=self._show,
            tools=tools,
        )
        state.record_turn(prompt, result, time.monotonic() - started)

    async def _command(self, line: str) -> None:
        state = self.state
        if line.split(" ", 1)[0].lower() == "/clear":
            self._send({"type": "clear"})  # the page clears itself; nothing else changes
            return
        conversation = state.conversation_id
        handled = COMMANDS.dispatch(state, line)
        if inspect.isawaitable(handled):
            await handled  # a command that runs the model or a tool, such as /compact
        text = self._take_output()
        if state.conversation_id != conversation:  # /new or /resume redraws the window
            messages = [{"role": m.role, "text": m.content} for m in state.history]
            self._send({"type": "conversation", "messages": messages, "note": text})
        elif text:
            self._send({"type": "command", "text": text})
        if not state.running:
            self._on_quit()

    def _take_output(self) -> str:
        text = self._output.getvalue().rstrip()
        self._output.seek(0)
        self._output.truncate()
        return text

    def _show(self, event: AgentEvent) -> None:
        self.state.count_tool(event)
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

    approvals = GuiApprovalProvider(push)  # one per window, so "this session" lasts
    sessions: list[GuiSession] = []  # filled once the model is reachable

    def send(prompt: str) -> bool:
        return bool(sessions) and sessions[0].send(prompt)

    def cancel() -> None:
        if sessions:
            sessions[0].cancel()

    def new_conversation() -> bool:
        return bool(sessions) and sessions[0].new_conversation()

    def approve(approval_id: str, choice: str) -> bool:
        return approvals.answer(approval_id, choice)

    window.expose(send, cancel, new_conversation, approve)
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
        tools = WorkspaceTools(
            lambda: create_agent_registry(Path.cwd(), approvals=approvals, web=web, backend=backend)
        )
        resources.callback(tools.close)
        session = GuiSession(
            application, tools, push, runtime.submit, model=info.model, on_quit=window.destroy
        )
        sessions.append(session)
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
