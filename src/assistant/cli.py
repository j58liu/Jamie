"""Typer commands for Jamie."""

import asyncio
import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text
from typer.core import TyperGroup

from assistant import STARTED, __version__
from assistant.agent import AgentError
from assistant.agent.events import EventSink
from assistant.agent.loop import MAX_STEPS
from assistant.cancellation import CancellationToken, TurnCancelled, interrupt_cancels
from assistant.config import ConfigurationError, ProjectConfig, load_config
from assistant.doctor import Status, run_checks
from assistant.eval_cli import app as eval_app
from assistant.knowledge_cli import memory_app
from assistant.llm.base import LLMError, managed_backend
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import (
    MemoryStore,
    MemoryStoreError,
    format_memory_context,
)
from assistant.research import ResearchError, ResearchWorkflow
from assistant.runtime import JamieApplication, JamieApplicationError, TurnResult
from assistant.tools import WebClient, create_agent_registry
from assistant.tools.scopes import SCOPE_MAX_STEPS, ToolScope, select_tools
from assistant.ui.approvals import TerminalApprovalProvider
from assistant.ui.gui import run_gui
from assistant.ui.renderer import EventRenderer, OneShotRenderer
from assistant.ui.repl import run_repl


class JamieGroup(TyperGroup):
    """Route a prompt without a command name to the agent command."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and self.get_command(ctx, args[0]) is None:
            agent_command = self.get_command(ctx, "agent")
            if agent_command is not None:
                return "agent", agent_command, args
        return super().resolve_command(ctx, args)


app = typer.Typer(
    name="jamie",
    cls=JamieGroup,
    help="A local-first macOS assistant.",
    add_completion=False,
)
app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")


def _start_memory(settings, kind: str, prompt: str) -> tuple[MemoryStore, str, str | None]:
    store = MemoryStore(settings.resolved_memory_path())
    session_id = store.start_session(kind)
    try:
        store.add_message(session_id, "user", prompt)
        context = format_memory_context(store.search_memories(prompt))
    except MemoryStoreError:
        _fail_memory(store, session_id)
        raise
    return store, session_id, context


def _complete_memory(store: MemoryStore, session_id: str, prompt: str, answer: str) -> None:
    store.add_message(session_id, "assistant", answer)
    store.remember_exchange(session_id, prompt, answer)
    store.finish_session(session_id, success=True)


def _fail_memory(store: MemoryStore | None, session_id: str | None) -> None:
    if store is None or session_id is None:
        return
    try:
        store.finish_session(session_id, success=False)
    except MemoryStoreError:
        pass


def version_callback(value: bool) -> None:
    if value:
        Console().print(f"jamie {__version__}", markup=False, highlight=False)
        raise typer.Exit()


async def _run_ask(
    settings: ProjectConfig,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool | None,
    event_sink: EventSink,
) -> TurnResult:
    async with JamieApplication(settings, backend_factory=LlamaCppBackend) as application:
        return await application.run_turn(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            event_sink=event_sink,
        )


async def _run_agent(
    settings: ProjectConfig,
    prompt: str,
    workspace: Path,
    *,
    max_steps: int,
    timeout: float,
    scope: ToolScope = ToolScope.ALL,
    plan: str = "auto",
) -> None:
    """One agent turn on the shared runtime; Ctrl+C cancels it at the next safe point."""
    console = Console(no_color=not settings.color)
    renderer = EventRenderer(console, Console(stderr=True, no_color=not settings.color))
    backend = LlamaCppBackend()
    token = CancellationToken()
    async with JamieApplication(
        settings,
        backend_factory=lambda: backend,
        extract_memories=settings.memory.extraction == "cue",
    ) as application:
        with WebClient() as web, interrupt_cancels(token):
            tools = create_agent_registry(
                workspace,
                approvals=TerminalApprovalProvider(console),
                web=web,
                backend=backend,
                connectors=application.connectors,
            )
            tools = select_tools(tools, scope)
            await application.run_turn(
                prompt,
                tools=tools,
                max_steps=max_steps,
                timeout_seconds=timeout,
                cancel=token,
                event_sink=renderer.handle,
                plan=plan,
            )


class StartupError(RuntimeError):
    """The interactive session cannot start; the message says why."""

    def __init__(self, server: str, reason: str) -> None:
        super().__init__(reason)
        self.server = server


def _startup_failure(error: StartupError) -> None:
    Console(stderr=True).print(
        "Jamie could not start the local model.\n\n"
        f"Server:\nllama.cpp at {error.server}\n\n"
        f"Reason:\n{error}\n\n"
        "Run:\njamie doctor",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


async def _run_repl(settings: ProjectConfig) -> None:
    backend = LlamaCppBackend()
    try:
        info = backend.server_info()
    except LLMError as exc:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
        raise StartupError(backend.server_url, str(exc)) from exc
    console = Console(no_color=not settings.color)
    approvals = TerminalApprovalProvider(console)  # one per session, so "allow session" lasts
    async with JamieApplication(
        settings,
        backend_factory=lambda: backend,
        extract_memories=settings.memory.extraction == "cue",
    ) as application:
        with WebClient() as web:
            await run_repl(
                application,
                console=console,
                model=info.model,
                started=STARTED,
                agent_tools_factory=lambda: create_agent_registry(
                    Path.cwd(),
                    approvals=approvals,
                    web=web,
                    backend=backend,
                    connectors=application.connectors,
                ),
            )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Run the Jamie CLI, or start the interactive session without a command."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        settings = load_config(None)
        asyncio.run(_run_repl(settings))
    except StartupError as exc:
        _startup_failure(exc)
        raise typer.Exit(code=1) from exc
    except (ConfigurationError, JamieApplicationError, LLMError, MemoryStoreError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="Text to send to the local model.")],
    temperature: Annotated[float, typer.Option(min=0.0, help="Sampling temperature.")] = 0.6,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Maximum number of generated tokens.")
    ] = 1024,
    thinking: Annotated[
        bool | None,
        typer.Option(
            "--thinking/--no-thinking",
            help="Turn model reasoning on or off. Defaults to the server setting.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Send one prompt to llama.cpp and print its answer."""
    if not prompt.strip():
        raise typer.BadParameter("Prompt must not be empty.", param_hint="PROMPT")
    if not math.isfinite(temperature):
        raise typer.BadParameter("Temperature must be finite.", param_hint="--temperature")

    try:
        settings = load_config(config)
        renderer = OneShotRenderer(
            Console(no_color=not settings.color),
            Console(stderr=True, no_color=not settings.color),
        )
        asyncio.run(
            _run_ask(
                settings,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
                event_sink=renderer.handle,
            )
        )
    except (ConfigurationError, JamieApplicationError, LLMError, MemoryStoreError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc


@app.command()
def agent(
    prompt: Annotated[str, typer.Argument(help="Task for the local agent.")],
    workspace: Annotated[Path, typer.Option(help="Workspace available to file tools.")] = Path("."),
    max_steps: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Maximum number of model actions. Defaults to 12, or 16 for --scope personal.",
            show_default=False,
        ),
    ] = None,
    scope: Annotated[
        ToolScope, typer.Option(help="Tools available for this task.")
    ] = ToolScope.ALL,
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Maximum seconds for the complete task.")
    ] = 300.0,
    plan: Annotated[
        str,
        typer.Option(help="Use the step planner: auto (multi-step requests), always, or never."),
    ] = "auto",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Run the local model with approved workspace and macOS tools."""
    if not prompt.strip():
        raise typer.BadParameter("Prompt must not be empty.", param_hint="PROMPT")
    if not math.isfinite(timeout):
        raise typer.BadParameter("Timeout must be finite.", param_hint="--timeout")

    try:
        settings = load_config(config)
        steps = max_steps if max_steps is not None else SCOPE_MAX_STEPS.get(scope, MAX_STEPS)
        if plan not in {"auto", "always", "never"}:
            raise typer.BadParameter("Use auto, always, or never.", param_hint="--plan")
        asyncio.run(
            _run_agent(
                settings,
                prompt,
                workspace,
                max_steps=steps,
                timeout=timeout,
                scope=scope,
                plan=plan,
            )
        )
    except TurnCancelled as exc:
        Console(stderr=True).print("Cancelled.", markup=False, highlight=False)
        raise typer.Exit(code=130) from exc
    except (
        AgentError,
        ConfigurationError,
        JamieApplicationError,
        LLMError,
        MemoryStoreError,
        ValueError,
    ) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="Question to research on the public web.")],
    max_sources: Annotated[
        int, typer.Option(min=2, max=8, help="Maximum number of pages to use.")
    ] = 5,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Research a question and return an answer with checked citations."""
    if not question.strip():
        raise typer.BadParameter("Question must not be empty.", param_hint="QUESTION")
    store = None
    session_id = None
    try:
        settings = load_config(config)
        store, session_id, _ = _start_memory(settings, "research", question)
        with WebClient() as web, managed_backend(LlamaCppBackend()) as backend:
            result = ResearchWorkflow(backend, web).run(
                question,
                max_sources=max_sources,
            )
        for source in result.sources:
            store.add_source(
                session_id,
                url=source.url,
                title=source.title,
                content=source.text,
            )
        _complete_memory(store, session_id, question, result.answer)
    except (ConfigurationError, MemoryStoreError, ResearchError) as exc:
        _fail_memory(store, session_id)
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    Console(no_color=not settings.color).print(result.answer, markup=False, highlight=False)


@app.command()
def gui(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
    hidden: Annotated[
        bool, typer.Option("--hidden", help="Start in the background without the window.")
    ] = False,
) -> None:
    """Open Jamie in a window; Jamie.app runs this command."""
    try:
        settings = load_config(config)
    except ConfigurationError as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    run_gui(settings, hidden=hidden)


connect_app = typer.Typer(help="Connect personal applications to Jamie.")
app.add_typer(connect_app, name="connect")


def _google_session(config: Path | None):
    from assistant.connectors import build_connectors

    settings = load_config(config)
    session = build_connectors(settings).google
    if session is None or not settings.google.enabled:
        raise ConfigurationError(
            "Add a [google] section with client_secrets_path to assistant.toml first."
        )
    return session


@connect_app.command("google")
def connect_google(
    scopes: Annotated[
        str,
        typer.Option(help="Access level: read, compose (drafts and send), or calendar-write."),
    ] = "read",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Sign in to Google in the browser and store the refresh token in Keychain."""
    import webbrowser

    from assistant.connectors import ConnectorError
    from assistant.connectors.google.auth import SCOPE_LEVELS

    if scopes not in SCOPE_LEVELS:
        raise typer.BadParameter(
            f"Choose one of: {', '.join(SCOPE_LEVELS)}.", param_hint="--scopes"
        )
    console = Console()
    try:
        session = _google_session(config)

        def open_browser(url: str) -> None:
            console.print("Opening Google sign-in. If no browser opens, visit:", markup=False)
            console.print(url, markup=False, highlight=False, soft_wrap=True)
            webbrowser.open(url)

        granted = session.connect(SCOPE_LEVELS[scopes], open_browser=open_browser)
    except (ConfigurationError, ConnectorError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    names = ", ".join(sorted(scope.rsplit("/", 1)[-1] for scope in granted))
    console.print(f"Google connected: {names}", markup=False, highlight=False)


@app.command()
def disconnect(
    provider: Annotated[str, typer.Argument(help="Provider to disconnect: google.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Revoke and delete stored credentials for a provider."""
    if provider != "google":
        raise typer.BadParameter("Only google can be disconnected.", param_hint="PROVIDER")
    try:
        _google_session(config).disconnect()
    except ConfigurationError as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    Console().print("Google disconnected.", markup=False, highlight=False)


@app.command()
def connectors(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Read a specific TOML configuration file."),
    ] = None,
) -> None:
    """Show the status of each personal-app connector."""
    from assistant.connectors import build_connectors

    try:
        found = build_connectors(load_config(config)).all()
    except ConfigurationError as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    console = Console()
    if not found:
        console.print("No connectors configured. See docs/design/connectors.md.", markup=False)
        return
    for connector in found:
        info = connector.info()
        capabilities = ", ".join(sorted(c.value for c in info.capabilities)) or "none"
        line = f"{info.name:<16} {info.status.value:<18} {capabilities}"
        console.print(
            line + (f"  {info.detail}" if info.detail else ""), markup=False, highlight=False
        )


SYMBOLS = {
    Status.PASS: ("✓", "green"),
    Status.WARN: ("!", "yellow"),
    Status.SKIP: ("–", "dim"),
    Status.FAIL: ("✗", "red"),
}


@app.command()
def doctor(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Validate a specific TOML configuration file."),
    ] = None,
) -> None:
    """Check Jamie's prerequisites without changing anything."""
    with managed_backend(LlamaCppBackend()) as backend:
        checks = run_checks(config, backend)
    console = Console()
    width = max(len(check.name) for check in checks)
    for check in checks:
        symbol, style = SYMBOLS[check.status]
        line = Text(f"{symbol} ", style=style)
        line.append(f"{check.name:<{width}}  ")
        line.append(check.detail, style="dim" if check.status is Status.PASS else style)
        console.print(line, soft_wrap=True)
    failed = [check.name for check in checks if check.status is Status.FAIL]
    console.print()
    if failed:
        console.print(
            f"Jamie is not ready: fix {', '.join(failed)}.", markup=False, highlight=False
        )
        raise typer.Exit(code=1)
    console.print("Jamie is ready.", markup=False, highlight=False)
