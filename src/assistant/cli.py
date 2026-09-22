"""Typer commands for Jamie."""

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

from assistant import __version__
from assistant.agent import AgentError, run_agent
from assistant.agent.events import EventSink
from assistant.config import ConfigurationError, ProjectConfig, load_config
from assistant.eval_cli import app as eval_app
from assistant.llm.base import LLMError
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import (
    MemoryStore,
    MemoryStoreError,
    RememberingTools,
    format_memory_context,
)
from assistant.research import ResearchError, ResearchWorkflow
from assistant.runtime import JamieApplication, JamieApplicationError, TurnResult
from assistant.security import ApprovalRequest
from assistant.tools import WebClient, create_agent_registry
from assistant.ui.renderer import OneShotRenderer

MINIMUM_PYTHON = (3, 12)


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
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(eval_app, name="eval")


class TerminalApprovalProvider:
    """Show the exact requested action and ask the terminal user to approve it."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def request(self, approval: ApprovalRequest) -> bool:
        arguments = json.dumps(
            approval.arguments,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        self._console.print("Approval required", markup=False, highlight=False)
        self._console.print(f"Tool: {approval.tool}", markup=False, highlight=False)
        self._console.print(
            f"Permission: {approval.permission.value}", markup=False, highlight=False
        )
        self._console.print(f"Arguments:\n{arguments}", markup=False, highlight=False)
        return typer.confirm("Allow this action?", default=False)


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
    event_sink: EventSink,
) -> TurnResult:
    async with JamieApplication(settings, backend_factory=LlamaCppBackend) as application:
        return await application.run_turn(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            event_sink=event_sink,
        )


@app.callback()
def main(
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
    """Run the Jamie CLI."""


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="Text to send to the local model.")],
    temperature: Annotated[float, typer.Option(min=0.0, help="Sampling temperature.")] = 0.6,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Maximum number of generated tokens.")
    ] = 1024,
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
    max_steps: Annotated[int, typer.Option(min=1, help="Maximum number of model actions.")] = 12,
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Maximum seconds for the complete task.")
    ] = 300.0,
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

    store = None
    session_id = None
    try:
        settings = load_config(config)
        console = Console(no_color=not settings.color)
        store, session_id, memory_context = _start_memory(settings, "agent", prompt)
        with WebClient() as web:
            tools = create_agent_registry(
                workspace,
                approvals=TerminalApprovalProvider(console),
                web=web,
            )
            answer = run_agent(
                LlamaCppBackend(),
                RememberingTools(tools, store, session_id),
                prompt,
                max_steps=max_steps,
                timeout_seconds=timeout,
                memory_context=memory_context,
            )
        _complete_memory(store, session_id, prompt, answer)
    except (AgentError, ConfigurationError, LLMError, MemoryStoreError, ValueError) as exc:
        _fail_memory(store, session_id)
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    console.print(answer, markup=False, highlight=False)


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
        with WebClient() as web:
            result = ResearchWorkflow(LlamaCppBackend(), web).run(
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
def doctor(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Validate a specific TOML configuration file."),
    ] = None,
) -> None:
    """Check only the Python version and local project configuration."""
    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    table = Table(title="Jamie doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    table.add_row(
        "Python version",
        Text("PASS" if python_ok else "FAIL", style="green" if python_ok else "red"),
        f"{python_version} (requires 3.12+)",
    )

    console = Console()
    config_ok = True
    try:
        settings = load_config(config)
    except ConfigurationError as exc:
        config_ok = False
        table.add_row("Project configuration", Text("FAIL", style="red"), Text(str(exc)))
    else:
        console = Console(no_color=not settings.color)
        table.add_row("Project configuration", Text("PASS", style="green"), "Valid")

    console.print(table)
    if not (python_ok and config_ok):
        raise typer.Exit(code=1)
