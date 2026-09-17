"""Typer commands for Jamie."""

import math
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from assistant import __version__
from assistant.config import ConfigurationError, load_config
from assistant.eval_cli import app as eval_app
from assistant.llm.base import LLMBackend, LLMError, Message
from assistant.llm.llamacpp import LlamaCppBackend

MINIMUM_PYTHON = (3, 12)

app = typer.Typer(
    name="jamie",
    help="A local-first macOS assistant.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(eval_app, name="eval")


def version_callback(value: bool) -> None:
    if value:
        Console().print(f"jamie {__version__}", markup=False, highlight=False)
        raise typer.Exit()


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
        backend: LLMBackend = LlamaCppBackend()
        response = backend.chat(
            [Message(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (ConfigurationError, LLMError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    Console(no_color=not settings.color).print(response.content, markup=False, highlight=False)
    if response.finish_reason == "length":
        Console(stderr=True, no_color=not settings.color).print(
            "The response reached the token limit; use --max-tokens to allow a longer answer.",
            markup=False,
            highlight=False,
        )


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
