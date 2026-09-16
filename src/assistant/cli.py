"""Typer commands for the milestone 0 foundation."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from assistant import __version__
from assistant.config import ConfigurationError, load_config

MINIMUM_PYTHON = (3, 12)

app = typer.Typer(
    help="A local-first macOS assistant. Milestone 0: project diagnostics only.",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        Console().print(f"assistant {__version__}", markup=False, highlight=False)
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
    """Run the assistant CLI."""


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
    table = Table(title="Assistant doctor")
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
