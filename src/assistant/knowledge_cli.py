"""`jamie memory`: inspect, add, correct, and forget personal memory."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from assistant.config import ConfigurationError, load_config
from assistant.knowledge import (
    MemoryCandidate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    PersonalMemory,
    PersonalMemoryError,
)
from assistant.knowledge.models import SOURCE_LABELS
from assistant.knowledge.retrieval import RecallFilters, contradictions, describe, recall

memory_app = typer.Typer(help="Inspect and correct what Jamie remembers about you.")
ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="Read a specific TOML configuration file.")
]


def _say(text: str) -> None:
    Console().print(text, markup=False, highlight=False, soft_wrap=True)


def _open(config: Path | None) -> PersonalMemory:
    try:
        return PersonalMemory(load_config(config).resolved_memory_path())
    except (ConfigurationError, PersonalMemoryError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc


def _line(record) -> str:
    extra = f" · expires {record.expires_at.date()}" if record.expires_at else ""
    status = "" if record.status is MemoryStatus.ACTIVE else f" · {record.status.value}"
    return (
        f"{record.id:>4}  {record.type.value:<12} {describe(record)}  "
        f"[{SOURCE_LABELS[record.source]}{extra}{status}]"
    )


@memory_app.command("list")
def list_memories(
    type_: Annotated[MemoryType | None, typer.Option("--type", help="Only this type.")] = None,
    all_records: Annotated[
        bool, typer.Option("--all", help="Include superseded, disputed, and forgotten records.")
    ] = False,
    config: ConfigOption = None,
) -> None:
    """List remembered facts."""
    memory = _open(config)
    statuses = tuple(MemoryStatus) if all_records else (MemoryStatus.ACTIVE,)
    records = memory.records(types=[type_] if type_ else None, statuses=statuses,
                             include_expired=all_records)  # fmt: skip
    if not records:
        _say("Nothing remembered yet.")
    for record in records:
        _say(_line(record))
    disputes = contradictions(memory)
    if disputes:
        _say(f"\n{len(disputes)} contradiction(s). Run: jamie memory conflicts")


@memory_app.command("search")
def search(
    query: str,
    person: Annotated[str | None, typer.Option(help="Only facts about this person.")] = None,
    project: Annotated[str | None, typer.Option(help="Only facts about this project.")] = None,
    type_: Annotated[MemoryType | None, typer.Option("--type")] = None,
    stated_only: Annotated[
        bool, typer.Option("--stated-only", help="Exclude inferred facts.")
    ] = False,
    config: ConfigOption = None,
) -> None:
    """Search facts the way Jamie retrieves them for a request."""
    filters = RecallFilters(
        types=(type_,) if type_ else None, person=person, project=project,
        minimum_source=MemorySource.VERIFIED_RECORD if stated_only else None,
    )  # fmt: skip
    results = recall(_open(config), query, filters=filters)
    if not results:
        _say("No matching memories.")
    for item in results:
        _say(_line(item.record))


@memory_app.command("add")
def add(
    type_: Annotated[MemoryType, typer.Argument(metavar="TYPE")],
    key: str,
    value: str,
    subject: Annotated[str, typer.Option(help="Who the fact is about.")] = "user",
    entity: Annotated[list[str] | None, typer.Option(help="Person or project it mentions.")] = None,
    config: ConfigOption = None,
) -> None:
    """Remember a fact you state, e.g. `jamie memory add preference meeting_time morning`."""
    result = _open(config).remember(
        MemoryCandidate(
            type=type_, subject=subject, key=key, value=value,
            source=MemorySource.EXPLICIT_USER_STATEMENT,
            evidence=f"Added with jamie memory add: {key} = {value}",
            entities=tuple(entity or ()),
        )
    )  # fmt: skip
    _say(f"{result.action.value}: {_line(result.record)}")
    for old in result.conflicts:
        _say(f"  replaces: {_line(old)}")


@memory_app.command("correct")
def correct(record_id: int, value: str, config: ConfigOption = None) -> None:
    """Replace a remembered value; the correction outranks everything else."""
    memory = _open(config)
    try:
        result = memory.correct(record_id, value)
    except (KeyError, ValueError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    _say(f"Corrected: {_line(result.record)}")


@memory_app.command("forget")
def forget(record_id: int, config: ConfigOption = None) -> None:
    """Erase a remembered fact."""
    try:
        _open(config).forget(record_id)
    except KeyError as exc:
        Console(stderr=True).print(f"Error: no memory {record_id}", markup=False)
        raise typer.Exit(code=1) from exc
    _say(f"Forgot memory {record_id}.")


@memory_app.command("show")
def show(record_id: int, config: ConfigOption = None) -> None:
    """Show a fact with its evidence and correction history."""
    memory = _open(config)
    try:
        chain = memory.history(record_id)
    except KeyError as exc:
        Console(stderr=True).print(f"Error: no memory {record_id}", markup=False)
        raise typer.Exit(code=1) from exc
    for record in chain:
        _say(_line(record))
        if record.evidence:
            _say(f"      evidence: {record.evidence}")
        _say(f"      learned {record.created_at:%Y-%m-%d %H:%M} · seen {record.mentions}x")


@memory_app.command("conflicts")
def conflicts(config: ConfigOption = None) -> None:
    """Facts that contradict a more authoritative record."""
    pairs = contradictions(_open(config))
    if not pairs:
        _say("No contradictions.")
    for winner, disputed in pairs:
        _say(f"Using:    {_line(winner)}")
        _say(f"Disputed: {_line(disputed)}")
        _say(f"  Keep the disputed value with: jamie memory correct {winner.id} "
             f"{disputed.value!r}\n")  # fmt: skip


@memory_app.command("people")
def people(config: ConfigOption = None) -> None:
    """People and projects Jamie knows about."""
    entities = _open(config).entities()
    if not entities:
        _say("No people or projects yet.")
    for name, kind, count in entities:
        _say(f"{name:<24} {kind.value:<12} {count} fact(s)")
