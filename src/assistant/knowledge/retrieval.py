"""Retrieve personal facts: FTS5 relevance, filters, recency, authority, and a budget.

Only records that match the request (or the requested person/project) are returned,
so unrelated private facts stay out of the prompt. Every line carries provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from assistant.knowledge.models import (
    AUTHORITY,
    SOURCE_LABELS,
    MemorySource,
    MemoryStatus,
    MemoryType,
    PersonalRecord,
)
from assistant.knowledge.store import PersonalMemory

CHARACTERS_PER_TOKEN = 4
RECENCY_HALF_LIFE_DAYS = 90
DEFAULT_BUDGET_TOKENS = 400
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = {
    "a", "an", "and", "are", "at", "be", "can", "do", "does", "for", "from", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "the", "to", "what",
    "when", "where", "which", "who", "with", "you", "your", "about", "should", "would",
}  # fmt: skip


@dataclass(frozen=True)
class RecallFilters:
    types: tuple[MemoryType, ...] | None = None
    person: str | None = None
    project: str | None = None
    minimum_source: MemorySource | None = None  # e.g. exclude inferences
    since: datetime | None = None


@dataclass(frozen=True)
class ScoredRecord:
    record: PersonalRecord
    score: float


def _match_query(text: str) -> str | None:
    tokens = []
    for token in _WORD.findall(text.casefold()):
        if len(token) > 1 and token not in _STOP and token not in tokens:
            tokens.append(token)
    return " OR ".join(f'"{t}"' for t in tokens[:16]) if tokens else None


def recall(
    memory: PersonalMemory,
    query: str,
    *,
    filters: RecallFilters | None = None,
    limit: int = 8,
) -> list[ScoredRecord]:
    filters = filters or RecallFilters()
    now = memory.now()
    candidates: dict[int, float] = {}
    match = _match_query(query)
    with memory._connect() as connection:
        if match is not None:
            for row in connection.execute(
                "SELECT rowid, bm25(personal_memories_fts) AS rank FROM personal_memories_fts "
                "WHERE personal_memories_fts MATCH ? ORDER BY rank LIMIT 50",
                (match,),
            ):
                candidates[row["rowid"]] = -row["rank"]  # bm25: lower is better
        for name in filter(None, (filters.person, filters.project)):
            found = memory.find_entity(name)
            if found is None:
                continue
            for row in connection.execute(
                "SELECT memory_id FROM memory_entities WHERE entity_id = ?", (found[0],)
            ):
                candidates.setdefault(row["memory_id"], 1.0)
    scored: list[ScoredRecord] = []
    for record_id, relevance in candidates.items():
        try:
            record = memory.get(record_id)
        except KeyError:
            continue
        if record.status is not MemoryStatus.ACTIVE:
            continue
        if record.expires_at is not None and record.expires_at <= now:
            continue
        if filters.types is not None and record.type not in filters.types:
            continue
        if filters.since is not None and record.updated_at < filters.since:
            continue
        if filters.minimum_source is not None and (
            record.authority < AUTHORITY[filters.minimum_source]
        ):
            continue
        if filters.person and not _mentions(record, filters.person):
            continue
        if filters.project and not _mentions(record, filters.project):
            continue
        age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400)
        recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        score = max(relevance, 0.01) * (0.6 + 0.4 * recency) * (0.7 + 0.1 * record.authority)
        scored.append(ScoredRecord(record, score))
    scored.sort(key=lambda item: item.score, reverse=True)
    return _consolidate(scored)[:limit]


def _mentions(record: PersonalRecord, name: str) -> bool:
    needle = name.casefold()
    return needle in record.subject.casefold() or any(
        needle in entity.casefold() for entity in record.entities
    )


def _consolidate(scored: list[ScoredRecord]) -> list[ScoredRecord]:
    """Drop repeated facts that differ only in case or spacing."""
    seen: set[tuple] = set()
    unique = []
    for item in scored:
        r = item.record
        signature = (r.type, r.subject.casefold(), r.key, " ".join(r.value.casefold().split()))
        if signature not in seen:
            seen.add(signature)
            unique.append(item)
    return unique


def contradictions(memory: PersonalMemory) -> list[tuple[PersonalRecord, PersonalRecord]]:
    """Disputed records paired with the active record they contradict."""
    pairs = []
    active = {(r.type, r.subject.casefold(), r.key, r.scope): r for r in memory.records()}
    for disputed in memory.records(statuses=(MemoryStatus.DISPUTED,)):
        winner = active.get((disputed.type, disputed.subject.casefold(), disputed.key,
                             disputed.scope))  # fmt: skip
        if winner is not None:
            pairs.append((winner, disputed))
    return pairs


def describe(record: PersonalRecord) -> str:
    who = "" if record.subject.casefold() == "user" else f"{record.subject} · "
    key = record.key.replace("_", " ")
    return f"{who}{key}: {record.value}"


def format_personal_context(
    records: list[ScoredRecord], budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> str | None:
    """Facts for the system prompt, with provenance, cut to fit the token budget."""
    if not records:
        return None
    header = (
        "Known facts about the user from local memory. Provenance is in brackets; "
        "facts stated or corrected by the user outrank inferred ones. Use only facts "
        "relevant to the request.\n"
    )
    budget = budget_tokens * CHARACTERS_PER_TOKEN - len(header)
    lines = []
    for item in records:
        r = item.record
        line = (
            f"- {describe(r)} [{r.type.value}; {SOURCE_LABELS[r.source]}; "
            f"{r.updated_at.date().isoformat()}; memory {r.id}]\n"
        )
        if len(line) > budget:
            break
        lines.append(line)
        budget -= len(line)
    return header + "".join(lines) if lines else None
