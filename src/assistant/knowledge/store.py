"""Personal memory records and entities, stored beside conversation memory.

The tables live in memory.sqlite3 but are versioned separately from the
conversation schema. Writes resolve contradictions by source authority, merge
repeated facts, and keep a supersession chain so corrections are traceable.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from assistant.knowledge.models import (
    AUTHORITY,
    DEFAULT_CONFIDENCE,
    EntityKind,
    MemoryCandidate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    PersonalRecord,
    WriteAction,
    WriteResult,
)

KNOWLEDGE_SCHEMA_VERSION = 1
MAX_VALUE = 500
MAX_EVIDENCE = 500


def _values(enum) -> str:
    return ", ".join(f"'{member.value}'" for member in enum)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS knowledge_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS personal_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK (type IN ({_values(MemoryType)})),
    subject TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ({_values(MemorySource)})),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    scope TEXT NOT NULL DEFAULT 'global',
    evidence TEXT NOT NULL,
    evidence_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    status TEXT NOT NULL CHECK (status IN ({_values(MemoryStatus)})),
    supersedes INTEGER REFERENCES personal_memories(id),
    mentions INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS personal_memories_slot
    ON personal_memories(type, subject, key, scope, status);

CREATE VIRTUAL TABLE IF NOT EXISTS personal_memories_fts USING fts5(
    subject, key, value, entities,
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ({_values(EntityKind)})),
    name TEXT NOT NULL COLLATE NOCASE,
    aliases TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (kind, name)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id INTEGER NOT NULL REFERENCES personal_memories(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
"""


class PersonalMemoryError(RuntimeError):
    """Personal memory could not be read or written."""


def normalize_key(key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", key.strip().casefold()).strip("_")
    if not cleaned:
        raise ValueError("A memory key must contain letters or digits.")
    return cleaned[:60]


def _same(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def _text(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class PersonalMemory:
    def __init__(self, path: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self.path = Path(path).expanduser()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._initialize()

    def now(self) -> datetime:
        return self._clock().astimezone(UTC)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            created = not self.path.exists()
            with self._connect() as connection:
                connection.executescript(SCHEMA)
                connection.execute(
                    "INSERT INTO knowledge_meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (str(KNOWLEDGE_SCHEMA_VERSION),),
                )
            if created:
                self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            raise PersonalMemoryError(f"Cannot open personal memory at {self.path}.") from exc

    # -- reading --------------------------------------------------------------------

    def _record(self, connection: sqlite3.Connection, row: sqlite3.Row) -> PersonalRecord:
        names = tuple(
            r["name"]
            for r in connection.execute(
                "SELECT e.name FROM entities e JOIN memory_entities m ON m.entity_id = e.id "
                "WHERE m.memory_id = ? ORDER BY e.name",
                (row["id"],),
            )
        )
        return PersonalRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            subject=row["subject"],
            key=row["key"],
            value=row["value"],
            source=MemorySource(row["source"]),
            confidence=row["confidence"],
            scope=row["scope"],
            evidence=row["evidence"],
            evidence_ref=row["evidence_ref"],
            created_at=_time(row["created_at"]),
            updated_at=_time(row["updated_at"]),
            expires_at=_time(row["expires_at"]),
            status=MemoryStatus(row["status"]),
            supersedes=row["supersedes"],
            mentions=row["mentions"],
            entities=names,
        )

    def get(self, record_id: int) -> PersonalRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM personal_memories WHERE id = ?", (record_id,)
            ).fetchone()
            if row is None:
                raise KeyError(record_id)
            return self._record(connection, row)

    def records(
        self,
        *,
        types: Iterable[MemoryType] | None = None,
        statuses: Iterable[MemoryStatus] = (MemoryStatus.ACTIVE,),
        include_expired: bool = False,
    ) -> list[PersonalRecord]:
        clauses = [f"status IN ({', '.join('?' for _ in statuses)})"]
        params: list[object] = [status.value for status in statuses]
        if types is not None:
            types = list(types)
            clauses.append(f"type IN ({', '.join('?' for _ in types)})")
            params += [t.value for t in types]
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(_text(self.now()))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM personal_memories WHERE {' AND '.join(clauses)} "
                "ORDER BY type, subject, key",
                params,
            ).fetchall()
            return [self._record(connection, row) for row in rows]

    def history(self, record_id: int) -> list[PersonalRecord]:
        """The record and every record it superseded, newest first."""
        chain = []
        current: int | None = record_id
        while current is not None:
            record = self.get(current)
            chain.append(record)
            current = record.supersedes
        return chain

    # -- writing --------------------------------------------------------------------

    def remember(self, candidate: MemoryCandidate) -> WriteResult:
        """Store a fact, merging repeats and resolving contradictions by authority."""
        key = normalize_key(candidate.key)
        value = candidate.value.strip()[:MAX_VALUE]
        subject = candidate.subject.strip() or "user"
        if not value:
            raise ValueError("A memory value must not be empty.")
        now = _text(self.now())
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM personal_memories WHERE type = ? AND subject = ? COLLATE NOCASE "
                "AND key = ? AND scope = ? AND status = ? "
                "AND (expires_at IS NULL OR expires_at > ?) ORDER BY id DESC LIMIT 1",
                (candidate.type.value, subject, key, candidate.scope,
                 MemoryStatus.ACTIVE.value, now),
            ).fetchone()  # fmt: skip
            new_authority = AUTHORITY[candidate.source]
            if existing is not None and _same(existing["value"], value):
                upgrade = new_authority > AUTHORITY[MemorySource(existing["source"])]
                connection.execute(
                    "UPDATE personal_memories SET mentions = mentions + 1, updated_at = ?, "
                    "source = CASE WHEN ? THEN ? ELSE source END, "
                    "confidence = CASE WHEN ? THEN ? ELSE confidence END, "
                    "evidence = CASE WHEN ? THEN ? ELSE evidence END WHERE id = ?",
                    (now, upgrade, candidate.source.value, upgrade, self._confidence(candidate),
                     upgrade, candidate.evidence[:MAX_EVIDENCE], existing["id"]),
                )  # fmt: skip
                self._link(connection, existing["id"], candidate.entities)
                self._index(connection, existing["id"])
                record = self._record(connection, self._row(connection, existing["id"]))
                return WriteResult(record, WriteAction.CONSOLIDATED)

            status = MemoryStatus.ACTIVE
            action = WriteAction.CREATED
            supersedes = None
            conflicts: tuple[PersonalRecord, ...] = ()
            if existing is not None:
                previous = self._record(connection, existing)
                conflicts = (previous,)
                if new_authority >= previous.authority:
                    connection.execute(
                        "UPDATE personal_memories SET status = ?, updated_at = ? WHERE id = ?",
                        (MemoryStatus.SUPERSEDED.value, now, previous.id),
                    )
                    self._unindex(connection, previous.id)
                    supersedes, action = previous.id, WriteAction.SUPERSEDED
                else:
                    status, action = MemoryStatus.DISPUTED, WriteAction.DISPUTED
            cursor = connection.execute(
                "INSERT INTO personal_memories (type, subject, key, value, source, confidence, "
                "scope, evidence, evidence_ref, created_at, updated_at, expires_at, status, "
                "supersedes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (candidate.type.value, subject, key, value, candidate.source.value,
                 self._confidence(candidate), candidate.scope, candidate.evidence[:MAX_EVIDENCE],
                 candidate.evidence_ref, now, now, _text(candidate.expires_at), status.value,
                 supersedes),
            )  # fmt: skip
            record_id = int(cursor.lastrowid)
            entities = candidate.entities
            if candidate.type is MemoryType.RELATIONSHIP and subject.casefold() != "user":
                entities = (*entities, subject)
            self._link(connection, record_id, entities)
            if status is MemoryStatus.ACTIVE:
                self._index(connection, record_id)
            record = self._record(connection, self._row(connection, record_id))
        return WriteResult(record, action, conflicts)

    def correct(self, record_id: int, value: str, *, evidence: str | None = None) -> WriteResult:
        """The user's correction always wins and keeps the old record in the history."""
        old = self.get(record_id)
        if old.status is MemoryStatus.DELETED:
            raise ValueError("A forgotten memory cannot be corrected; add a new one instead.")
        with self._connect() as connection:  # a correction may target a disputed record too
            connection.execute(
                "UPDATE personal_memories SET status = ? WHERE id = ? AND status = ?",
                (MemoryStatus.SUPERSEDED.value, record_id, MemoryStatus.DISPUTED.value),
            )
        return self.remember(
            MemoryCandidate(
                type=old.type, subject=old.subject, key=old.key, value=value,
                source=MemorySource.USER_CORRECTION,
                evidence=evidence or f"User corrected {old.key} from {old.value!r}.",
                scope=old.scope, entities=old.entities,
            )
        )  # fmt: skip

    def forget(self, record_id: int) -> None:
        """Delete at the user's request: the content is erased, not just hidden."""
        with self._connect() as connection:
            if self._row(connection, record_id) is None:
                raise KeyError(record_id)
            connection.execute(
                "UPDATE personal_memories SET status = ?, value = '', evidence = '', "
                "evidence_ref = NULL, updated_at = ? WHERE id = ?",
                (MemoryStatus.DELETED.value, _text(self.now()), record_id),
            )
            connection.execute("DELETE FROM memory_entities WHERE memory_id = ?", (record_id,))
            self._unindex(connection, record_id)

    # -- entities -------------------------------------------------------------------

    def add_entity(self, name: str, kind: EntityKind = EntityKind.PERSON,
                   aliases: Iterable[str] = ()) -> int:  # fmt: skip
        with self._connect() as connection:
            return self._entity(connection, name, kind, aliases)

    def find_entity(self, name: str) -> tuple[int, str, EntityKind] | None:
        needle = name.strip().casefold()
        with self._connect() as connection:
            for row in connection.execute("SELECT * FROM entities"):
                aliases = [a.casefold() for a in row["aliases"].split("|") if a]
                if row["name"].casefold() == needle or needle in aliases:
                    return row["id"], row["name"], EntityKind(row["kind"])
        return None

    def entities(self) -> list[tuple[str, EntityKind, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.name, e.kind, COUNT(m.memory_id) AS n FROM entities e "
                "LEFT JOIN memory_entities m ON m.entity_id = e.id GROUP BY e.id ORDER BY e.name"
            ).fetchall()
        return [(row["name"], EntityKind(row["kind"]), row["n"]) for row in rows]

    # -- internals ------------------------------------------------------------------

    @staticmethod
    def _confidence(candidate: MemoryCandidate) -> float:
        default = DEFAULT_CONFIDENCE[candidate.source]
        if candidate.confidence is None:
            return default
        # A model's self-reported confidence can lower, never raise, a record's weight.
        return max(0.0, min(default, candidate.confidence))

    @staticmethod
    def _row(connection: sqlite3.Connection, record_id: int):
        return connection.execute(
            "SELECT * FROM personal_memories WHERE id = ?", (record_id,)
        ).fetchone()

    def _entity(self, connection, name: str, kind: EntityKind, aliases: Iterable[str] = ()) -> int:
        name = name.strip()
        row = connection.execute(
            "SELECT id, aliases FROM entities WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row is not None:
            merged = sorted({*filter(None, row["aliases"].split("|")), *aliases})
            connection.execute("UPDATE entities SET aliases = ? WHERE id = ?",
                               ("|".join(merged), row["id"]))  # fmt: skip
            return row["id"]
        cursor = connection.execute(
            "INSERT INTO entities (kind, name, aliases, created_at) VALUES (?, ?, ?, ?)",
            (kind.value, name, "|".join(sorted(set(aliases))), _text(self.now())),
        )
        return int(cursor.lastrowid)

    def _link(self, connection, record_id: int, names: Iterable[str]) -> None:
        for name in {n.strip() for n in names if n and n.strip()}:
            entity_id = self._entity(connection, name, EntityKind.PERSON)
            connection.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (record_id, entity_id),
            )

    def _index(self, connection, record_id: int) -> None:
        row = self._row(connection, record_id)
        names = " ".join(
            r["name"] + " " + r["aliases"].replace("|", " ")
            for r in connection.execute(
                "SELECT e.name, e.aliases FROM entities e JOIN memory_entities m "
                "ON m.entity_id = e.id WHERE m.memory_id = ?",
                (record_id,),
            )
        )
        self._unindex(connection, record_id)
        connection.execute(
            "INSERT INTO personal_memories_fts (rowid, subject, key, value, entities) "
            "VALUES (?, ?, ?, ?, ?)",
            (record_id, row["subject"], row["key"].replace("_", " "), row["value"], names),
        )

    @staticmethod
    def _unindex(connection, record_id: int) -> None:
        connection.execute("DELETE FROM personal_memories_fts WHERE rowid = ?", (record_id,))
