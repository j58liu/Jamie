"""Typed personal memory: what Jamie knows about the user, and how it knows it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
    PROFILE = "profile"  # working hours, timezone, name
    PREFERENCE = "preference"  # prefers morning meetings
    RELATIONSHIP = "relationship"  # Alex is a project collaborator
    PROJECT = "project"  # Project Jamie has a release milestone
    KNOWLEDGE = "knowledge"  # a fact from a document, with a source reference


class MemorySource(StrEnum):
    """How a record was learned. Authority comes from this, never from model confidence."""

    USER_CORRECTION = "user_correction"
    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    VERIFIED_RECORD = "verified_record"
    INFERENCE = "inference"


AUTHORITY = {
    MemorySource.USER_CORRECTION: 4,
    MemorySource.EXPLICIT_USER_STATEMENT: 3,
    MemorySource.VERIFIED_RECORD: 2,
    MemorySource.INFERENCE: 1,
}

DEFAULT_CONFIDENCE = {
    MemorySource.USER_CORRECTION: 1.0,
    MemorySource.EXPLICIT_USER_STATEMENT: 1.0,
    MemorySource.VERIFIED_RECORD: 0.9,
    MemorySource.INFERENCE: 0.5,
}

SOURCE_LABELS = {
    MemorySource.USER_CORRECTION: "corrected by user",
    MemorySource.EXPLICIT_USER_STATEMENT: "stated by user",
    MemorySource.VERIFIED_RECORD: "verified record",
    MemorySource.INFERENCE: "inferred, may be wrong",
}


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer or more authoritative record
    DISPUTED = "disputed"  # contradicts a more authoritative record; not used as fact
    DELETED = "deleted"  # forgotten at the user's request; content erased


class EntityKind(StrEnum):
    PERSON = "person"
    PROJECT = "project"
    ORGANIZATION = "organization"


@dataclass(frozen=True)
class MemoryCandidate:
    """A proposed record, before conflict resolution."""

    type: MemoryType
    subject: str
    key: str
    value: str
    source: MemorySource
    evidence: str
    scope: str = "global"
    evidence_ref: str | None = None
    entities: tuple[str, ...] = ()
    expires_at: datetime | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PersonalRecord:
    id: int
    type: MemoryType
    subject: str
    key: str
    value: str
    source: MemorySource
    confidence: float
    scope: str
    evidence: str
    evidence_ref: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    status: MemoryStatus
    supersedes: int | None
    mentions: int
    entities: tuple[str, ...] = field(default=())

    @property
    def authority(self) -> int:
        return AUTHORITY[self.source]


class WriteAction(StrEnum):
    CREATED = "created"
    CONSOLIDATED = "consolidated"  # same fact again; evidence count increased
    SUPERSEDED = "superseded"  # replaced an older value
    DISPUTED = "disputed"  # kept, but a more authoritative record wins


@dataclass(frozen=True)
class WriteResult:
    record: PersonalRecord
    action: WriteAction
    conflicts: tuple[PersonalRecord, ...] = ()
