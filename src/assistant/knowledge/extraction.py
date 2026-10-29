"""Decide which statements become long-term memory.

The model proposes candidates from the **user's own message only**; tool output,
emails, and Jamie's replies are never read here. Python then gates every candidate:
the evidence must be the user's verbatim words outside any quotation, reported
speech and suggestions are dropped, temporary task context is not stored, and the
source (not the model's confidence) sets the record's authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.knowledge.models import (
    MemoryCandidate,
    MemorySource,
    MemoryType,
    WriteResult,
)
from assistant.knowledge.store import PersonalMemory, normalize_key
from assistant.llm.base import LLMBackend, Message, constrained_to

MAX_ITEMS = 5
MAX_EXPIRY_DAYS = 90

# Cheap pre-filter so ordinary requests never cost an extra model call.
_CUES = re.compile(
    r"\b(remember|don't forget|i prefer|i'd prefer|i like|i love|i hate|i dislike|"
    r"i always|i never|i usually|i work|i'm working|i am working|my (?:name|manager|boss|"
    r"professor|partner|wife|husband|team|project|working hours|timezone|birthday)|"
    r"call me|i promised|i'll reply|i will reply|i need to reply|i'm a|i am a|"
    r"is my|are my|actually,? i|from now on|going forward)\b",
    re.IGNORECASE,
)

StatementKind = str  # explicit_user_statement | inference | third_party | suggestion | ...
STORABLE = {"explicit_user_statement", "inference"}

INSTRUCTIONS = """You extract long-term memory from ONE message the user wrote.
Classify each candidate statement:
- explicit_user_statement: the user directly states a fact or preference about themselves,
  their relationships, or their projects ("I prefer morning meetings").
- inference: a reasonable guess that the user did not state. Use sparingly.
- third_party: what someone else said, wrote, or wants, including quoted text.
- suggestion: an idea or option, not a settled fact ("maybe we could meet Friday").
- temporary_context: only matters for the current request ("find the email from today").
- commitment: the user promised to do something ("I promised to reply to Sam by Friday").
Types: profile, preference, relationship, project, commitment.
"evidence" must be copied exactly from the user's message. Use short snake_case keys.
For a relationship, subject is the person's name and key describes the relationship.
Return at most 5 items, or an empty list when nothing should be remembered."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": MAX_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "enum": [
                        "explicit_user_statement", "inference", "third_party", "suggestion",
                        "temporary_context", "commitment"]},
                    "type": {"type": "string", "enum": [
                        "profile", "preference", "relationship", "project", "commitment"]},
                    "subject": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "evidence": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "expires_in_days": {"type": ["integer", "null"]},
                },
                "required": ["statement", "type", "subject", "key", "value", "evidence"],
            },
        }
    },
    "required": ["items"],
}  # fmt: skip


class ProposedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    statement: str
    type: str
    subject: str = Field(max_length=120)
    key: str = Field(max_length=80)
    value: str = Field(max_length=500)
    evidence: str = Field(max_length=500)
    entities: list[str] = Field(default_factory=list, max_length=5)
    expires_in_days: int | None = None


class Proposal(BaseModel):
    items: list[ProposedItem] = Field(default_factory=list, max_length=MAX_ITEMS)


@dataclass
class Decision:
    item: ProposedItem
    accepted: bool
    reason: str
    candidate: MemoryCandidate | None = None


@dataclass
class ExtractionOutcome:
    decisions: list[Decision] = field(default_factory=list)
    written: list[WriteResult] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)


def likely_personal(message: str) -> bool:
    return bool(_CUES.search(message))


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{12,}'|^>.*$", re.MULTILINE)
_REPORTED = re.compile(
    r"\b(?:he|she|they|[A-Z][a-z]+)\s+(?:said|says|wrote|writes|thinks|thought|wants|"
    r"asked|told me|mentioned|suggested)\b"
)


def _quoted_spans(message: str) -> list[str]:
    return [_normalize(m.group(0)) for m in _QUOTED.finditer(message)]


def gate(message: str, item: ProposedItem, *, now: datetime,
         evidence_ref: str | None = None) -> Decision:  # fmt: skip
    """Accept a proposed item only when the user's own words support it."""
    if item.statement not in STORABLE and item.statement != "commitment":
        return Decision(item, False, f"{item.statement} is not stored")
    evidence = _normalize(item.evidence)
    if len(evidence) < 3 or evidence not in _normalize(message):
        return Decision(item, False, "evidence is not the user's own words")
    if any(evidence in span or span in evidence for span in _quoted_spans(message) if span):
        return Decision(item, False, "evidence is quoted text, treated as third party")
    sentence = next((s for s in re.split(r"(?<=[.!?])\s+", message)
                     if evidence in _normalize(s)), message)  # fmt: skip
    reported = _REPORTED.search(sentence)
    if reported is not None:
        spoken_from = len(_normalize(sentence[: reported.start()]))
        if _normalize(sentence).find(evidence) >= spoken_from:
            return Decision(item, False, "reported speech, treated as third party")
    if item.statement == "commitment" or item.type == "commitment":
        return Decision(item, True, "commitment")
    try:
        memory_type = MemoryType(item.type)
        key = normalize_key(item.key)
    except ValueError:
        return Decision(item, False, "unsupported type or key")
    if not item.value.strip():
        return Decision(item, False, "empty value")
    source = (
        MemorySource.EXPLICIT_USER_STATEMENT
        if item.statement == "explicit_user_statement"
        else MemorySource.INFERENCE
    )
    expires = None
    if item.expires_in_days is not None:
        expires = now + timedelta(days=max(1, min(item.expires_in_days, MAX_EXPIRY_DAYS)))
    subject = item.subject.strip()
    if memory_type is not MemoryType.RELATIONSHIP or subject.casefold() in {"me", "i", ""}:
        subject = "user" if memory_type is not MemoryType.RELATIONSHIP else subject or "user"
    candidate = MemoryCandidate(
        type=memory_type, subject=subject, key=key, value=item.value.strip(), source=source,
        evidence=item.evidence.strip(), evidence_ref=evidence_ref,
        entities=tuple(e.strip() for e in item.entities if e.strip()), expires_at=expires,
    )  # fmt: skip
    return Decision(item, True, "accepted", candidate)


class MemoryExtractor:
    def __init__(
        self,
        backend: LLMBackend,
        memory: PersonalMemory,
        *,
        on_commitment: Callable[[str, str], None] | None = None,
    ) -> None:
        self._backend = backend
        self._memory = memory
        self._on_commitment = on_commitment

    def propose(self, message: str) -> Proposal:
        messages = [Message("system", INSTRUCTIONS), Message("user", message)]
        with constrained_to(SCHEMA):
            response = self._backend.chat(messages, temperature=0.0, max_tokens=500)
        try:
            return Proposal.model_validate(json.loads(response.content))
        except (ValueError, ValidationError):
            return Proposal()

    def process(self, message: str, *, evidence_ref: str | None = None,
                force: bool = False) -> ExtractionOutcome:  # fmt: skip
        outcome = ExtractionOutcome()
        if not force and not likely_personal(message):
            return outcome
        now = self._memory.now()
        for item in self.propose(message).items:
            decision = gate(message, item, now=now, evidence_ref=evidence_ref)
            outcome.decisions.append(decision)
            if not decision.accepted:
                continue
            if decision.candidate is None:  # a commitment is reported, not stored as a fact
                text = item.value.strip() or item.evidence.strip()
                outcome.commitments.append(text)
                if self._on_commitment is not None:
                    self._on_commitment(text, item.evidence.strip())
                continue
            outcome.written.append(self._memory.remember(decision.candidate))
        return outcome
