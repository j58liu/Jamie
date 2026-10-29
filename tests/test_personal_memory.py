from datetime import UTC, datetime, timedelta

import pytest

from assistant.knowledge import (
    MemoryCandidate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    PersonalMemory,
    WriteAction,
)
from assistant.knowledge.retrieval import (
    RecallFilters,
    contradictions,
    format_personal_context,
    recall,
)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 25, 12, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def memory(tmp_path, clock):
    return PersonalMemory(tmp_path / "memory.sqlite3", clock=clock)


def fact(key, value, *, type_=MemoryType.PREFERENCE, source=MemorySource.EXPLICIT_USER_STATEMENT,
         subject="user", **extra):  # fmt: skip
    return MemoryCandidate(type=type_, subject=subject, key=key, value=value, source=source,
                           evidence=f"I said {key} is {value}", **extra)  # fmt: skip


def test_repeated_fact_consolidates(memory):
    first = memory.remember(fact("meeting time", "morning"))
    again = memory.remember(fact("Meeting_Time", "Morning "))
    assert first.action is WriteAction.CREATED and again.action is WriteAction.CONSOLIDATED
    assert again.record.id == first.record.id and again.record.mentions == 2
    assert len(memory.records()) == 1


def test_newer_statement_replaces_older_one(memory):
    old = memory.remember(fact("meeting_time", "morning")).record
    new = memory.remember(fact("meeting_time", "afternoon"))
    assert new.action is WriteAction.SUPERSEDED and new.conflicts[0].id == old.id
    assert memory.get(old.id).status is MemoryStatus.SUPERSEDED
    assert [r.value for r in memory.records()] == ["afternoon"]


def test_inference_cannot_override_a_stated_fact(memory):
    memory.remember(fact("meeting_time", "morning"))
    guess = memory.remember(fact("meeting_time", "evening", source=MemorySource.INFERENCE,
                                 confidence=0.99))  # fmt: skip
    assert guess.action is WriteAction.DISPUTED
    assert guess.record.confidence == 0.5  # model confidence never raises authority
    assert [r.value for r in memory.records()] == ["morning"]
    [(winner, disputed)] = contradictions(memory)
    assert (winner.value, disputed.value) == ("morning", "evening")


def test_correction_wins_and_keeps_history(memory):
    original = memory.remember(fact("working_hours", "9 to 5", type_=MemoryType.PROFILE)).record
    corrected = memory.correct(original.id, "10 to 6")
    assert corrected.record.source is MemorySource.USER_CORRECTION
    assert [r.value for r in memory.history(corrected.record.id)] == ["10 to 6", "9 to 5"]
    later = memory.remember(fact("working_hours", "8 to 4", type_=MemoryType.PROFILE))
    assert later.action is WriteAction.DISPUTED  # a plain statement does not undo a correction


def test_forget_erases_content_and_search(memory):
    record = memory.remember(fact("doctor", "Dr. Patel", type_=MemoryType.PROFILE)).record
    memory.forget(record.id)
    gone = memory.get(record.id)
    assert gone.status is MemoryStatus.DELETED and gone.value == "" and gone.evidence == ""
    assert recall(memory, "who is my doctor Patel") == []
    with pytest.raises(ValueError):
        memory.correct(record.id, "Dr. Lee")


def test_expired_facts_are_not_used(memory, clock):
    memory.remember(fact("location", "Montreal", type_=MemoryType.PROFILE,
                         expires_at=clock.now + timedelta(days=3)))  # fmt: skip
    assert recall(memory, "where am I location")
    clock.now += timedelta(days=4)
    assert recall(memory, "where am I location") == []
    assert memory.records() == []


def test_relationships_link_people_and_aliases(memory):
    memory.add_entity("Alex Chen", aliases=["Alex"])
    memory.remember(fact("relationship", "project collaborator", type_=MemoryType.RELATIONSHIP,
                         subject="Alex Chen"))  # fmt: skip
    memory.remember(fact("meeting_time", "morning"))
    results = recall(memory, "", filters=RecallFilters(person="Alex"))
    assert [r.record.value for r in results] == ["project collaborator"]
    assert recall(memory, "what does Alex do")[0].record.subject == "Alex Chen"


def test_retrieval_is_relevant_filtered_and_budgeted(memory):
    memory.remember(fact("meeting_time", "morning"))
    memory.remember(fact("medication", "takes insulin daily", type_=MemoryType.PROFILE))
    memory.remember(fact("meeting_length", "30 minutes"))
    memory.remember(fact("meeting_room", "prefers video calls", source=MemorySource.INFERENCE))
    meetings = recall(memory, "schedule a meeting with Sam")
    values = [r.record.value for r in meetings]
    assert "takes insulin daily" not in values  # unrelated private facts stay out
    assert set(values) == {"morning", "30 minutes", "prefers video calls"}
    stated = recall(memory, "meeting", filters=RecallFilters(
        minimum_source=MemorySource.EXPLICIT_USER_STATEMENT))  # fmt: skip
    assert "prefers video calls" not in [r.record.value for r in stated]
    typed = recall(memory, "meeting insulin", filters=RecallFilters(types=(MemoryType.PROFILE,)))
    assert [r.record.value for r in typed] == ["takes insulin daily"]
    context = format_personal_context(meetings, budget_tokens=65)
    assert context.count("\n- ") == 1  # the budget keeps only what fits
    assert "stated by user" in context and "memory " in context


def test_who_am_i_recalls_profile_facts_but_a_passing_my_does_not(memory):
    memory.remember(fact("name", "James", type_=MemoryType.PROFILE))
    memory.remember(fact("meeting_time", "morning"))

    assert [r.record.value for r in recall(memory, "Who am I?")] == ["James"]
    assert [r.record.value for r in recall(memory, "what do you know about me")] == ["James"]
    assert recall(memory, "open my downloads folder") == []


def test_recent_and_authoritative_facts_rank_first(memory, clock):
    memory.remember(fact("lunch", "sushi on fridays", source=MemorySource.INFERENCE))
    clock.now += timedelta(days=200)
    memory.remember(fact("dinner", "sushi on sundays"))
    ranked = [r.record.key for r in recall(memory, "sushi")]
    assert ranked == ["dinner", "lunch"]
