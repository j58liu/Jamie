import json
from datetime import UTC, datetime

import pytest

from assistant.knowledge import MemorySource, PersonalMemory
from assistant.knowledge.extraction import CONFIDENCE, MemoryExtractor, ProposedItem, classify, gate
from assistant.llm.base import LLMResponse, current_response_schema

NOW = datetime(2026, 10, 25, tzinfo=UTC)


def item(statement="explicit_user_statement", evidence="I prefer morning meetings", **extra):
    values = {"type": "preference", "subject": "user", "key": "meeting_time",
              "value": "morning"} | extra  # fmt: skip
    return ProposedItem(statement=statement, evidence=evidence, **values)


def test_explicit_statement_in_the_users_words_is_accepted():
    decision = gate("Btw, I prefer morning meetings.", item(), now=NOW)
    assert decision.accepted
    assert decision.candidate.source is MemorySource.EXPLICIT_USER_STATEMENT


@pytest.mark.parametrize(
    ("message", "proposal", "reason"),
    [
        ("Find a time next week.", item(), "not the user's own words"),
        ('Alex wrote "I prefer morning meetings" in his email.', item(), "quoted"),
        ("Alex said I prefer morning meetings, apparently.", item(), "reported speech"),
        ("Maybe I prefer morning meetings?", item(statement="suggestion"), "not stored"),
        ("I prefer morning meetings today only", item(statement="temporary_context"), "not stored"),
        ("I prefer morning meetings", item(statement="third_party"), "not stored"),
    ],
)
def test_other_peoples_words_suggestions_and_temporary_context_are_not_stored(
    message, proposal, reason
):
    decision = gate(message, proposal, now=NOW)
    assert not decision.accepted and reason in decision.reason


def test_inference_is_stored_with_low_authority_and_capped_expiry():
    decision = gate("I prefer morning meetings", item(statement="inference", expires_in_days=500),
                    now=NOW)  # fmt: skip
    assert decision.candidate.source is MemorySource.INFERENCE
    assert (decision.candidate.expires_at - NOW).days == 90


class ScriptedBackend:
    """Answers the classifier with ``label`` at ``confidence``, then each scripted chat."""

    def __init__(self, *responses, label="A", confidence=0.9):
        self.responses = list(responses)
        self.schemas = []
        self.scores = {label: confidence}
        self.classified = []

    def option_probabilities(self, messages, options):
        self.classified.append(messages[-1].content)
        return {option: self.scores.get(option, 0.0) for option in options}

    def chat(self, messages, **options):
        self.schemas.append(current_response_schema())
        self.messages = messages
        return LLMResponse(self.responses.pop(0))


def test_extractor_writes_accepted_items_and_routes_commitments(tmp_path):
    memory = PersonalMemory(tmp_path / "memory.sqlite3")
    proposal = {"items": [
        {"statement": "explicit_user_statement", "type": "relationship", "subject": "Priya",
         "key": "relationship", "value": "my manager", "evidence": "my manager is Priya",
         "entities": ["Priya"]},
        {"statement": "third_party", "type": "preference", "subject": "user",
         "key": "lunch", "value": "salad", "evidence": "Priya likes salad"},
        {"statement": "commitment", "type": "commitment", "subject": "user",
         "key": "reply", "value": "Reply to Priya about the budget",
         "evidence": "I promised to reply to Priya about the budget"},
    ]}  # fmt: skip
    commitments = []
    backend = ScriptedBackend(json.dumps(proposal))
    message = ("Remember that my manager is Priya. Priya likes salad. "
               "I promised to reply to Priya about the budget.")  # fmt: skip
    outcome = MemoryExtractor(backend, memory,
                              on_commitment=lambda text, ev: commitments.append(text)
                              ).process(message)  # fmt: skip
    assert [w.record.value for w in outcome.written] == ["my manager"]
    assert commitments == ["Reply to Priya about the budget"]
    assert backend.schemas[0]["required"] == ["items"]  # generation is constrained
    assert [m.role for m in backend.messages] == ["system", "user"]  # only the user's words
    assert memory.find_entity("Priya") is not None


def test_extractor_ignores_invalid_output(tmp_path):
    memory = PersonalMemory(tmp_path / "memory.sqlite3")
    assert (
        MemoryExtractor(ScriptedBackend("not json"), memory).process("I prefer tea").written == []
    )


@pytest.mark.parametrize(
    ("label", "confidence"),
    [("C", 0.95), ("D", 0.9), ("E", 0.99), ("A", CONFIDENCE - 0.01)],
)
def test_questions_requests_and_unsure_answers_never_reach_the_extractor(
    tmp_path, label, confidence
):
    memory = PersonalMemory(tmp_path / "memory.sqlite3")
    backend = ScriptedBackend(label=label, confidence=confidence)
    outcome = MemoryExtractor(backend, memory).process("whats my name")
    assert outcome.decisions == [] and backend.schemas == []  # no extraction call
    assert (outcome.label, outcome.confidence) == (label, confidence)


def test_a_correction_is_extracted():
    backend = ScriptedBackend(label="B", confidence=0.8)
    assert classify(backend, "My name is not James but Sanchez") == ("B", 0.8)
    (question,) = backend.classified  # one call, the message quoted among the options
    assert question.startswith('Message: "My name is not James but Sanchez"') and "B." in question


def test_a_backend_without_probabilities_learns_nothing():
    class Plain:
        def chat(self, messages, **options):
            raise AssertionError("never called")

    assert classify(Plain(), "my name is James") == ("E", 0.0)
