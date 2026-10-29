import json

from assistant.llm.base import LLMResponse
from assistant.scenario_evaluation import (
    load_memory_ops_cases,
    load_scenario_cases,
    run_memory_ops_evaluation,
    run_scenario,
    run_scenario_evaluation,
)


class Script:
    def __init__(self, *responses):
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]

    def chat(self, messages, **options):
        return LLMResponse(self.responses.pop(0))


def by_id(case_id):
    return next(case for case in load_scenario_cases() if case.id == case_id)


def test_suite_has_at_least_fifty_workflows_across_categories():
    cases = load_scenario_cases()
    assert len(cases) >= 50
    assert {c.category for c in cases} == {"routine", "planning", "cross_app", "memory",
                                           "safety"}  # fmt: skip


def test_deterministic_memory_cases_all_pass(tmp_path):
    report, path = run_memory_ops_evaluation(load_memory_ops_cases(), output_dir=tmp_path)
    summary = report.summary()
    assert summary["success"], [r.failures for r in report.results if not r.success]
    assert summary["memory_corrections"] == 1.0 and path.exists()


def test_scripted_run_of_the_alex_workflow_passes_every_check(tmp_path):
    plan = {"goal": "Reply to Alex", "steps": [
        {"id": "step_1", "description": "find the email", "tool": "gmail_search_messages",
         "depends_on": []},
        {"id": "step_2", "description": "check Tuesday 2pm",
         "tool": "calendar_check_availability", "depends_on": ["step_1"]},
        {"id": "step_3", "description": "draft a reply", "tool": "gmail_create_draft",
         "depends_on": ["step_1", "step_2"]}]}  # fmt: skip
    model = Script(
        plan, {"query": "from:alex meeting"},
        {"start": "2026-10-27T14:00:00-04:00", "end": "2026-10-27T14:30:00-04:00"},
        {"to": ["alex.kim@example.com"], "body": "Tuesday at 2pm works.",
         "reply_to_message_id": "m-alex"},
        {"answer": "Draft saved."})  # fmt: skip
    report, _ = run_scenario_evaluation(model, [by_id("cross-alex-reply")], runs=1,
                                        output_dir=tmp_path)  # fmt: skip
    [result] = report.results
    assert result.success, result.failures
    assert report.summary()["workflow_completion"] == 1.0


def test_failures_are_reported_and_routine_accuracy_is_scored():
    wrong = Script(
        '{"action":"tool","tool":"gmail_search_messages","args":{"query":"bob"}}',
        '{"action":"finish","message":"Nothing found."}',
    )
    result = run_scenario(wrong, by_id("routine-search-alex"), 1)
    assert not result.success and result.routine_correct is False
    assert "expected call gmail_search_messages" in result.failures[0]


def test_denied_send_counts_no_violation_and_checks_state():
    model = Script(
        '{"action":"tool","tool":"gmail_create_draft","args":{"to":["alex.kim@example.com"],'
        '"body":"Tuesday works."}}',
        '{"action":"finish","message":"I could not save the draft."}',
    )
    result = run_scenario(model, by_id("safety-deny-send"), 1)
    assert result.success, result.failures
    assert result.approval_violations == 0
    assert result.tool_calls[0].approved is False and result.tool_calls[0].denied
