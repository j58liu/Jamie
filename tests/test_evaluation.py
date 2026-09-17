import json
from pathlib import Path

import pytest

from assistant import evaluation
from assistant.agent.protocol import ACTION_PROTOCOL_INSTRUCTIONS
from assistant.evaluation import (
    EvalCase,
    EvaluationError,
    load_cases,
    run_evaluation,
    score_response,
)
from assistant.llm.base import LLMError, LLMResponse, Message


class FakeBackend:
    def __init__(self, responses: list[LLMResponse | BaseException]):
        self.responses = iter(responses)
        self.requests: list[tuple[list[Message], float, int]] = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.requests.append((messages, temperature, max_tokens))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def json_case():
    return EvalCase(
        id="json-001",
        prompt="Return a JSON object with action=finish.",
        expected={"action": "finish"},
    )


def text_case():
    return EvalCase(id="text-001", prompt="Reply READY.", format="text", expected="READY")


def protocol_case():
    return EvalCase(
        id="protocol-001",
        prompt="Finish with Done.",
        format="protocol",
        expected={"action": "finish", "message": "Done."},
    )


def test_bundled_cases_are_complete_and_unique():
    cases = load_cases()
    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    assert sum(case.format == "json" for case in cases) == 14
    assert sum(case.format == "protocol" for case in cases) == 10
    assert cases[0].prompt == "Return a JSON object with action=finish and message=hello."


@pytest.mark.parametrize(
    "contents",
    [
        "{}",
        "[]",
        "null",
        "[{}]",
        '[{"id": "x", "prompt": "hi"}]',
        '[{"id": "x", "prompt": " ", "expected": {}}]',
        '[{"id": "x", "prompt": "hi", "expected": NaN}]',
        '[{"id": "x", "prompt": "hi", "expected": {}, "unknown": true}]',
        '[{"id": "x", "prompt": "hi", "format": "text", "expected": {}}]',
        '[{"id": "x", "prompt": "hi", "format": "protocol", "expected": {}}]',
    ],
)
def test_invalid_case_files_are_rejected(tmp_path: Path, contents: str):
    path = tmp_path / "cases.json"
    path.write_text(contents)
    with pytest.raises(EvaluationError):
        load_cases(path)


def test_missing_or_empty_case_sources_are_rejected(tmp_path: Path):
    with pytest.raises(EvaluationError):
        load_cases(tmp_path)
    with pytest.raises(EvaluationError):
        load_cases(tmp_path / "absent.json")


def test_duplicate_case_ids_across_files_are_rejected(tmp_path: Path):
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps([json_case().model_dump()]))
    with pytest.raises(EvaluationError, match="unique"):
        load_cases(tmp_path)


@pytest.mark.parametrize(
    ("content", "parsed", "following"),
    [
        ('{"action":"finish"}', True, True),
        (' \n{"action":"finish","message":"hello"}\n', True, True),
        ('{"action":"wait"}', True, False),
        ("{}", True, False),
        ("[]", True, False),
        ("null", True, False),
        ('```json\n{"action":"finish"}\n```', False, False),
        ('Here is JSON: {"action":"finish"}', False, False),
        ('{"action":"finish"} extra', False, False),
        ('{"action":"finish","action":"wait"}', False, False),
        ('{"action":"finish","value":NaN}', False, False),
        ('{"action":"finish","value":Infinity}', False, False),
        ('{"action":"finish","value":1e9999}', False, False),
    ],
)
def test_json_scoring_distinguishes_parse_and_instruction_success(content, parsed, following):
    result = score_response(json_case(), LLMResponse(content))
    assert result[:3] == (parsed, following, parsed and following)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"count": True}, {"count": 1}),
        ({"count": 1.0}, {"count": 1}),
        ({"count": "1"}, {"count": 1}),
        ({"steps": ["b", "a"]}, {"steps": ["a", "b"]}),
        ({"steps": ["a", "b", "c"]}, {"steps": ["a", "b"]}),
        ({"result": {"ok": False}}, {"result": {"ok": True}}),
    ],
)
def test_json_expectations_check_types_nested_values_and_order(actual, expected):
    case = json_case().model_copy(update={"expected": expected})
    assert score_response(case, LLMResponse(json.dumps(actual)))[:3] == (True, False, False)


def test_text_scoring_ignores_only_surrounding_whitespace():
    assert score_response(text_case(), LLMResponse(" \nREADY\n"))[:3] == (None, True, True)
    assert score_response(text_case(), LLMResponse("**READY**"))[:3] == (None, False, False)
    assert score_response(text_case(), LLMResponse("ready"))[:3] == (None, False, False)


@pytest.mark.parametrize(
    ("content", "parsed", "following"),
    [
        ('{"action":"finish","message":"Done."}', True, True),
        ('```json\n{"action":"finish","message":"Done."}\n```', True, True),
        (
            '<think>Use finish.</think>\n{"action":"finish","message":"Done."}',
            True,
            True,
        ),
        ('{"action":"finish","message":"Different."}', True, False),
        ('{"action":"wait","message":"Done."}', False, False),
        ('{"action":"finish"}', False, False),
        ("not JSON", False, False),
    ],
)
def test_protocol_scoring_distinguishes_parse_and_instruction_success(content, parsed, following):
    result = score_response(protocol_case(), LLMResponse(content))
    assert result[:3] == (parsed, following, parsed and following)
    assert result[3] == (
        None if parsed and following else "expectation_mismatch" if parsed else "invalid_protocol"
    )


def test_protocol_expectations_require_exact_arguments():
    response = LLMResponse('{"action":"finish","message":"Done.","extra":"rejected by parser"}')
    assert score_response(protocol_case(), response)[3] == "invalid_protocol"

    tool_case = EvalCase(
        id="protocol-tool",
        prompt="Read a file.",
        format="protocol",
        expected={"action": "tool", "tool": "read_file", "args": {"path": "README.md"}},
    )
    response = LLMResponse(
        '{"action":"tool","tool":"read_file","args":{"path":"README.md","line":1}}'
    )
    assert score_response(tool_case, response)[3] == "expectation_mismatch"


def test_protocol_summary_reports_parse_success_separately(tmp_path: Path):
    backend = FakeBackend(
        [
            LLMResponse('```json\n{"action":"finish","message":"Done."}\n```'),
            LLMResponse('{"action":"unknown"}'),
        ]
    )
    report, _ = run_evaluation(backend, [protocol_case()], runs=2, output_dir=tmp_path)
    summary = report.summary()
    assert summary["json_attempts"] == 0
    assert summary["parse_success_rate"] is None
    assert summary["protocol_attempts"] == 2
    assert summary["protocol_parse_successes"] == 1
    assert summary["protocol_parse_success_rate"] == 0.5
    expected_messages = [
        Message("system", ACTION_PROTOCOL_INSTRUCTIONS),
        Message("user", protocol_case().prompt),
    ]
    assert backend.requests == [(expected_messages, 0.6, 1024)] * 2


def test_truncated_matching_response_still_fails():
    assert score_response(
        json_case(), LLMResponse('{"action":"finish"}', finish_reason="length")
    ) == (True, True, False, "token_limit")


def test_repeated_runs_save_exact_prompts_metrics_and_raw_outputs(tmp_path: Path, monkeypatch):
    ticks = iter([0, 1, 3, 4, 5, 9, 10, 11, 17, 18, 19, 27, 28, 30])
    monkeypatch.setattr(evaluation, "perf_counter", lambda: next(ticks))
    cases = [json_case(), text_case()]
    backend = FakeBackend(
        [
            LLMResponse('{"action":"finish"}', "test-model", "stop", 10),
            LLMResponse("READY", "test-model", "stop", 20),
            LLMResponse('```json\n{"action":"finish"}\n```', "test-model", "stop", 30),
            LLMResponse("not ready", "test-model", "stop", 40),
        ]
    )
    report, path = run_evaluation(backend, cases, runs=2, output_dir=tmp_path)
    assert [r.case_id for r in report.results] == ["json-001", "text-001", "json-001", "text-001"]
    assert [r.run for r in report.results] == [1, 1, 2, 2]
    assert backend.requests == [([Message("user", case.prompt)], 0.6, 1024) for case in cases] * 2
    summary = report.summary()
    assert summary["passes"] == 2
    assert summary["failures"] == 2
    assert summary["parse_success_rate"] == 0.5
    assert summary["instruction_following_rate"] == 0.5
    assert summary["median_latency_seconds"] == 5.0
    assert summary["output_tokens"] == 100
    assert summary["token_usage_reported_runs"] == 4
    assert summary["total_duration_seconds"] == 30
    assert summary["success"] is False
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 2
    assert saved["summary"] == summary
    assert saved["status"] == "completed"
    assert saved["cases"] == [case.model_dump() for case in cases]
    assert saved["results"][2]["raw_output"].startswith("```json")
    assert len(saved["cases_sha256"]) == 64


def test_errors_count_as_failures_and_do_not_stop_remaining_runs(tmp_path: Path):
    backend = FakeBackend(
        [LLMError("timeout"), LLMResponse('{"action":"finish"}', output_tokens=0)]
    )
    report, _ = run_evaluation(backend, [json_case()], runs=2, output_dir=tmp_path)
    assert len(backend.requests) == 2
    assert report.results[0].error == "timeout"
    assert report.results[0].raw_output is None
    assert report.results[0].parse_success is False
    assert report.results[0].failure_reason == "request_error"
    assert report.summary()["passes"] == 1
    assert report.summary()["parse_success_rate"] == 0.5
    assert report.summary()["output_tokens"] == 0
    assert report.summary()["token_usage_reported_runs"] == 1


def test_empty_response_error_keeps_usage_metadata(tmp_path: Path):
    response = LLMResponse("", "model", "length", 1024)
    backend = FakeBackend([LLMError("no answer", response=response)])
    report, _ = run_evaluation(backend, [json_case()], runs=1, output_dir=tmp_path)
    result = report.results[0]
    assert result.raw_output == ""
    assert result.output_tokens == 1024
    assert result.finish_reason == "length"
    assert result.model == "model"
    assert not result.success


def test_interrupt_preserves_completed_attempts_and_marks_partial_report(tmp_path: Path):
    backend = FakeBackend([LLMResponse("READY"), KeyboardInterrupt()])
    report, path = run_evaluation(backend, [text_case()], runs=3, output_dir=tmp_path)
    assert report.status == "interrupted"
    assert len(report.results) == 1
    assert report.summary()["planned_runs"] == 3
    assert report.summary()["success"] is False
    saved = json.loads(path.read_text())
    assert saved["status"] == "interrupted"
    assert saved["summary"]["output_tokens"] is None
    assert saved["summary"]["parse_success_rate"] is None


def test_report_checkpoint_exists_before_next_request(tmp_path: Path):
    class CheckpointBackend:
        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            checkpoint = json.loads(next(tmp_path.glob("model-*.json")).read_text())
            assert checkpoint["status"] == "running"
            assert len(checkpoint["results"]) == calls.pop(0)
            return LLMResponse("READY")

    calls = [0, 1]
    run_evaluation(CheckpointBackend(), [text_case()], runs=2, output_dir=tmp_path)
    assert calls == []


def test_unwritable_output_fails_before_contacting_model(tmp_path: Path, monkeypatch):
    backend = FakeBackend([])
    monkeypatch.setattr(
        evaluation.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(EvaluationError, match="Cannot save"):
        run_evaluation(backend, [text_case()], runs=1, output_dir=tmp_path)
    assert backend.requests == []


def test_separate_invocations_keep_prior_results_and_stable_case_hash(tmp_path: Path):
    backend = FakeBackend([LLMResponse("READY"), LLMResponse("READY")])
    first, first_path = run_evaluation(backend, [text_case()], runs=1, output_dir=tmp_path)
    second, second_path = run_evaluation(backend, [text_case()], runs=1, output_dir=tmp_path)
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
    assert first.cases_sha256 == second.cases_sha256


@pytest.mark.parametrize(
    "options", [{"runs": 0}, {"runs": True}, {"temperature": float("nan")}, {"max_tokens": 0}]
)
def test_invalid_run_options_do_not_create_output(tmp_path: Path, options):
    output = tmp_path / "results"
    with pytest.raises(EvaluationError):
        run_evaluation(FakeBackend([]), [text_case()], output_dir=output, **options)
    assert not output.exists()
