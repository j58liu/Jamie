import json
from collections import deque
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli, eval_cli
from assistant.evaluation import EvaluationError
from assistant.llm.base import LLMResponse
from assistant.research import VERIFICATION_INSTRUCTIONS
from assistant.research_evaluation import (
    ClaimExpectation,
    ResearchEvalCase,
    load_research_cases,
    run_research_evaluation,
)
from assistant.tools import SearchResult

runner = CliRunner()


class ResearchBackend:
    def __init__(self):
        self.responses = deque(
            [
                '{"queries":["water boiling pressure"]}',
                "The sources agree on pressure effects [S1] [S2].",
                "Water boils at 100 °C at standard pressure. [S1]\n"
                "Lower pressure lowers the boiling temperature. [S2]",
            ]
        )

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        if messages[0].content == VERIFICATION_INSTRUCTIONS:
            data = json.loads(messages[-1].content)
            sources = {s["id"]: s["text"] for s in data["sources"]}
            review = {
                "claims": [
                    {
                        "index": p["index"],
                        "supported": True,
                        "evidence": [
                            {"source_id": s, "quote": sources[s][:800]} for s in p["source_ids"]
                        ],
                    }
                    for p in data["passages"]
                ]
            }
            return LLMResponse(json.dumps(review), model="test-model", output_tokens=5)
        return LLMResponse(self.responses.popleft(), model="test-model", output_tokens=5)


class FakeUrlBackend(ResearchBackend):
    def __init__(self):
        self.responses = deque(
            [
                '{"queries":["water boiling pressure"]}',
                "Cross-check [S1] [S2].",
                "Claim https://fake.example/source",
                "Claim https://fake.example/source",
            ]
        )


def research_case() -> ResearchEvalCase:
    first = "https://chemistry.example/water"
    second = "https://physics.test/pressure"
    return ResearchEvalCase(
        id="research-001",
        question="How do temperature and pressure affect boiling?",
        search_results=[
            SearchResult(title="Water", url=first, snippet=""),
            SearchResult(title="Pressure", url=second, snippet=""),
        ],
        pages={
            first: "<html><body>Water boils at 100 °C at standard pressure.</body></html>",
            second: "<html><body>Lower pressure lowers the boiling temperature.</body></html>",
        },
        expected_claims=[
            ClaimExpectation(
                answer_contains="100",
                evidence_contains="100 °C",
                supported_by=[first],
            ),
            ClaimExpectation(
                answer_contains="Lower pressure",
                evidence_contains="Lower pressure lowers",
                supported_by=[second],
            ),
        ],
        min_sources=2,
    )


def test_research_evaluation_measures_citations_sources_and_claim_support(tmp_path: Path):
    report, path = run_research_evaluation(
        ResearchBackend(),
        [research_case()],
        runs=1,
        output_dir=tmp_path,
    )
    assert path.exists()
    result = report.results[0]
    assert result.success
    assert result.source_count == 2
    assert result.citations_exist
    assert result.citations_resolve
    assert result.claims_supported
    assert result.no_fake_urls
    assert result.multiple_source_coverage
    summary = report.summary()
    assert summary["passes"] == 1
    assert summary["median_source_count"] == 2
    assert summary["output_tokens"] == 20
    assert json.loads(path.read_text())["summary"]["success"] is True


def test_research_evaluation_fails_answers_with_unverified_urls(tmp_path: Path):
    report, _ = run_research_evaluation(
        FakeUrlBackend(),
        [research_case()],
        runs=1,
        output_dir=tmp_path,
    )
    result = report.results[0]
    assert not result.success
    assert result.failure_reason == "research_error"
    assert not result.citations_exist
    assert not result.citations_resolve
    assert not result.no_fake_urls


def test_research_eval_cli_prints_all_required_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    case_path = tmp_path / "research.json"
    case_path.write_text(json.dumps([research_case().model_dump(mode="json")]))
    monkeypatch.setattr(eval_cli, "LlamaCppBackend", ResearchBackend)
    result = runner.invoke(
        cli.app,
        ["eval", "research", "--runs", "1", "--cases", str(case_path)],
    )
    assert result.exit_code == 0, result.output
    for label in (
        "Median source count",
        "Citations exist",
        "Citations resolve",
        "Claims supported",
        "No fake URLs",
        "Multiple-source coverage",
    ):
        assert label in result.stdout
    assert "[1/1]" in result.stderr


def test_research_case_loader_rejects_fake_support_url(tmp_path: Path):
    case = research_case().model_dump(mode="json")
    case["expected_claims"][0]["supported_by"] = ["https://fake.example/claim"]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]))
    with pytest.raises(EvaluationError, match="support URLs"):
        load_research_cases(path)


def test_bundled_research_cases_are_valid():
    cases = load_research_cases()
    assert len(cases) >= 2
    assert len({case.id for case in cases}) == len(cases)
