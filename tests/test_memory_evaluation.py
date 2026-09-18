import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.evaluation import EvaluationError
from assistant.memory_evaluation import (
    MemoryEvalCase,
    load_memory_cases,
    run_memory_evaluation,
)

runner = CliRunner()


def memory_case() -> MemoryEvalCase:
    return MemoryEvalCase(
        id="memory-001",
        memories=[
            "We chose SQLite as the database library.",
            "Safari is the preferred browser.",
        ],
        query="What database library did we choose?",
        expected_contains="SQLite",
        top_k=2,
    )


def test_memory_evaluation_reports_recall_and_rank(tmp_path: Path):
    report, path = run_memory_evaluation([memory_case()], output_dir=tmp_path)
    assert path.exists()
    assert report.results[0].expected_rank == 1
    assert report.results[0].success
    assert report.summary()["recall_at_k"] == 1.0
    assert report.summary()["mean_reciprocal_rank"] == 1.0
    assert json.loads(path.read_text())["summary"]["success"] is True


def test_memory_eval_cli_does_not_call_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps([memory_case().model_dump(mode="json")]))
    result = runner.invoke(cli.app, ["eval", "memory", "--cases", str(case_path)])
    assert result.exit_code == 0, result.output
    assert "Recall@k" in result.stdout
    assert "Mean reciprocal rank" in result.stdout
    assert "100.0%" in result.stdout
    assert len(list((tmp_path / "evals/results").glob("memory-*.json"))) == 1


def test_memory_case_requires_unique_expected_match(tmp_path: Path):
    case = memory_case().model_dump(mode="json")
    case["memories"][1] = "SQLite appears here too."
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([case]))
    with pytest.raises(EvaluationError, match="exactly one"):
        load_memory_cases(path)


def test_bundled_memory_cases_pass_fts_baseline(tmp_path: Path):
    cases = load_memory_cases()
    report, _ = run_memory_evaluation(cases, output_dir=tmp_path)
    assert len(cases) == 5
    assert report.summary()["success"] is True
