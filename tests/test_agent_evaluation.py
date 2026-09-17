import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assistant import cli, eval_cli
from assistant.agent_evaluation import (
    AgentEvalCase,
    load_agent_cases,
    run_agent_evaluation,
)
from assistant.evaluation import EvaluationError
from assistant.llm.base import LLMResponse

runner = CliRunner()


class ReadThenAnswerBackend:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.calls += 1
        if messages[-1].content.startswith("Tool observation:"):
            content = '{"action":"finish","message":"README says hello world."}'
        else:
            content = '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}'
        return LLMResponse(
            content,
            model="test-model",
            finish_reason="stop",
            output_tokens=4,
        )


def read_case() -> AgentEvalCase:
    return AgentEvalCase(
        id="agent-read-001",
        prompt="What does README.md contain?",
        files={"README.md": "hello world\n"},
        expected_tool="read_file",
        expected_args={"path": "README.md"},
        tool_output_contains=["hello world"],
        answer_contains=["hello world"],
    )


def test_agent_evaluation_requires_tool_result_and_correct_answer(tmp_path: Path):
    report, path = run_agent_evaluation(
        ReadThenAnswerBackend(),
        [read_case()],
        runs=2,
        timeout_seconds=2,
        output_dir=tmp_path / "results",
    )
    assert path.exists()
    assert report.summary()["passes"] == 2
    assert report.summary()["expected_tool_calls"] == 2
    assert report.summary()["correct_tool_outputs"] == 2
    assert report.summary()["answer_checks"] == 2
    assert report.summary()["output_tokens"] == 16
    assert all(result.tool_calls[0].output == "hello world\n" for result in report.results)
    assert all(len(result.model_outputs) == 2 for result in report.results)
    saved = json.loads(path.read_text())
    assert saved["summary"]["success"] is True
    assert saved["cases"][0]["files"] == {"README.md": "hello world\n"}


def test_agent_evaluation_fails_when_model_finishes_without_tool(tmp_path: Path):
    class FinishBackend:
        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            return LLMResponse('{"action":"finish","message":"hello world"}')

    report, _ = run_agent_evaluation(
        FinishBackend(),
        [read_case()],
        runs=1,
        timeout_seconds=2,
        output_dir=tmp_path,
    )
    result = report.results[0]
    assert not result.success
    assert result.failure_reason == "expected_tool_not_called"
    assert result.answer_check


def test_agent_eval_cli_runs_complete_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "agent.json"
    cases.write_text(json.dumps([read_case().model_dump(mode="json")]))
    backend = ReadThenAnswerBackend()
    monkeypatch.setattr(eval_cli, "LlamaCppBackend", lambda: backend)

    result = runner.invoke(
        cli.app,
        ["eval", "agent", "--runs", "2", "--cases", str(cases)],
    )
    assert result.exit_code == 0, result.output
    assert backend.calls == 4
    assert "Expected tool called" in result.stdout
    assert "Correct tool output" in result.stdout
    assert "2/2" in result.stdout
    assert "[2/2]" in result.stderr
    report = json.loads(next((tmp_path / "evals/results").glob("agent-*.json")).read_text())
    assert report["summary"]["success"] is True


@pytest.mark.parametrize("name", ["../secret.txt", "/tmp/secret.txt", ""])
def test_agent_cases_reject_fixture_paths_outside_workspace(tmp_path: Path, name: str):
    path = tmp_path / "cases.json"
    case = read_case().model_dump(mode="json")
    case["files"] = {name: "secret"}
    path.write_text(json.dumps([case]))
    with pytest.raises(EvaluationError, match="Fixture paths"):
        load_agent_cases(path)


def test_agent_eval_rejects_nonfinite_timeout_before_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    backend = ReadThenAnswerBackend()
    monkeypatch.setattr(eval_cli, "LlamaCppBackend", lambda: backend)
    result = runner.invoke(cli.app, ["eval", "agent", "--timeout", "nan"])
    assert result.exit_code == 2
    assert backend.calls == 0
    assert not (tmp_path / "evals").exists()
