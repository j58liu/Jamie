import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from assistant import cli, eval_cli
from assistant.llm.llamacpp import LlamaCppBackend

runner = CliRunner()


@pytest.fixture
def eval_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"id": "json-001", "prompt": "Return JSON.", "expected": {"action": "finish"}}])
    )
    requests = []
    responses = []

    def handler(request):
        requests.append(request)
        if responses:
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"action":"finish"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 12},
            },
        )

    monkeypatch.setattr(
        eval_cli, "LlamaCppBackend", lambda: LlamaCppBackend(transport=httpx.MockTransport(handler))
    )
    return cases, requests, responses


def test_eval_runs_repeatedly_and_reports_real_usage(eval_http, tmp_path: Path):
    cases, requests, _ = eval_http
    result = runner.invoke(cli.app, ["eval", "model", "--runs", "2", "--cases", str(cases)])
    assert result.exit_code == 0, result.output
    assert len(requests) == 2
    assert "test-model" in result.stdout
    assert "2/2" in result.stdout
    assert "100.0%" in result.stdout
    assert "24" in result.stdout
    assert "Total duration" in result.stdout
    assert "[2/2]" in result.stderr
    report = json.loads(next((tmp_path / "evals/results").glob("*.json")).read_text())
    assert report["summary"]["success"] is True
    assert report["summary"]["output_tokens"] == 24
    assert report["request_timeout_seconds"] == 600.0
    assert "600s" in result.stdout
    assert report["results"][0]["raw_output"] == '{"action":"finish"}'


def test_eval_defaults_to_five_runs_and_can_override_generation_options(eval_http):
    cases, requests, _ = eval_http
    result = runner.invoke(
        cli.app,
        ["eval", "model", "--cases", str(cases), "--temperature", "0.2", "--max-tokens", "64"],
    )
    assert result.exit_code == 0
    assert len(requests) == 5
    assert all(json.loads(request.content)["temperature"] == 0.2 for request in requests)
    assert all(json.loads(request.content)["max_tokens"] == 64 for request in requests)


def test_eval_failure_does_not_skip_later_attempts(eval_http, tmp_path: Path):
    cases, requests, responses = eval_http
    responses.append(httpx.Response(503, text="Loading"))
    result = runner.invoke(cli.app, ["eval", "model", "--runs", "2", "--cases", str(cases)])
    assert result.exit_code == 1
    assert len(requests) == 2
    assert "1/2" in result.stdout
    assert "50.0%" in result.stdout
    assert "FAIL" in result.stdout
    assert "Traceback" not in result.output
    report = json.loads(next((tmp_path / "evals/results").glob("*.json")).read_text())
    assert "HTTP 503" in report["results"][0]["error"]
    assert report["summary"]["token_usage_reported_runs"] == 1


def test_eval_handles_interrupt_with_partial_report(eval_http, tmp_path: Path):
    cases, _, responses = eval_http
    responses.append(KeyboardInterrupt())
    result = runner.invoke(cli.app, ["eval", "model", "--cases", str(cases)])
    assert result.exit_code == 130
    assert "INTERRUPTED" in result.stdout
    report = json.loads(next((tmp_path / "evals/results").glob("*.json")).read_text())
    assert report["status"] == "interrupted"
    assert report["summary"]["completed_runs"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--runs", "0"],
        ["--runs", "-1"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
        ["--temperature", "-1"],
        ["--max-tokens", "0"],
    ],
)
def test_eval_rejects_options_without_http_or_reports(eval_http, tmp_path: Path, args):
    _, requests, _ = eval_http
    result = runner.invoke(cli.app, ["eval", "model", *args])
    assert result.exit_code == 2
    assert requests == []
    assert not (tmp_path / "evals").exists()


def test_eval_rejects_bad_cases_before_requests(eval_http, tmp_path: Path):
    cases, requests, _ = eval_http
    cases.write_text("invalid JSON")
    result = runner.invoke(cli.app, ["eval", "model", "--cases", str(cases)])
    assert result.exit_code == 1
    assert "Cannot load evaluation cases" in result.stderr
    assert requests == []
    assert not (tmp_path / "evals").exists()


def test_eval_accepts_explicit_config_and_output_directory(eval_http, tmp_path: Path):
    cases, _, _ = eval_http
    (tmp_path / "assistant.toml").write_text("invalid TOML")
    config = tmp_path / "custom.toml"
    config.write_text("color = false\n")
    result = runner.invoke(
        cli.app,
        [
            "eval",
            "model",
            "--runs",
            "1",
            "--cases",
            str(cases),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "custom-results"),
        ],
    )
    assert result.exit_code == 0
    assert len(list((tmp_path / "custom-results").glob("*.json"))) == 1
    assert "\x1b[" not in result.output


@pytest.mark.parametrize(
    "args", [["--help"], ["eval", "--help"], ["eval", "model", "--help"], ["doctor"]]
)
def test_help_and_diagnostics_do_not_contact_model_or_create_reports(
    eval_http, tmp_path: Path, args
):
    _, requests, _ = eval_http
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert requests == []
    assert not (tmp_path / "evals").exists()
