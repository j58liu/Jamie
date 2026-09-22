import json
import sqlite3
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from assistant import cli
from assistant.agent.events import AssistantFinished, TurnStarted
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.runtime import TurnResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(tmp_path / "memory.sqlite3"))


def test_version_matches_installed_package_and_does_not_load_config(tmp_path: Path):
    (tmp_path / "assistant.toml").write_text("invalid TOML", encoding="utf-8")

    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"jamie {version('j-cli')}"


def test_help_exposes_agent_ask_research_doctor_and_version():
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "agent" in result.output
    assert "doctor" in result.output
    assert "ask" in result.output
    assert "research" in result.output
    assert "jamie" in result.output
    assert "--version" in result.output


def test_doctor_checks_only_python_and_configuration(tmp_path: Path):
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Python version" in result.output
    assert "Project configuration" in result.output
    assert result.output.count("PASS") == 2
    assert "FAIL" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_doctor_accepts_explicit_config(tmp_path: Path):
    path = tmp_path / "custom.toml"
    path.write_text("color = false\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor", "--config", str(path)])

    assert result.exit_code == 0
    assert result.output.count("PASS") == 2
    assert "\x1b[" not in result.output


@pytest.mark.parametrize("content", ["color = [", 'color = "false"', "unknown = true"])
def test_doctor_reports_invalid_default_config(tmp_path: Path, content: str):
    (tmp_path / "assistant.toml").write_text(content, encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "Project configuration" in result.output
    assert "FAIL" in result.output
    assert "Traceback" not in result.output


def test_doctor_reports_missing_explicit_config():
    result = runner.invoke(cli.app, ["doctor", "--config", "missing.toml"])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert "FAIL" in result.output


@pytest.mark.parametrize(
    ("python_version", "expected_exit"),
    [((3, 11, 9), 1), ((3, 12, 0), 0), ((3, 13, 2), 0)],
)
def test_doctor_checks_python_version(
    monkeypatch: pytest.MonkeyPatch, python_version: tuple[int, int, int], expected_exit: int
):
    monkeypatch.setattr(cli, "sys", SimpleNamespace(version_info=python_version))

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == expected_exit
    assert ".".join(map(str, python_version)) in result.output
    assert "requires 3.12+" in result.output
    assert ("FAIL" in result.output) == (expected_exit == 1)


@pytest.fixture
def chat_http(monkeypatch: pytest.MonkeyPatch):
    requests: list[httpx.Request] = []
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "Hello [bold]Jamie[/bold]!"}}]
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: backend)
    return requests, response


def test_ask_prints_literal_answer_and_sends_prompt(chat_http):
    requests, _ = chat_http

    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 0
    assert result.stdout == "Hello [bold]Jamie[/bold]!\n"
    assert result.stderr == ""
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.6,
        "max_tokens": 1024,
        "stream": False,
    }


def test_ask_uses_shared_application(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class FakeApplication:
        def __init__(self, settings, *, backend_factory):
            calls.append(("init", settings.color, backend_factory))

        async def __aenter__(self):
            calls.append(("start",))
            return self

        async def run_turn(self, prompt, *, temperature, max_tokens, event_sink):
            calls.append(("run_turn", prompt, temperature, max_tokens))
            event_sink(TurnStarted(prompt))
            event_sink(AssistantFinished("Shared answer"))
            return TurnResult("Shared answer")

        async def __aexit__(self, exc_type, exc_value, traceback):
            calls.append(("close",))

    monkeypatch.setattr(cli, "JamieApplication", FakeApplication)
    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Shared answer\n"
    assert calls == [
        ("init", True, cli.LlamaCppBackend),
        ("start",),
        ("run_turn", "hello", 0.6, 1024),
        ("close",),
    ]


def test_ask_accepts_options_and_explicit_config(chat_http, tmp_path: Path):
    requests, _ = chat_http
    path = tmp_path / "custom.toml"
    path.write_text("color = false\n", encoding="utf-8")
    (tmp_path / "assistant.toml").write_text("invalid TOML", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["ask", "hello", "--temperature", "0", "--max-tokens", "64", "--config", str(path)],
    )

    assert result.exit_code == 0
    payload = json.loads(requests[0].content)
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 64
    assert "\x1b[" not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [""],
        ["   "],
        ["hello", "--temperature", "-1"],
        ["hello", "--temperature", "nan"],
        ["hello", "--temperature", "inf"],
        ["hello", "--max-tokens", "0"],
    ],
)
def test_ask_rejects_invalid_input_without_http(chat_http, arguments: list[str]):
    requests, _ = chat_http

    result = runner.invoke(cli.app, ["ask", *arguments])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert requests == []


def test_ask_rejects_invalid_config_without_http(chat_http, tmp_path: Path):
    requests, _ = chat_http
    (tmp_path / "assistant.toml").write_text("invalid TOML", encoding="utf-8")

    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert requests == []


@pytest.mark.parametrize("command", [["--help"], ["--version"], ["doctor"], ["ask", "--help"]])
def test_local_commands_do_not_contact_model(chat_http, command: list[str]):
    requests, _ = chat_http

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 0
    assert requests == []


def test_ask_reports_http_failure_on_stderr(chat_http):
    _, response = chat_http
    response.status_code = 503

    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "HTTP 503" in result.stderr
    assert "Traceback" not in result.output


def test_ask_reports_unreachable_server(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: backend)

    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Cannot reach llama.cpp" in result.stderr
    assert "127.0.0.1:8080" in result.stderr
    assert "Traceback" not in result.output


def test_research_prints_cited_answer(monkeypatch: pytest.MonkeyPatch):
    class FakeWeb:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    class FakeWorkflow:
        def __init__(self, backend, web):
            pass

        def run(self, question, *, max_sources):
            assert question == "What is supported?"
            assert max_sources == 4
            return SimpleNamespace(
                answer="Supported claim [Source](https://example.com/).",
                sources=[
                    SimpleNamespace(
                        url="https://example.com/",
                        title="Source",
                        text="Supported claim.",
                    )
                ],
            )

    monkeypatch.setattr(cli, "WebClient", FakeWeb)
    monkeypatch.setattr(cli, "ResearchWorkflow", FakeWorkflow)
    result = runner.invoke(
        cli.app,
        ["research", "What is supported?", "--max-sources", "4"],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "Supported claim [Source](https://example.com/).\n"
    with sqlite3.connect(Path.cwd() / "memory.sqlite3") as connection:
        assert connection.execute("SELECT url, title FROM sources").fetchone() == (
            "https://example.com/",
            "Source",
        )


@pytest.mark.parametrize("question", ["", "   "])
def test_research_rejects_empty_question_without_starting_web(
    question: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        cli,
        "WebClient",
        lambda: (_ for _ in ()).throw(AssertionError("web must not start")),
    )
    result = runner.invoke(cli.app, ["research", question])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_ask_warns_when_answer_reaches_token_limit(monkeypatch: pytest.MonkeyPatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Partial answer"},
                        "finish_reason": "length",
                    }
                ]
            },
        )
    )
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: LlamaCppBackend(transport=transport))

    result = runner.invoke(cli.app, ["ask", "hello"])

    assert result.exit_code == 0
    assert result.stdout == "Partial answer\n"
    assert "--max-tokens" in result.stderr
