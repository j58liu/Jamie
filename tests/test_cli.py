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
from assistant.llm.base import LLMResponse, ServerInfo
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


def test_no_arguments_starts_the_interactive_session(monkeypatch: pytest.MonkeyPatch):
    started: list[str] = []

    async def fake_repl(settings) -> None:
        started.append("repl")

    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    result = runner.invoke(cli.app, [])

    assert result.exit_code == 0
    assert started == ["repl"]


def test_help_and_commands_do_not_start_the_interactive_session(
    monkeypatch: pytest.MonkeyPatch, doctor_server
):
    def fail_repl(settings):
        raise AssertionError("REPL must not start for an explicit command.")

    monkeypatch.setattr(cli, "_run_repl", fail_repl)

    assert runner.invoke(cli.app, ["--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["--version"]).exit_code == 0
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 0
    answer = {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer))
    monkeypatch.setattr(cli, "LlamaCppBackend", lambda: LlamaCppBackend(transport=transport))
    ask = runner.invoke(cli.app, ["ask", "hello"])
    assert ask.exit_code == 0 and ask.stdout == "Hi\n"


def test_interactive_session_builds_one_backend_and_one_runtime(monkeypatch: pytest.MonkeyPatch):
    backends, sessions = [], []

    class Backend:
        def __init__(self):
            backends.append(self)
            self.requests = 0

        server_url = "http://127.0.0.1:8080"

        def server_info(self):
            return ServerInfo("qwen", None, 8192)

        def chat(self, messages, *, temperature=0.6, max_tokens=1024, thinking=None):
            self.requests += 1
            return LLMResponse(f"answer {self.requests}")

    async def fake_run_repl(application, *, console, model):
        sessions.append((application, model))
        for prompt in ("one", "two", "three"):
            await application.run_turn(prompt)

    monkeypatch.setattr(cli, "LlamaCppBackend", Backend)
    monkeypatch.setattr(cli, "run_repl", fake_run_repl)

    result = runner.invoke(cli.app, [])

    assert result.exit_code == 0, result.output
    assert len(backends) == 1 and backends[0].requests == 3
    assert len(sessions) == 1 and sessions[0][1] == "qwen"


PROPS = {
    "default_generation_settings": {"n_ctx": 8192},
    "model_alias": "empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M",
    "model_path": None,
}


@pytest.fixture
def doctor_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A llama.cpp stand-in; tests change ``state`` to simulate server problems."""
    model = tmp_path / "models" / "Qwen3.8-9B-Q4_K_M.gguf"
    model.parent.mkdir()
    model.write_bytes(b"\0" * 1024)
    state = {"props": {**PROPS, "model_path": str(model)}, "status": 200, "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request.url.path)
        if state["status"] is None:
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(state["status"], json=state["props"])

    monkeypatch.setattr(
        cli, "LlamaCppBackend", lambda: LlamaCppBackend(transport=httpx.MockTransport(handler))
    )
    return state


def test_doctor_reports_every_prerequisite_and_ready(
    doctor_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(tmp_path / "data" / "memory.sqlite3"))
    before = sorted(tmp_path.rglob("*"))
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    for name in (
        "✓ Python",
        "✓ Jamie configuration",
        "✓ Model files",
        "✓ Inference backend",
        "✓ Application data directory",
        "– Browser runtime",
    ):
        assert any(line.startswith(name) for line in lines), name
    assert "Qwen3.8-9B-Q4_K_M.gguf (0.0 GB)" in result.output
    assert "Qwen3.8-9B-Distill · 8,192-token context" in result.output
    assert "(created on first use)" in result.output
    assert lines[-1] == "Jamie is ready."
    assert doctor_server["requests"] == ["/props"]  # never asks the model to generate
    assert sorted(tmp_path.rglob("*")) == before  # doctor changes nothing


def test_doctor_fails_when_the_inference_backend_is_down(doctor_server):
    doctor_server["status"] = None

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "✗ Inference backend" in result.output
    assert "Cannot reach llama.cpp at http://127.0.0.1:8080" in result.output
    assert "– Model files" in result.output and "needs a running inference backend" in result.output
    assert "Jamie is not ready: fix Inference backend." in result.output


def test_doctor_explains_a_model_that_is_still_loading(doctor_server):
    doctor_server["status"] = 503

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "llama.cpp is still loading the model." in result.output


def test_doctor_warns_when_model_file_is_not_visible(doctor_server):
    doctor_server["props"] = {**PROPS, "model_path": "/elsewhere/model.gguf"}

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "! Model files" in result.output
    assert "/elsewhere/model.gguf is not visible here" in result.output


def test_doctor_fails_for_an_unwritable_data_directory(
    doctor_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(locked / "jamie" / "memory.sqlite3"))
    try:
        result = runner.invoke(cli.app, ["doctor"])
    finally:
        locked.chmod(0o700)

    assert result.exit_code == 1
    assert "✗ Application data directory" in result.output
    assert "is not writable" in result.output


def test_doctor_accepts_explicit_config(doctor_server, tmp_path: Path):
    path = tmp_path / "custom.toml"
    path.write_text("color = false\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor", "--config", str(path)])

    assert result.exit_code == 0
    assert f"{path} is valid" in result.output
    assert "\x1b[" not in result.output


@pytest.mark.parametrize("content", ["color = [", 'color = "false"', "unknown = true"])
def test_doctor_reports_invalid_default_config(doctor_server, tmp_path: Path, content: str):
    (tmp_path / "assistant.toml").write_text(content, encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "✗ Jamie configuration" in result.output
    assert "– Application data directory" in result.output
    assert "Traceback" not in result.output


def test_doctor_reports_missing_explicit_config(doctor_server):
    result = runner.invoke(cli.app, ["doctor", "--config", "missing.toml"])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert "✗ Jamie configuration" in result.output


@pytest.mark.parametrize(
    ("python_version", "expected_exit"),
    [((3, 11, 9), 1), ((3, 12, 0), 0), ((3, 13, 2), 0)],
)
def test_doctor_checks_python_version(
    doctor_server,
    monkeypatch: pytest.MonkeyPatch,
    python_version: tuple[int, int, int],
    expected_exit: int,
):
    monkeypatch.setattr(
        "assistant.doctor.sys", SimpleNamespace(version_info=python_version, platform="darwin")
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == expected_exit
    assert ".".join(map(str, python_version)) in result.output
    assert "requires 3.12+" in result.output


def test_startup_explains_an_unreachable_model_and_points_to_doctor(doctor_server):
    doctor_server["status"] = None

    result = runner.invoke(cli.app, [])

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "Jamie could not start the local model.",
        "",
        "Server:",
        "llama.cpp at http://127.0.0.1:8080",
        "",
        "Reason:",
        "Cannot reach llama.cpp at http://127.0.0.1:8080. Check that the server is running.",
        "",
        "Run:",
        "jamie doctor",
    ]


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

        async def run_turn(self, prompt, *, temperature, max_tokens, thinking, event_sink):
            calls.append(("run_turn", prompt, temperature, max_tokens, thinking))
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
        ("run_turn", "hello", 0.6, 1024, None),
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
    assert "chat_template_kwargs" not in payload
    assert "\x1b[" not in result.output


@pytest.mark.parametrize(("flag", "enabled"), [("--thinking", True), ("--no-thinking", False)])
def test_ask_sends_thinking_toggle(chat_http, flag: str, enabled: bool):
    requests, _ = chat_http

    result = runner.invoke(cli.app, ["ask", "hello", flag])

    assert result.exit_code == 0, result.output
    assert json.loads(requests[0].content)["chat_template_kwargs"] == {"enable_thinking": enabled}


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


@pytest.mark.parametrize("command", [["--help"], ["--version"], ["ask", "--help"]])
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
