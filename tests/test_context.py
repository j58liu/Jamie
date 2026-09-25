import io
import json

import httpx
import pytest
from rich.console import Console

from assistant.llm.base import ContextUsage, LLMError, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.ui.commands import ReplState, create_command_registry
from assistant.ui.status import status_line


def _server(requests: list, *, n_ctx: int = 16_384, tokens: int = 6_240, fail: str = ""):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == fail:
            return httpx.Response(500)
        if path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": n_ctx}})
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<rendered>"})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(tokens))})
        return httpx.Response(404)

    return LlamaCppBackend(transport=httpx.MockTransport(handler))


def test_backend_counts_the_templated_history_and_reads_the_maximum():
    requests = []
    history = [Message("user", "hi"), Message("assistant", "hello")]

    usage = _server(requests).context_usage(history)

    assert usage == ContextUsage(6_240, 16_384)
    assert [r.url.path for r in requests] == ["/props", "/apply-template", "/tokenize"]
    assert json.loads(requests[1].content) == {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    }
    assert json.loads(requests[2].content) == {
        "content": "<rendered>",
        "add_special": True,
        "parse_special": True,
    }
    assert requests[0].extensions["timeout"]["read"] == 2.0


def test_empty_history_uses_no_tokens_and_skips_tokenizing():
    requests = []
    assert _server(requests).context_usage([]) == ContextUsage(0, 16_384)
    assert [r.url.path for r in requests] == ["/props"]


@pytest.mark.parametrize("fail", ["/props", "/apply-template", "/tokenize"])
def test_backend_reports_measurement_failures(fail: str):
    with pytest.raises(LLMError, match="did not report context usage"):
        _server([], fail=fail).context_usage([Message("user", "hi")])


def test_usage_arithmetic():
    usage = ContextUsage(6_240, 16_384)
    assert (usage.available, usage.percent) == (10_144, 38)
    assert ContextUsage(9_000, 8_192).available == 0


class FakeApplication:
    def __init__(self, usage: ContextUsage | None = None, error: bool = False):
        self.usage, self.error, self.measured = usage, error, []

    def context_usage(self, history):
        self.measured.append(list(history))
        if self.error:
            raise LLMError("llama.cpp did not report context usage.")
        return self.usage


def _state(application) -> ReplState:
    console = Console(file=io.StringIO(), width=120, no_color=True)
    return ReplState(console, application=application, history=[Message("user", "x" * 400)])


def test_context_command_matches_the_spec_layout():
    state = _state(FakeApplication(ContextUsage(6_240, 16_384)))

    create_command_registry().dispatch(state, "/context")

    lines = state.console.file.getvalue().splitlines()
    assert lines[:6] == [
        "Context",
        "",
        "Used:        6,240 tokens",
        "Maximum:    16,384 tokens",
        "Available:  10,144 tokens",
        "Usage:         38%",
    ]
    assert "Use /compact to summarize the conversation, or /new to start over." in lines
    assert not any(line.startswith("Warning") for line in lines)
    assert status_line(state).startswith(" ctx 6.2k/16k │ ")


def test_context_warns_when_nearly_full():
    state = _state(FakeApplication(ContextUsage(7_000, 8_192)))
    create_command_registry().dispatch(state, "/context")
    assert "Warning: 85% of the context is used." in state.console.file.getvalue()
    assert status_line(state).startswith(" ctx 7.0k/8.2k │ ")


def test_context_falls_back_to_an_estimate_when_the_server_cannot_measure():
    state = _state(FakeApplication(error=True))

    create_command_registry().dispatch(state, "/context")

    output = state.console.file.getvalue()
    assert "Used:      ~100 tokens (estimated)" in output
    assert "Maximum:    unknown (llama.cpp did not report context usage.)" in output
    assert status_line(state).startswith(" ctx ~100 │ ")


def test_new_and_resume_remeasure_the_history():
    application = FakeApplication(ContextUsage(0, 8_192))
    state = _state(application)

    create_command_registry().dispatch(state, "/new")

    assert application.measured == [[]]
    assert state.context == ContextUsage(0, 8_192)
