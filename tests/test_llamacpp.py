import json

import httpx
import pytest

from assistant.cancellation import CancellationToken, cancellation_scope
from assistant.llm.base import LLMBackend, LLMError, LLMResponse, Message, constrained_to
from assistant.llm.llamacpp import LlamaCppBackend


def completion(content: str = "Hello!") -> dict:
    return {
        "model": "local-qwen",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def test_chat_posts_messages_and_reads_response():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = completion("Hi, café!\nSecond line.")
        payload["choices"][0]["message"]["reasoning_content"] = "Internal reasoning"
        payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 8}
        return httpx.Response(200, json=payload)

    backend: LLMBackend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    result = backend.chat(
        [
            Message(role="system", content="Be brief."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="café?"),
        ]
    )

    assert result == LLMResponse("Hi, café!\nSecond line.", "local-qwen", "stop", 8)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "café?"},
        ],
        "temperature": 0.6,
        "max_tokens": 1024,
        "stream": False,
    }
    assert request.extensions["timeout"]["connect"] == 5.0
    assert request.extensions["timeout"]["read"] == 600.0


def test_chat_accepts_server_and_generation_options():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:9999/v1/chat/completions"
        assert json.loads(request.content)["temperature"] == 0.0
        assert json.loads(request.content)["max_tokens"] == 32
        assert request.extensions["timeout"]["read"] == 30.0
        return httpx.Response(200, json=completion())

    backend = LlamaCppBackend(
        "http://localhost:9999/", timeout=30.0, transport=httpx.MockTransport(handler)
    )

    assert backend.chat([Message("user", "Hi")], temperature=0.0, max_tokens=32).content == "Hello!"


@pytest.mark.parametrize("thinking", [True, False])
def test_chat_sends_thinking_toggle_only_when_set(thinking: bool):
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    backend.chat([Message("user", "Hi")], thinking=thinking)
    backend.chat([Message("user", "Hi")])

    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": thinking}
    assert "chat_template_kwargs" not in payloads[1]


@pytest.mark.parametrize("status", [301, 400, 404, 429, 500, 503])
def test_http_errors_are_reported_without_retrying(status: int):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="Server error")

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError, match=f"HTTP {status}"):
        backend.chat([Message("user", "Hi")])
    assert len(requests) == 1


@pytest.mark.parametrize("streamed", [False, True])
def test_a_request_longer_than_the_context_says_so(streamed: bool):
    error = {
        "code": 400,
        "message": "request (9012 tokens) exceeds the available context size (8192 tokens)",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 9012,
        "n_ctx": 8192,
    }
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"error": error}))
    )

    # A cancellation token makes the backend stream, which reads errors differently.
    with cancellation_scope(CancellationToken() if streamed else None):
        with pytest.raises(LLMError) as raised:
            backend.chat([Message("user", "Hi")])

    assert str(raised.value) == (
        "The request needs 9,012 tokens, more than the model's 8,192-token context. "
        "Shorten the conversation or the request."
    )


def test_other_server_errors_keep_llamacpp_reason():
    body = {"error": {"code": 500, "message": "slot unavailable", "type": "server_error"}}
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(500, json=body))
    )

    with pytest.raises(LLMError, match="HTTP 500: slot unavailable"):
        backend.chat([Message("user", "Hi")])


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError, "Cannot reach llama.cpp"),
        (httpx.RemoteProtocolError, "Cannot reach llama.cpp"),
        (httpx.ConnectTimeout, "timed out"),
        (httpx.ReadTimeout, "timed out"),
    ],
)
def test_transport_errors_are_reported(error: type[httpx.RequestError], message: str):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("Request failed", request=request)

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError, match=message):
        backend.chat([Message("user", "Hi")])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"role": "assistant"}}]},
        {"choices": [{"message": {"role": "assistant", "content": None}}]},
        {"choices": [{"message": {"role": "assistant", "content": 123}}]},
        {"choices": [{"message": {"role": "user", "content": "Wrong role"}}]},
    ],
)
def test_malformed_chat_responses_are_rejected(payload: object):
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )

    with pytest.raises(LLMError, match="invalid chat response"):
        backend.chat([Message("user", "Hi")])


@pytest.mark.parametrize("content", [b"not JSON", b"\xff", b"<html>Error</html>"])
def test_invalid_json_is_rejected(content: bytes):
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    )

    with pytest.raises(LLMError, match="invalid chat response"):
        backend.chat([Message("user", "Hi")])


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_answers_explain_token_budget(content: str):
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=completion(content)))
    )

    with pytest.raises(LLMError, match="no answer text.*--max-tokens"):
        backend.chat([Message("user", "Hi")])


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 10}])
def test_missing_completion_usage_is_unknown(usage):
    payload = completion()
    payload["usage"] = usage
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    assert backend.chat([Message("user", "Hi")]).output_tokens is None


@pytest.mark.parametrize("tokens", [-1, True, "12", 1.5])
def test_invalid_completion_usage_is_rejected(tokens):
    payload = completion()
    payload["usage"] = {"completion_tokens": tokens}
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with pytest.raises(LLMError, match="invalid chat response"):
        backend.chat([Message("user", "Hi")])


def test_empty_answer_preserves_model_finish_reason_and_token_usage():
    payload = completion("")
    payload["choices"][0]["finish_reason"] = "length"
    payload["usage"] = {"completion_tokens": 1024}
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with pytest.raises(LLMError) as error:
        backend.chat([Message("user", "Hi")])
    assert error.value.response == LLMResponse("", "local-qwen", "length", 1024)


def test_a_response_schema_constrains_only_requests_inside_its_scope():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"action":"finish","message":"ok"}'))

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    schema = {"type": "object"}
    with constrained_to(schema):
        backend.chat([Message("user", "Hi")])
    backend.chat([Message("user", "Hi")])

    assert payloads[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }
    assert "response_format" not in payloads[1]
