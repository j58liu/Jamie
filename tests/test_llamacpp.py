import json

import httpx
import pytest

from assistant.llm.base import LLMBackend, LLMError, LLMResponse, Message
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

    assert result == LLMResponse("Hi, café!\nSecond line.", "local-qwen", "stop")
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
    assert request.extensions["timeout"]["read"] == 120.0


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


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": 1.5},
        {"max_tokens": True},
    ],
)
def test_invalid_options_fail_before_http(options: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid input must not send an HTTP request")

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError):
        backend.chat([Message("user", "Hi")], **options)


def test_empty_conversation_fails_before_http():
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Empty conversations must not send an HTTP request")

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="At least one message"):
        backend.chat([])
