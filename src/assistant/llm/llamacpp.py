"""Synchronous llama.cpp chat requests using plain HTTP."""

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from contextvars import copy_context
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.cancellation import CancellationToken, TurnCancelled, current_token
from assistant.llm.answer import AnswerStream, answer_text
from assistant.llm.base import (
    ContextUsage,
    LLMError,
    LLMResponse,
    Message,
    ServerInfo,
    current_response_schema,
)

CANCEL_POLL_SECONDS = 0.1
CONTEXT_TIMEOUT_SECONDS = 2.0


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    role: Literal["assistant"]
    content: str


class _Choice(BaseModel):
    model_config = ConfigDict(strict=True)

    message: _ResponseMessage
    finish_reason: str | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(strict=True)

    completion_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(strict=True)

    choices: list[_Choice] = Field(min_length=1)
    model: str | None = None
    usage: _Usage | None = None


class _Delta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class _StreamChoice(BaseModel):
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _Chunk(BaseModel):
    choices: list[_StreamChoice] = Field(default_factory=list)
    model: str | None = None
    usage: _Usage | None = None


class _Settings(BaseModel):
    n_ctx: int = Field(ge=1)


class _Props(BaseModel):
    default_generation_settings: _Settings
    model_alias: str | None = None
    model_path: str | None = None


class _Template(BaseModel):
    prompt: str


class _Tokens(BaseModel):
    tokens: list[int]


def _status_error(response: httpx.Response, url: str) -> str:
    """Prefer llama.cpp's own reason, such as a request longer than the context."""
    try:
        error = response.json().get("error") or {}
        needed, limit = error.get("n_prompt_tokens"), error.get("n_ctx")
        if error.get("type") == "exceed_context_size_error" and needed and limit:
            return (
                f"The request needs {needed:,} tokens, more than the model's {limit:,}-token "
                "context. Shorten the conversation or the request."
            )
        if error.get("message"):
            return f"llama.cpp returned HTTP {response.status_code}: {error['message']}"
    except (ValueError, AttributeError, httpx.ResponseNotRead):
        pass
    return f"llama.cpp returned HTTP {response.status_code} at {url}."


class LlamaCppBackend:
    """Send text chats to a running llama.cpp server's loaded model."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/v1/chat/completions"
        self._timeout = httpx.Timeout(timeout, connect=5.0)
        self._transport = transport
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._runner: asyncio.Runner | None = None
        self._stream_lock = RLock()
        self._closed = False

    def close(self) -> None:
        """Close both connection pools and the streaming event loop once."""
        with self._stream_lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._sync_client is not None:
                    self._sync_client.close()
                if self._runner is not None and self._async_client is not None:
                    self._runner.run(self._async_client.aclose())
            finally:
                if self._runner is not None:
                    self._runner.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _client(self) -> httpx.Client:
        with self._stream_lock:
            if self._closed:
                raise LLMError("The llama.cpp backend is closed.")
            if self._sync_client is None:
                self._sync_client = httpx.Client(**self._client_options())
            return self._sync_client

    def _run_stream(self, payload, token, on_delta) -> LLMResponse:
        with self._stream_lock:
            if self._closed:
                raise LLMError("The llama.cpp backend is closed.")
            if self._runner is None:
                self._runner = asyncio.Runner()
            return self._runner.run(
                self._stream_until_cancelled(payload, token, on_delta), context=copy_context()
            )

    @property
    def read_timeout_seconds(self) -> float | None:
        """Return the configured read timeout for evaluation reports."""
        return self._timeout.read

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        thinking: bool | None = None,
    ) -> LLMResponse:
        return self._chat(
            messages, temperature=temperature, max_tokens=max_tokens, thinking=thinking
        )

    def stream_chat(
        self,
        messages: list[Message],
        *,
        on_delta: Callable[[str], None],
        temperature: float = 0.6,
        max_tokens: int = 1024,
        thinking: bool | None = None,
    ) -> LLMResponse:
        """Stream visible direct-chat text; action callers continue using buffered chat."""
        return self._chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            on_delta=on_delta,
        )

    def _chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool | None,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """POST one request; raise LLMError on transport failures.

        ``thinking`` toggles the chat template's reasoning; None keeps the server default.
        Inside a cancellation scope the request streams, because llama.cpp only stops
        generating for a closed connection when it next writes a token.
        """
        payload = {
            "messages": [asdict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        schema = current_response_schema()
        if schema is not None:  # llama.cpp turns the schema into a grammar for sampling
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }

        token = current_token()
        try:
            if token is None and on_delta is None:
                result = self._complete(payload)
            else:
                result = self._run_stream(payload, token or CancellationToken(), on_delta)
        except httpx.TimeoutException as exc:
            raise LLMError(f"Request to llama.cpp timed out at {self._url}.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(_status_error(exc.response, self._url)) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Cannot reach llama.cpp at {self._url}. Check that the server is running."
            ) from exc
        except ValidationError as exc:
            raise LLMError("llama.cpp returned an invalid chat response.") from exc

        if not result.content.strip():
            raise LLMError(
                "llama.cpp returned no answer text. Try increasing --max-tokens "
                "if the model used its token budget for reasoning.",
                response=result,
            )
        return result

    def server_info(self) -> ServerInfo:
        """Confirm the server is up with a loaded model; raise LLMError with the reason."""
        url = f"{self._base_url}/props"
        try:
            response = self._client().get(url, timeout=CONTEXT_TIMEOUT_SECONDS)
            response.raise_for_status()
            props = _Props.model_validate_json(response.content)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 503:
                raise LLMError("llama.cpp is still loading the model. Try again shortly.") from exc
            raise LLMError(f"llama.cpp returned HTTP {exc.response.status_code} at {url}.") from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Cannot reach llama.cpp at {self._base_url}. Check that the server is running."
            ) from exc
        except ValidationError as exc:
            raise LLMError(f"{url} did not describe a loaded model.") from exc
        path = props.model_path or None
        model = props.model_alias or (Path(path).name if path else "unknown model")
        return ServerInfo(model, path, props.default_generation_settings.n_ctx)

    @property
    def server_url(self) -> str:
        return self._base_url

    def context_usage(
        self, messages: list[Message], *, thinking: bool | None = None
    ) -> ContextUsage:
        """Count ``messages`` exactly as the chat template renders them."""
        try:
            client = self._client()
            props = client.get(f"{self._base_url}/props", timeout=CONTEXT_TIMEOUT_SECONDS)
            props.raise_for_status()
            maximum = _Props.model_validate_json(props.content)
            used = 0
            if messages:
                template_payload = {
                    "messages": [asdict(message) for message in messages],
                    "add_generation_prompt": True,
                }
                if thinking is not None:
                    template_payload["chat_template_kwargs"] = {"enable_thinking": thinking}
                template = client.post(
                    f"{self._base_url}/apply-template",
                    json=template_payload,
                    timeout=CONTEXT_TIMEOUT_SECONDS,
                )
                template.raise_for_status()
                tokens = client.post(
                    f"{self._base_url}/tokenize",
                    json={
                        "content": _Template.model_validate_json(template.content).prompt,
                        "add_special": True,
                        "parse_special": True,
                    },
                    timeout=CONTEXT_TIMEOUT_SECONDS,
                )
                tokens.raise_for_status()
                used = len(_Tokens.model_validate_json(tokens.content).tokens)
        except (httpx.HTTPError, ValidationError) as exc:
            raise LLMError("llama.cpp did not report context usage.") from exc
        return ContextUsage(used, maximum.default_generation_settings.n_ctx)

    def _client_options(self) -> dict:
        return {"timeout": self._timeout, "transport": self._transport, "trust_env": False}

    def _complete(self, payload: dict) -> LLMResponse:
        response = self._client().post(self._url, json=payload)
        response.raise_for_status()
        completion = _ChatCompletion.model_validate_json(response.content)
        choice = completion.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
        )

    async def _stream_until_cancelled(
        self,
        payload: dict,
        token: CancellationToken,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Cancelling the task closes the socket, even while the prompt is processed."""
        token.raise_if_cancelled()
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_options())
        request = asyncio.ensure_future(self._stream(self._async_client, payload, on_delta))
        while not request.done():
            await asyncio.wait({request}, timeout=CANCEL_POLL_SECONDS)
            if token.cancelled and not request.done():
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
                raise TurnCancelled
        try:
            return request.result()
        finally:
            # Allow response iterator cleanup to finish while retaining the client and loop.
            await asyncio.sleep(0)

    async def _stream(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        streamed = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        model = finish_reason = None
        output_tokens = None
        input_tokens = None
        first_token = None
        started = time.perf_counter()
        visible = AnswerStream()
        complete = False
        async with client.stream("POST", self._url, json=streamed) as response:
            if response.is_error:
                await response.aread()  # keep llama.cpp's reason for the error message
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                # Some compatible servers return a complete response even when asked to stream.
                await response.aread()
                completion = _ChatCompletion.model_validate_json(response.content)
                choice = completion.choices[0]
                if on_delta is not None:
                    on_delta(answer_text(choice.message.content))
                return LLMResponse(
                    choice.message.content,
                    completion.model,
                    choice.finish_reason,
                    completion.usage.completion_tokens if completion.usage else None,
                    input_tokens=completion.usage.prompt_tokens if completion.usage else None,
                )
            async for line in response.aiter_lines():
                data = line.removeprefix("data:").strip()
                if not line.startswith("data:") or not data:
                    continue
                if data == "[DONE]":
                    complete = True
                    break
                chunk = _Chunk.model_validate_json(data)
                model = chunk.model or model
                if chunk.usage is not None:
                    output_tokens = chunk.usage.completion_tokens
                    input_tokens = chunk.usage.prompt_tokens
                for choice in chunk.choices:
                    if first_token is None and (
                        choice.delta.content or choice.delta.reasoning_content
                    ):
                        first_token = time.perf_counter() - started  # reasoning counts
                    parts.append(choice.delta.content or "")
                    if on_delta is not None and (text := visible.feed(choice.delta.content or "")):
                        on_delta(text)
                    finish_reason = choice.finish_reason or finish_reason
        if not complete or finish_reason is None:
            raise LLMError("llama.cpp returned an incomplete streamed answer.")
        if on_delta is not None and (text := visible.finish()):
            on_delta(text)
        return LLMResponse(
            "".join(parts), model, finish_reason, output_tokens, first_token, input_tokens
        )
