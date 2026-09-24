"""Synchronous llama.cpp chat requests using plain HTTP."""

import asyncio
import math
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.cancellation import CancellationToken, TurnCancelled, current_token
from assistant.llm.base import ContextUsage, LLMError, LLMResponse, Message, ServerInfo

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
        """POST one request; raise LLMError on transport failures.

        ``thinking`` toggles the chat template's reasoning; None keeps the server default.
        Inside a cancellation scope the request streams, because llama.cpp only stops
        generating for a closed connection when it next writes a token.
        """
        if not messages:
            raise ValueError("At least one message is required.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be a finite, non-negative number.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Max tokens must be a positive integer.")

        payload = {
            "messages": [asdict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}

        token = current_token()
        try:
            if token is None:
                result = self._complete(payload)
            else:
                result = asyncio.run(self._stream_until_cancelled(payload, token))
        except httpx.TimeoutException as exc:
            raise LLMError(f"Request to llama.cpp timed out at {self._url}.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"llama.cpp returned HTTP {exc.response.status_code} at {self._url}."
            ) from exc
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
        options = {**self._client_options(), "timeout": CONTEXT_TIMEOUT_SECONDS}
        try:
            with httpx.Client(**options) as client:
                response = client.get(url)
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

    def context_usage(self, messages: list[Message]) -> ContextUsage:
        """Count ``messages`` exactly as the chat template renders them."""
        options = {**self._client_options(), "timeout": CONTEXT_TIMEOUT_SECONDS}
        try:
            with httpx.Client(**options) as client:
                props = client.get(f"{self._base_url}/props")
                props.raise_for_status()
                maximum = _Props.model_validate_json(props.content)
                used = 0
                if messages:
                    template = client.post(
                        f"{self._base_url}/apply-template",
                        json={"messages": [asdict(message) for message in messages]},
                    )
                    template.raise_for_status()
                    tokens = client.post(
                        f"{self._base_url}/tokenize",
                        json={
                            "content": _Template.model_validate_json(template.content).prompt,
                            "add_special": True,
                            "parse_special": True,
                        },
                    )
                    tokens.raise_for_status()
                    used = len(_Tokens.model_validate_json(tokens.content).tokens)
        except (httpx.HTTPError, ValidationError) as exc:
            raise LLMError("llama.cpp did not report context usage.") from exc
        return ContextUsage(used, maximum.default_generation_settings.n_ctx)

    def _client_options(self) -> dict:
        return {"timeout": self._timeout, "transport": self._transport, "trust_env": False}

    def _complete(self, payload: dict) -> LLMResponse:
        with httpx.Client(**self._client_options()) as client:
            response = client.post(self._url, json=payload)
            response.raise_for_status()
        completion = _ChatCompletion.model_validate_json(response.content)
        choice = completion.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
        )

    async def _stream_until_cancelled(self, payload: dict, token: CancellationToken) -> LLMResponse:
        """Cancelling the task closes the socket, even while the prompt is processed."""
        token.raise_if_cancelled()
        async with httpx.AsyncClient(**self._client_options()) as client:
            request = asyncio.ensure_future(self._stream(client, payload))
            while not request.done():
                await asyncio.wait({request}, timeout=CANCEL_POLL_SECONDS)
                if token.cancelled and not request.done():
                    request.cancel()
                    with suppress(asyncio.CancelledError):
                        await request
                    raise TurnCancelled
            return request.result()

    async def _stream(self, client: httpx.AsyncClient, payload: dict) -> LLMResponse:
        streamed = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        model = finish_reason = None
        output_tokens = None
        first_token = None
        started = time.perf_counter()
        async with client.stream("POST", self._url, json=streamed) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                data = line.removeprefix("data:").strip()
                if not line.startswith("data:") or not data:
                    continue
                if data == "[DONE]":
                    break
                chunk = _Chunk.model_validate_json(data)
                model = chunk.model or model
                if chunk.usage is not None:
                    output_tokens = chunk.usage.completion_tokens
                for choice in chunk.choices:
                    if first_token is None and (
                        choice.delta.content or choice.delta.reasoning_content
                    ):
                        first_token = time.perf_counter() - started  # reasoning counts
                    parts.append(choice.delta.content or "")
                    finish_reason = choice.finish_reason or finish_reason
        return LLMResponse("".join(parts), model, finish_reason, output_tokens, first_token)
