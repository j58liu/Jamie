"""Synchronous llama.cpp chat requests using plain HTTP."""

import math
from dataclasses import asdict
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.llm.base import LLMError, LLMResponse, Message


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(strict=True)

    role: Literal["assistant"]
    content: str


class _Choice(BaseModel):
    model_config = ConfigDict(strict=True)

    message: _ResponseMessage
    finish_reason: str | None = None


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(strict=True)

    choices: list[_Choice] = Field(min_length=1)
    model: str | None = None


class LlamaCppBackend:
    """Send text chats to a running llama.cpp server's loaded model."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._timeout = httpx.Timeout(timeout, connect=5.0)
        self._transport = transport

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """POST one non-streaming request; raise LLMError on transport failures."""
        if not messages:
            raise ValueError("At least one message is required.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be a finite, non-negative number.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Max tokens must be a positive integer.")

        try:
            with httpx.Client(
                timeout=self._timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                response = client.post(
                    self._url,
                    json={
                        "messages": [asdict(message) for message in messages],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
                response.raise_for_status()
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

        try:
            completion = _ChatCompletion.model_validate_json(response.content)
        except ValidationError as exc:
            raise LLMError("llama.cpp returned an invalid chat response.") from exc

        choice = completion.choices[0]
        if not choice.message.content.strip():
            raise LLMError(
                "llama.cpp returned no answer text. Try increasing --max-tokens "
                "if the model used its token budget for reasoning."
            )
        return LLMResponse(
            content=choice.message.content,
            model=completion.model,
            finish_reason=choice.finish_reason,
        )
