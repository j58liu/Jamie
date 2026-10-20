import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from assistant.agent.loop import AgentTimeoutError
from assistant.cancellation import CancellationToken, cancellation_scope
from assistant.config import ProjectConfig
from assistant.llm.base import LLMError, Message
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.runtime import JamieApplication
from assistant.tools.registry import create_read_only_registry


def test_connections_and_stream_loop_are_reused_across_worker_threads_and_closed():
    body = (
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    requests = []

    def handle(request):
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text=body)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
        )

    backend = LlamaCppBackend(transport=httpx.MockTransport(handle))
    with backend:
        backend.chat([Message("user", "hello")])
        sync = backend._sync_client
        backend.chat([Message("user", "again")])
        assert backend._sync_client is sync and not sync.is_closed
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index in range(2):
                assert (
                    pool.submit(
                        backend.stream_chat, [Message("user", "hello")], on_delta=lambda text: None
                    )
                    .result()
                    .content
                    == "Hello"
                )
                if index == 0:
                    client = backend._async_client
                    loop = backend._runner.get_loop()
                else:
                    assert backend._async_client is client and backend._runner.get_loop() is loop
    assert sync.is_closed and client.is_closed and loop.is_closed()
    backend.close()
    with pytest.raises(LLMError, match="closed"):
        backend.chat([Message("user", "no")])
    assert len(requests) == 4


def test_async_close_works_inside_an_application_event_loop():
    backend = LlamaCppBackend(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"Hi"},'
                    '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                ),
            )
        )
    )

    async def run():
        with cancellation_scope(CancellationToken()):
            await asyncio.to_thread(backend.chat, [Message("user", "hello")])
        await backend.aclose()

    asyncio.run(run())
    assert backend._async_client.is_closed


def test_agent_timeout_cancels_the_request_before_closing_reused_clients(tmp_path):
    async def handle(request):
        await asyncio.sleep(30)
        raise AssertionError("The timed out request was not cancelled.")

    backend = LlamaCppBackend(transport=httpx.MockTransport(handle))
    backend.context_usage = lambda messages: None

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
            backend_factory=lambda: backend,
        ) as app:
            with pytest.raises(AgentTimeoutError):
                await app.run_turn(
                    "wait", tools=create_read_only_registry(tmp_path), timeout_seconds=0.03
                )

    started = time.monotonic()
    asyncio.run(run())
    assert time.monotonic() - started < 1
    assert backend._async_client.is_closed
