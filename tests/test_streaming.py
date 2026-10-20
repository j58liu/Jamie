import asyncio
import io
import json
import threading

import httpx
import pytest
from rich.console import Console

from assistant.agent.events import AssistantDelta, AssistantFinished, TurnFailed
from assistant.cancellation import CancellationToken, TurnCancelled
from assistant.config import ProjectConfig
from assistant.llm.base import ContextUsage, LLMError
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory import MemoryStore
from assistant.runtime import JamieApplication
from assistant.ui.renderer import OneShotRenderer


def chunk(content=None, **choice):
    delta = {} if content is None else {"content": content}
    return ("data: " + json.dumps({"choices": [{"delta": delta, **choice}]}) + "\n\n").encode()


@pytest.mark.parametrize("cancel", [False, True])
def test_visible_deltas_arrive_before_completion_on_the_caller_thread(tmp_path, cancel):
    released = threading.Event()
    token = CancellationToken()
    caller = threading.get_ident()
    events = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for text in ("<thi", "nk>private", " reasoning</th", "ink>\n", "Hel"):
                yield chunk(text)
            assert await asyncio.to_thread(released.wait, 1), "the first delta was buffered"
            if cancel:
                while True:
                    await asyncio.sleep(0.01)
            yield chunk("lo", finish_reason="stop")
            yield b"data: [DONE]\n\n"

    def receive(event):
        assert threading.get_ident() == caller
        events.append(event)
        if isinstance(event, AssistantDelta):
            released.set()
            if cancel:
                token.cancel()

    backend = LlamaCppBackend(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    )
    database = tmp_path / "memory.sqlite3"
    backend.context_usage = lambda messages: ContextUsage(16, 8192)

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(database)), backend_factory=lambda: backend
        ) as app:
            return await app.run_turn("hello", cancel=token, event_sink=receive)

    if cancel:
        with pytest.raises(TurnCancelled):
            asyncio.run(run())
        assert isinstance(events[-1], TurnFailed)
        assert MemoryStore(database).search_memories("hello") == []
    else:
        assert asyncio.run(run()).message == "Hello"
        assert isinstance(events[-1], AssistantFinished)
        assert "Hello" in MemoryStore(database).search_memories("hello")[0].content
    assert "private" not in repr(events)
    assert "<think>" not in repr(events)
    assert "".join(e.text for e in events if isinstance(e, AssistantDelta)) == (
        "Hel" if cancel else "Hello"
    )


def test_incomplete_stream_is_not_saved_and_next_turn_can_succeed(tmp_path):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        body = (
            chunk("Partial")
            if attempts == 1
            else chunk("Complete", finish_reason="stop") + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    backend = LlamaCppBackend(transport=httpx.MockTransport(handler))
    database = tmp_path / "memory.sqlite3"
    backend.context_usage = lambda messages: ContextUsage(16, 8192)

    async def run():
        async with JamieApplication(
            ProjectConfig(memory_path=str(database)), backend_factory=lambda: backend
        ) as app:
            with pytest.raises(LLMError, match="incomplete streamed"):
                await app.run_turn("first")
            assert (await app.run_turn("second")).message == "Complete"

    asyncio.run(run())
    assert MemoryStore(database).search_memories("first") == []


def test_one_shot_renderer_prints_streamed_answer_once():
    console = Console(file=io.StringIO())
    renderer = OneShotRenderer(console, console)
    renderer.handle(AssistantDelta("Hel"))
    assert console.file.getvalue() == "Hel"
    renderer.handle(AssistantDelta("lo"))
    renderer.handle(AssistantFinished("Hello"))
    assert console.file.getvalue() == "Hello\n"
