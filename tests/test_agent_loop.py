import json
from threading import Event
from time import perf_counter

import pytest

from assistant.agent.loop import (
    INVALID_OUTPUT_MESSAGE,
    MAX_PARSE_RETRIES,
    MAX_REPEATED_FAILED_ACTIONS,
    MAX_STEPS,
    REPEATED_RESULT_MESSAGE,
    TIMEOUT_SECONDS,
    AgentInvalidOutputError,
    AgentRepeatedActionError,
    AgentStepLimitError,
    AgentTimeoutError,
    ToolResult,
    run_agent,
)
from assistant.agent.protocol import ACTION_PROTOCOL_INSTRUCTIONS
from assistant.llm.base import LLMResponse, Message


class FakeBackend:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.requests: list[list[Message]] = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.requests.append(list(messages))
        return LLMResponse(next(self.responses))


class FakeTools:
    def __init__(self, results: list[ToolResult] | None = None):
        self.results = iter(results or [])
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool, args):
        self.calls.append((tool, args))
        return next(self.results)


def tool_action(tool: str = "read_file", args: str = '{"path":"README.md"}') -> str:
    return f'{{"action":"tool","tool":"{tool}","args":{args}}}'


def finish_action(message: str = "Done.") -> str:
    return json.dumps({"action": "finish", "message": message})


def test_default_limits_are_bounded():
    assert MAX_STEPS == 12
    assert MAX_PARSE_RETRIES == 2
    assert MAX_REPEATED_FAILED_ACTIONS == 3
    assert TIMEOUT_SECONDS == 300.0


def test_finish_action_returns_message_without_calling_tools():
    backend = FakeBackend([finish_action("Complete.")])
    tools = FakeTools()

    assert run_agent(backend, tools, "Answer the question.") == "Complete."
    assert tools.calls == []
    assert backend.requests == [
        [
            Message("system", ACTION_PROTOCOL_INSTRUCTIONS),
            Message("user", "Answer the question."),
        ]
    ]


def test_memory_context_is_appended_to_the_system_message():
    backend = FakeBackend([finish_action("SQLite was discussed.")])
    context = "Relevant memory: the database library was SQLite."
    assert (
        run_agent(backend, FakeTools(), "What did we discuss?", memory_context=context)
        == "SQLite was discussed."
    )
    assert backend.requests[0] == [
        Message("system", f"{ACTION_PROTOCOL_INSTRUCTIONS}\n\n{context}"),
        Message("user", "What did we discuss?"),
    ]


def test_tool_result_is_added_before_the_next_action():
    backend = FakeBackend([tool_action(), finish_action("The file was read.")])
    tools = FakeTools([ToolResult(success=True, output="README contents")])

    assert run_agent(backend, tools, "Read the README.") == "The file was read."
    assert tools.calls == [("read_file", {"path": "README.md"})]
    second_request = backend.requests[1]
    assert second_request[2] == Message("assistant", tool_action())
    assert second_request[3].role == "user"
    label, payload, instruction = second_request[3].content.splitlines()
    assert label == "Tool observation:"
    assert instruction == "Return the next action."
    assert json.loads(payload) == {
        "tool": "read_file",
        "args": {"path": "README.md"},
        "success": True,
        "output": "README contents",
    }


def test_invalid_output_is_retried_with_correction():
    backend = FakeBackend(["not JSON", finish_action()])

    assert run_agent(backend, FakeTools(), "Finish.") == "Done."
    assert len(backend.requests) == 2
    assert backend.requests[1][-2:] == [
        Message("assistant", "not JSON"),
        Message("user", INVALID_OUTPUT_MESSAGE),
    ]


def test_invalid_output_stops_after_retry_limit():
    backend = FakeBackend(["invalid one", "invalid two", "invalid three"])

    with pytest.raises(AgentInvalidOutputError, match="invalid output 3 times"):
        run_agent(backend, FakeTools(), "Finish.")
    assert len(backend.requests) == 3


def test_zero_parse_retries_allows_only_initial_response():
    backend = FakeBackend(["invalid"])

    with pytest.raises(AgentInvalidOutputError, match="invalid output 1 time"):
        run_agent(backend, FakeTools(), "Finish.", max_parse_retries=0)
    assert len(backend.requests) == 1


def test_step_limit_stops_tool_loop():
    backend = FakeBackend([tool_action(), tool_action()])
    tools = FakeTools(
        [
            ToolResult(success=True, output="first"),
            ToolResult(success=True, output="second"),
        ]
    )

    with pytest.raises(AgentStepLimitError, match="2-step limit"):
        run_agent(backend, tools, "Keep reading.", max_steps=2)
    assert len(backend.requests) == 2
    assert len(tools.calls) == 2


def test_three_consecutive_failures_of_same_action_stop_loop():
    first = tool_action(args='{"path":"README.md","line":1}')
    reordered = tool_action(args='{"line":1,"path":"README.md"}')
    backend = FakeBackend([first, reordered, first])
    tools = FakeTools([ToolResult(success=False, output="failed")] * 3)

    with pytest.raises(AgentRepeatedActionError, match="failed 3 consecutive times"):
        run_agent(backend, tools, "Read one line.")
    assert len(backend.requests) == 3
    assert len(tools.calls) == 3


def test_success_resets_repeated_failure_detection():
    backend = FakeBackend(
        [tool_action(), tool_action(), tool_action(), tool_action(), finish_action()]
    )
    tools = FakeTools(
        [
            ToolResult(success=False, output="failed"),
            ToolResult(success=False, output="failed"),
            ToolResult(success=True, output="worked"),
            ToolResult(success=False, output="failed again"),
        ]
    )

    assert run_agent(backend, tools, "Read the file.") == "Done."
    assert len(tools.calls) == 4


def test_repeated_identical_result_asks_the_model_to_use_it():
    search = tool_action("web_search", '{"query":"natural gas"}')
    backend = FakeBackend([search, search, finish_action("Summary.")])
    tools = FakeTools([ToolResult(success=True, output="same results")] * 2)

    assert run_agent(backend, tools, "Research natural gas.") == "Summary."
    assert REPEATED_RESULT_MESSAGE not in backend.requests[1][-1].content
    assert backend.requests[2][-1].content.endswith(REPEATED_RESULT_MESSAGE)


def test_third_identical_result_stops_loop():
    search = tool_action("web_search", '{"query":"natural gas"}')
    backend = FakeBackend([search] * 3)
    tools = FakeTools([ToolResult(success=True, output="same results")] * 3)

    with pytest.raises(AgentRepeatedActionError, match="same result 3 consecutive times"):
        run_agent(backend, tools, "Research natural gas.")
    assert len(tools.calls) == 3


def test_denied_approval_ends_the_turn_without_asking_the_model_again():
    backend = FakeBackend([tool_action("open_application", '{"name":"Safari"}')])
    denied = ToolResult(success=False, output="User approval was not granted.", denied=True)

    answer = run_agent(backend, FakeTools([denied]), "Open Safari.")

    assert answer == "Stopped: open_application was not approved."
    assert len(backend.requests) == 1


def test_different_failed_action_resets_repeated_failure_detection():
    read = tool_action()
    inspect = tool_action("inspect_file", '{"path":"README.md"}')
    backend = FakeBackend([read, read, inspect, read, read, finish_action()])
    tools = FakeTools([ToolResult(success=False, output="failed")] * 5)

    assert run_agent(backend, tools, "Inspect the file.") == "Done."
    assert len(tools.calls) == 5


def test_model_call_stops_at_total_timeout():
    release = Event()

    class BlockingBackend:
        def chat(self, messages, *, temperature=0.6, max_tokens=1024):
            release.wait(1)
            return LLMResponse(finish_action())

    started = perf_counter()
    try:
        with pytest.raises(AgentTimeoutError, match="0.02-second timeout"):
            run_agent(BlockingBackend(), FakeTools(), "Wait.", timeout_seconds=0.02)
    finally:
        release.set()
    assert perf_counter() - started < 0.5


def test_tool_call_stops_at_total_timeout():
    release = Event()

    class BlockingTools:
        def execute(self, tool, args):
            release.wait(1)
            return ToolResult(success=True, output="late")

    started = perf_counter()
    try:
        with pytest.raises(AgentTimeoutError, match="0.02-second timeout"):
            run_agent(
                FakeBackend([tool_action()]),
                BlockingTools(),
                "Wait.",
                timeout_seconds=0.02,
            )
    finally:
        release.set()
    assert perf_counter() - started < 0.5


@pytest.mark.parametrize(
    "options",
    [
        {"max_steps": 0},
        {"max_parse_retries": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
    ],
)
def test_invalid_limits_fail_before_model_call(options):
    backend = FakeBackend([])
    with pytest.raises(ValueError):
        run_agent(backend, FakeTools(), "Prompt.", **options)
    assert backend.requests == []


@pytest.mark.parametrize("prompt", ["", "   "])
def test_invalid_prompt_fails_before_model_call(prompt):
    backend = FakeBackend([])
    with pytest.raises(ValueError, match="prompt"):
        run_agent(backend, FakeTools(), prompt)
    assert backend.requests == []


def test_blank_memory_context_is_rejected_before_model_call():
    backend = FakeBackend([])
    with pytest.raises(ValueError, match="memory_context"):
        run_agent(backend, FakeTools(), "Prompt.", memory_context="  ")
    assert backend.requests == []
