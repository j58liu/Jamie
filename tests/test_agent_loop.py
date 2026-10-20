import json
from threading import Event
from time import perf_counter

import pytest

from assistant.agent.loop import (
    INVALID_OUTPUT_MESSAGE,
    MAX_PARSE_RETRIES,
    MAX_REPEATED_ACTIONS,
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
from assistant.agent.protocol import ACTION_PROTOCOL_INSTRUCTIONS, action_schema
from assistant.llm.base import LLMResponse, Message, current_response_schema


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
    assert MAX_REPEATED_ACTIONS == 3
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

    with pytest.raises(AgentRepeatedActionError, match="failed 3 times"):
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

    with pytest.raises(AgentRepeatedActionError, match="same result 3 times"):
        run_agent(backend, tools, "Research natural gas.")
    assert len(tools.calls) == 3


def test_denied_approval_ends_the_turn_without_asking_the_model_again():
    backend = FakeBackend([tool_action("open_application", '{"name":"Safari"}')])
    denied = ToolResult(success=False, output="User approval was not granted.", denied=True)

    answer = run_agent(backend, FakeTools([denied]), "Open Safari.")

    assert answer == "Stopped: open_application was not approved."
    assert len(backend.requests) == 1


def test_repeats_count_across_the_turn_even_with_other_actions_between():
    # The model alternates between a folder that exists and one that does not.
    listed = tool_action("list_directory", '{"path":"src"}')
    missing = tool_action("list_directory", '{"path":"src/agent"}')
    backend = FakeBackend([listed, missing, listed, missing, listed])
    tools = FakeTools(
        [
            ToolResult(success=True, output="assistant/"),
            ToolResult(success=False, output="Path does not exist"),
        ]
        * 3
    )

    with pytest.raises(AgentRepeatedActionError, match="same result 3 times"):
        run_agent(backend, tools, "What is in the agent folder?")
    assert len(tools.calls) == 5
    assert backend.requests[3][-1].content.endswith(REPEATED_RESULT_MESSAGE)


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


def test_action_requests_are_constrained_but_model_calls_inside_tools_are_not():
    seen: list[tuple[str, object]] = []

    class Backend(FakeBackend):
        def chat(self, messages, **options):
            seen.append(("model", current_response_schema()))
            return super().chat(messages, **options)

    class Tools(FakeTools):
        def definitions(self):
            return [{"name": "research"}, {"name": "read_file"}]

        def execute(self, tool, args):
            seen.append(("tool", current_response_schema()))  # e.g. research calls the model
            return super().execute(tool, args)

    backend = Backend([tool_action("research", '{"question":"q"}'), finish_action()])
    tools = Tools([ToolResult(success=True, output="report")])

    run_agent(backend, tools, "Research q.")

    schema = action_schema(["research", "read_file"])
    assert seen == [("model", schema), ("tool", None), ("model", schema)]
    assert schema["oneOf"][0]["properties"]["tool"] == {"enum": ["research", "read_file"]}
