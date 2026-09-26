"""Run the model action loop with bounded retries and tool execution."""

import json
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from assistant.agent.events import (
    AgentEvent,
    ApprovalRequested,
    AssistantFinished,
    EventSink,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnStarted,
    discard_event,
    emit_event,
    event_scope,
)
from assistant.agent.protocol import (
    ACTION_PROTOCOL_INSTRUCTIONS,
    FinishAction,
    ProtocolError,
    ToolAction,
    action_schema,
    parse_action,
)
from assistant.cancellation import CancellationToken, cancellation_scope, check_cancelled
from assistant.llm.answer import answer_text
from assistant.llm.base import LLMBackend, Message, constrained_to

MAX_STEPS = 12
MAX_PARSE_RETRIES = 2
MAX_REPEATED_FAILED_ACTIONS = 3
TIMEOUT_SECONDS = 300.0

INVALID_OUTPUT_MESSAGE = """\
Your previous response was invalid JSON or did not match the action protocol.
Return exactly one valid action. A tool call must use
{"action":"tool","tool":"tool_name","args":{}}.
Keep the tool name in "tool" and put every argument inside "args".\
"""

REPEATED_RESULT_MESSAGE = """\
This action returned the same result as your previous call. Do not repeat it.
Use the results you have: return a finish action, or choose a different action.\
"""


class AgentError(RuntimeError):
    """The agent stopped before producing a finish action."""


class AgentInvalidOutputError(AgentError):
    """The model exhausted its retries without returning a valid action."""


class AgentRepeatedActionError(AgentError):
    """The same tool action failed too many consecutive times."""


class AgentStepLimitError(AgentError):
    """The model exhausted the configured action limit."""


class AgentTimeoutError(AgentError):
    """The complete agent run exceeded its deadline."""


class ToolResult(BaseModel):
    """A tool result returned to the model as an observation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    success: bool
    output: str
    denied: bool = False  # the user declined approval; the turn stops instead of retrying
    final: bool = False  # the output is the turn's answer, shown as is


class ToolExecutor(Protocol):
    """Execute one named tool with decoded JSON arguments."""

    def execute(self, tool: str, args: dict[str, Any]) -> ToolResult: ...


class _DeadlineExceeded(Exception):
    pass


def _call_before_deadline[Result](
    call: Callable[[], Result],
    deadline: float,
    event_sink: EventSink = discard_event,
    cancel: CancellationToken | None = None,
) -> Result:
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise _DeadlineExceeded

    completed: Queue[tuple[bool | None, object]] = Queue()
    approval_shown = Event()

    def progress(event: AgentEvent) -> None:
        completed.put((None, event))
        if isinstance(event, ApprovalRequested):
            # The approval prompt shares the terminal, so it waits until the caller has
            # shown everything before it. Never prompt for a turn that already timed out.
            if not approval_shown.wait(max(0.0, deadline - perf_counter())):
                raise _DeadlineExceeded
            approval_shown.clear()

    def invoke() -> None:
        try:
            with event_scope(progress), cancellation_scope(cancel):
                completed.put((True, call()))
        except BaseException as exc:
            completed.put((False, exc))

    Thread(target=invoke, name="jamie-agent-call", daemon=True).start()
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise _DeadlineExceeded
        try:
            succeeded, value = completed.get(timeout=remaining)
        except Empty as exc:
            raise _DeadlineExceeded from exc
        if succeeded is None:
            event_sink(cast("AgentEvent", value))
            if isinstance(value, ApprovalRequested):
                approval_shown.set()
        elif succeeded:
            return cast("Result", value)
        else:
            raise cast("BaseException", value)


def _request_action(
    llm: LLMBackend,
    messages: list[Message],
    max_parse_retries: int,
    schema: dict[str, Any],
) -> ToolAction | FinishAction:
    last_error: ProtocolError | None = None
    for attempt in range(max_parse_retries + 1):
        check_cancelled()
        # Without the schema, small models often write a sentence before the JSON action.
        with constrained_to(schema):
            response = llm.chat(messages)
        messages.append(Message(role="assistant", content=response.content))
        try:
            action = parse_action(response.content)
        except ProtocolError as exc:
            last_error = exc
            if attempt < max_parse_retries:
                messages.append(Message(role="user", content=INVALID_OUTPUT_MESSAGE))
        else:
            return action
    attempts = max_parse_retries + 1
    unit = "time" if attempts == 1 else "times"
    raise AgentInvalidOutputError(
        f"The model returned invalid output {attempts} {unit}."
    ) from last_error


def tool_observation(action: ToolAction, result: ToolResult) -> Message:
    """Format one tool result as the next user message to the model."""
    content = json.dumps(
        {
            "tool": action.tool,
            "args": action.args,
            "success": result.success,
            "output": result.output,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return Message(
        role="user",
        content=f"Tool observation:\n{content}\nReturn the next action.",
    )


def _action_key(action: ToolAction) -> str:
    return json.dumps(
        {"tool": action.tool, "args": action.args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _system_instructions(tools: ToolExecutor) -> str:
    instructions = getattr(tools, "instructions", None)
    if instructions is None:
        return ACTION_PROTOCOL_INSTRUCTIONS
    extra = instructions()
    if not extra.strip():
        return ACTION_PROTOCOL_INSTRUCTIONS
    return f"{extra.strip()}\n\n{ACTION_PROTOCOL_INSTRUCTIONS}"


def _optional_text(tools: ToolExecutor, method: str, *args: object) -> str | None:
    callback = getattr(tools, method, None)
    if callback is None:
        return None
    return callback(*args)


def _optional_call(tools: ToolExecutor, method: str) -> None:
    callback = getattr(tools, method, None)
    if callback is not None:
        callback()


def _run_loop(
    llm: LLMBackend,
    tools: ToolExecutor,
    prompt: str,
    *,
    max_steps: int,
    max_parse_retries: int,
    deadline: float,
    memory_context: str | None,
    history: list[Message] | None,
) -> str:
    _optional_call(tools, "begin_turn")
    try:
        system_content = _system_instructions(tools)
        definitions = getattr(tools, "definitions", None)
        schema = action_schema([d["name"] for d in definitions()] if definitions else None)
        if memory_context is not None:
            system_content = f"{system_content}\n\n{memory_context}"
        messages = [
            Message(role="system", content=system_content),
        ]
        messages.extend(history or [])
        messages.append(Message(role="user", content=prompt))
        last_failed_action: str | None = None
        repeated_failures = 0
        last_result: tuple[str, str] | None = None
        repeated_results = 0

        for _ in range(max_steps):
            action = _request_action(llm, messages, max_parse_retries, schema)
            if isinstance(action, FinishAction):
                verification = _optional_text(tools, "verification_message")
                if verification is not None:
                    messages.append(Message(role="user", content=verification))
                    continue
                return answer_text(action.message)

            # The wrapper stops waiting at the deadline but cannot stop this
            # thread, so check here to avoid side effects after a reported timeout.
            if perf_counter() >= deadline:
                raise _DeadlineExceeded
            check_cancelled()
            emit_event(ToolStarted(action.tool, deepcopy(action.args)))
            started = perf_counter()
            try:
                result = tools.execute(action.tool, dict(action.args))
            except Exception as exc:
                emit_event(ToolFailed(action.tool, str(exc), perf_counter() - started))
                raise
            duration = perf_counter() - started
            if result.success:
                emit_event(ToolFinished(action.tool, result, duration))
            else:
                emit_event(ToolFailed(action.tool, result.output, duration))
            if result.denied:
                # Retrying or working around a "no" would ignore the user's decision.
                return f"Stopped: {action.tool} was not approved."
            if result.final:
                return result.output
            observation = tool_observation(action, result)
            action_key = _action_key(action)
            # Small models re-run a successful search instead of using its results.
            result_key = (action_key, result.output) if result.success else None
            same = result_key is not None and result_key == last_result
            repeated_results = repeated_results + 1 if same else 1
            last_result = result_key
            if same:
                observation = Message(
                    role="user", content=f"{observation.content}\n{REPEATED_RESULT_MESSAGE}"
                )
            messages.append(observation)

            if result.success:
                last_failed_action = None
                repeated_failures = 0
            elif action_key == last_failed_action:
                repeated_failures += 1
            else:
                last_failed_action = action_key
                repeated_failures = 1

            if repeated_failures >= MAX_REPEATED_FAILED_ACTIONS:
                raise AgentRepeatedActionError(
                    f"The tool action {action.tool!r} failed "
                    f"{MAX_REPEATED_FAILED_ACTIONS} consecutive times."
                )
            if repeated_results >= MAX_REPEATED_FAILED_ACTIONS:
                raise AgentRepeatedActionError(
                    f"The tool action {action.tool!r} returned the same result "
                    f"{MAX_REPEATED_FAILED_ACTIONS} consecutive times."
                )
        if perf_counter() >= deadline:
            raise _DeadlineExceeded
        raise AgentStepLimitError(f"The agent reached the {max_steps}-step limit.")
    finally:
        _optional_call(tools, "end_turn")


def run_agent(
    llm: LLMBackend,
    tools: ToolExecutor,
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    max_parse_retries: int = MAX_PARSE_RETRIES,
    timeout_seconds: float = TIMEOUT_SECONDS,
    memory_context: str | None = None,
    history: list[Message] | None = None,
    event_sink: EventSink = discard_event,
    cancel: CancellationToken | None = None,
) -> str:
    """Run bounded model actions until the model returns a finish action.

    Cancellation is checked before each model request and tool call, and is passed to
    the model and tools so they can stop early; it raises TurnCancelled.
    """

    event_sink(TurnStarted(prompt))
    try:
        answer = _optional_text(tools, "handle_command", prompt.strip())
        if answer is None:
            deadline = perf_counter() + timeout_seconds
            try:
                answer = _call_before_deadline(
                    partial(
                        _run_loop,
                        llm,
                        tools,
                        prompt,
                        max_steps=max_steps,
                        max_parse_retries=max_parse_retries,
                        deadline=deadline,
                        memory_context=memory_context,
                        history=history,
                    ),
                    deadline,
                    event_sink,
                    cancel,
                )
            except _DeadlineExceeded as exc:
                raise AgentTimeoutError(
                    f"The agent exceeded its {timeout_seconds:g}-second timeout."
                ) from exc
    except BaseException as exc:
        event_sink(TurnFailed(str(exc) or type(exc).__name__))
        raise
    event_sink(AssistantFinished(answer))
    return answer
