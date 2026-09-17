"""Run the model action loop with bounded retries and tool execution."""

import json
import math
from collections.abc import Callable
from functools import partial
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from assistant.agent.protocol import (
    ACTION_PROTOCOL_INSTRUCTIONS,
    FinishAction,
    ProtocolError,
    ToolAction,
    parse_action,
)
from assistant.llm.base import LLMBackend, Message

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


class ToolExecutor(Protocol):
    """Execute one named tool with decoded JSON arguments."""

    def execute(self, tool: str, args: dict[str, Any]) -> ToolResult: ...


class _DeadlineExceeded(Exception):
    pass


def _validate_limits(max_steps: int, max_parse_retries: int, timeout_seconds: float) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    if (
        isinstance(max_parse_retries, bool)
        or not isinstance(max_parse_retries, int)
        or max_parse_retries < 0
    ):
        raise ValueError("max_parse_retries must be a non-negative integer.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite, positive number.")


def _call_before_deadline[Result](
    call: Callable[[], Result],
    deadline: float,
) -> Result:
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise _DeadlineExceeded

    completed: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            completed.put((True, call()))
        except BaseException as exc:
            completed.put((False, exc))

    Thread(target=invoke, name="jamie-agent-call", daemon=True).start()
    try:
        succeeded, value = completed.get(timeout=remaining)
    except Empty as exc:
        raise _DeadlineExceeded from exc
    if succeeded:
        return cast("Result", value)
    raise cast("BaseException", value)


def _request_action(
    llm: LLMBackend,
    messages: list[Message],
    deadline: float,
    max_parse_retries: int,
) -> ToolAction | FinishAction:
    last_error: ProtocolError | None = None
    for attempt in range(max_parse_retries + 1):
        if perf_counter() >= deadline:
            raise _DeadlineExceeded
        response = llm.chat(messages)
        if perf_counter() >= deadline:
            raise _DeadlineExceeded
        messages.append(Message(role="assistant", content=response.content))
        try:
            action = parse_action(response.content)
        except ProtocolError as exc:
            last_error = exc
            if attempt < max_parse_retries:
                messages.append(Message(role="user", content=INVALID_OUTPUT_MESSAGE))
        else:
            if perf_counter() >= deadline:
                raise _DeadlineExceeded
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
    if not isinstance(extra, str):
        raise TypeError("ToolExecutor.instructions must return a string.")
    if not extra.strip():
        return ACTION_PROTOCOL_INSTRUCTIONS
    return f"{extra.strip()}\n\n{ACTION_PROTOCOL_INSTRUCTIONS}"


def _run_loop(
    llm: LLMBackend,
    tools: ToolExecutor,
    prompt: str,
    *,
    max_steps: int,
    max_parse_retries: int,
    deadline: float,
) -> str:
    messages = [
        Message(role="system", content=_system_instructions(tools)),
        Message(role="user", content=prompt),
    ]
    last_failed_action: str | None = None
    repeated_failures = 0

    for _ in range(max_steps):
        action = _request_action(
            llm,
            messages,
            deadline,
            max_parse_retries,
        )
        if isinstance(action, FinishAction):
            return action.message

        if perf_counter() >= deadline:
            raise _DeadlineExceeded
        result = tools.execute(action.tool, dict(action.args))
        if perf_counter() >= deadline:
            raise _DeadlineExceeded
        if not isinstance(result, ToolResult):
            raise TypeError("ToolExecutor.execute must return ToolResult.")
        messages.append(tool_observation(action, result))

        action_key = _action_key(action)
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
    if perf_counter() >= deadline:
        raise _DeadlineExceeded
    raise AgentStepLimitError(f"The agent reached the {max_steps}-step limit.")


def run_agent(
    llm: LLMBackend,
    tools: ToolExecutor,
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    max_parse_retries: int = MAX_PARSE_RETRIES,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> str:
    """Run bounded model actions until the model returns a finish action."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-blank string.")
    _validate_limits(max_steps, max_parse_retries, timeout_seconds)

    deadline = perf_counter() + timeout_seconds
    try:
        return _call_before_deadline(
            partial(
                _run_loop,
                llm,
                tools,
                prompt,
                max_steps=max_steps,
                max_parse_retries=max_parse_retries,
                deadline=deadline,
            ),
            deadline,
        )
    except _DeadlineExceeded as exc:
        raise AgentTimeoutError(
            f"The agent exceeded its {timeout_seconds:g}-second timeout."
        ) from exc
