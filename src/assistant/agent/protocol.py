"""Parse the two actions accepted from the language model."""

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

ACTION_PROTOCOL_INSTRUCTIONS = """\
Communicate with the harness using exactly one JSON object.

To call a tool, use:
{"action":"tool","tool":"tool_name","args":{}}

To finish, use:
{"action":"finish","message":"your final response"}

Only these two actions are allowed. Return no explanation or Markdown fences.
Use a tool only when the request needs it. Create, edit, or move files only when
the user asks. As soon as you can answer, finish.\
"""


class ProtocolError(ValueError):
    """The model response is not a valid action."""


class ToolAction(BaseModel):
    """Ask the harness to call one tool with JSON arguments."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: Literal["tool"]
    tool: str
    args: dict[str, Any]


class FinishAction(BaseModel):
    """Return the final message to the user."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: Literal["finish"]
    message: str


Action = Annotated[ToolAction | FinishAction, Field(discriminator="action")]
_ACTION_ADAPTER = TypeAdapter(Action)
_JSON_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z", re.DOTALL)


def _reject_constant(value: str) -> None:
    # NaN and Infinity would survive into tool args and make audit records invalid JSON.
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _remove_thinking(text: str) -> str:
    candidate = text.strip()
    if not candidate.startswith("<think>"):
        return candidate
    closing = candidate.find("</think>", len("<think>"))
    if closing == -1:
        raise ProtocolError("The model response contains an unclosed think block.")
    return candidate[closing + len("</think>") :].strip()


def _remove_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    match = _JSON_FENCE.fullmatch(text)
    if match is None:
        raise ProtocolError("The model response contains an invalid JSON fence.")
    return match.group("body").strip()


def validate_action(value: object) -> ToolAction | FinishAction:
    """Validate a decoded JSON value as one of the two supported actions."""
    try:
        return _ACTION_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise ProtocolError("The model response does not match the action protocol.") from exc


def action_schema(tool_names: list[str] | None = None) -> dict[str, Any]:
    """JSON schema for the two actions; ``tool_names`` limits which tools can be named."""
    tool = {"enum": tool_names} if tool_names else {"type": "string"}
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"const": "tool"},
                    "tool": tool,
                    "args": {"type": "object"},
                },
                "required": ["action", "tool", "args"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"action": {"const": "finish"}, "message": {"type": "string"}},
                "required": ["action", "message"],
                "additionalProperties": False,
            },
        ]
    }


def parse_action(text: str) -> ToolAction | FinishAction:
    """Parse a model response without extracting JSON from surrounding prose."""
    candidate = _remove_json_fence(_remove_thinking(text))
    try:
        value = json.loads(candidate, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ProtocolError("The model response is not valid JSON.") from exc
    return validate_action(value)
