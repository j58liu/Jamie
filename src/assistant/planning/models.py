"""Plan schema: a bounded list of steps with explicit dependencies.

The model proposes the steps; the runtime validates them and enforces order,
permissions, budgets, and verification.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

MAX_PLAN_STEPS = 8
MAX_CALLS_PER_STEP = 5
ASK_USER = "ask_user"


class StepStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"  # a dependency did not complete, or no calls were needed
    STOPPED = "stopped"  # approval denied or access missing; nothing else runs
    NEEDS_REVIEW = "needs_review"  # the provider's write result is ambiguous
    NEEDS_USER = "needs_user"  # Jamie must ask the user before continuing


class Verification(BaseModel):
    """Read the result back with the provider id instead of assuming it worked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    argument: str  # argument of the verifying tool, e.g. "event_id"
    result_field: str = "id"  # field of the step's output holding the provider id


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^step_[0-9]{1,2}$")
    description: str = Field(min_length=1, max_length=300)
    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)  # known values; rest filled later
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_PLAN_STEPS)
    multiple: bool = False  # may run up to five times, e.g. one event per deadline
    expected_output: Literal["any", "nonempty"] = "any"
    verification: Verification | None = None
    question: str | None = Field(default=None, max_length=300)  # for ask_user
    status: StepStatus = StepStatus.PENDING


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=300)
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)


def planner_schema(tool_names: list[str]) -> dict:
    """The JSON schema the model's plan is constrained to."""
    return {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PLAN_STEPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "tool": {"type": "string", "enum": [*tool_names, ASK_USER]},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "multiple": {"type": "boolean"},
                        "question": {"type": ["string", "null"]},
                    },
                    "required": ["id", "description", "tool", "depends_on"],
                },
            },
        },
        "required": ["goal", "steps"],
    }
