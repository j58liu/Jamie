"""Check a proposed plan before anything runs."""

from __future__ import annotations

from assistant.planning.models import ASK_USER, MAX_PLAN_STEPS, Plan, PlanStep, Verification
from assistant.security.policy import DEFAULT_TOOL_PERMISSIONS, PermissionLevel


class PlanValidationError(ValueError):
    """The plan cannot be executed as written."""


# How to read back each write. The runtime adds these; the model does not choose them.
VERIFIERS = {
    "calendar_create_event": Verification(tool="calendar_get_event", argument="event_id"),
    "calendar_update_event": Verification(tool="calendar_get_event", argument="event_id"),
    "gmail_create_draft": Verification(tool="gmail_get_draft", argument="draft_id"),
}

# Retrying these with the same operation key returns the first result instead of a copy.
IDEMPOTENT_WRITES = {"calendar_create_event", "gmail_create_draft", "calendar_update_event"}


def is_read(tool: str) -> bool:
    return DEFAULT_TOOL_PERMISSIONS.get(tool) is PermissionLevel.READ


def validate_plan(plan: Plan, schemas: dict[str, dict]) -> Plan:
    """Validate ids, dependencies, tools, and known arguments; fill runtime rules."""
    if len(plan.steps) > MAX_PLAN_STEPS:
        raise PlanValidationError(f"A plan may have at most {MAX_PLAN_STEPS} steps.")
    seen: set[str] = set()
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.id in seen:
            raise PlanValidationError(f"Step id {step.id} is repeated.")
        for dependency in step.depends_on:
            if dependency not in seen:
                raise PlanValidationError(
                    f"{step.id} depends on {dependency}, which is not an earlier step."
                )
        if step.tool == ASK_USER:
            if not (step.question or "").strip():
                raise PlanValidationError(f"{step.id} asks the user but has no question.")
        elif step.tool not in schemas:
            raise PlanValidationError(f"{step.id} uses unknown tool {step.tool!r}.")
        else:
            properties = schemas[step.tool].get("properties", {})
            unknown = set(step.args) - set(properties)
            if unknown:
                raise PlanValidationError(
                    f"{step.id} passes unknown arguments to {step.tool}: {sorted(unknown)}."
                )
        seen.add(step.id)
        updates: dict = {}
        if step.verification is None and step.tool in VERIFIERS:
            updates["verification"] = VERIFIERS[step.tool]
        if (
            step.expected_output == "any"
            and is_read(step.tool)
            and ("search" in step.tool or step.tool.endswith("_get_message"))
        ):
            updates["expected_output"] = "nonempty"
        if step.multiple and is_read(step.tool):
            updates["multiple"] = False  # repetition is for writes, e.g. one event each
        steps.append(step.model_copy(update=updates))
    return plan.model_copy(update={"steps": steps})
