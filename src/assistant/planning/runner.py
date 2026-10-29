"""Choose between the agent loop and the planner, and run a plan as one turn."""

from __future__ import annotations

import re
from functools import partial
from time import perf_counter

from assistant.agent.events import AssistantFinished, EventSink, TurnFailed, TurnStarted
from assistant.agent.loop import AgentTimeoutError, _call_before_deadline, _DeadlineExceeded
from assistant.cancellation import CancellationToken
from assistant.planning.executor import PlanBudget, PlanExecutor, PlanOutcome

_ACTIONS = re.compile(
    r"\b(find|search|look up|check|summari[sz]e|add|create|schedule|book|prepare|draft|"
    r"reply|respond|send|remind|list|update|move|reschedule|read|extract|compare|email)\b",
    re.IGNORECASE,
)


def needs_plan(prompt: str) -> bool:
    """Requests that chain three or more actions go to the planner."""
    verbs = {match.group(1).lower() for match in _ACTIONS.finditer(prompt)}
    return len(verbs) >= 3


def run_plan(
    backend,
    tools,
    prompt: str,
    *,
    memory_context: str | None = None,
    event_sink: EventSink,
    cancel: CancellationToken | None = None,
    budget: PlanBudget | None = None,
) -> PlanOutcome:
    budget = budget or PlanBudget()
    event_sink(TurnStarted(prompt))
    begin = getattr(tools, "begin_turn", None)
    end = getattr(tools, "end_turn", None)
    turn_cancel = cancel or CancellationToken()
    deadline = perf_counter() + budget.total_seconds
    executor = PlanExecutor(backend, tools, budget=budget, memory_context=memory_context)
    try:
        if callable(begin):
            begin()
        try:
            outcome = _call_before_deadline(
                partial(executor.run, prompt), deadline, event_sink, turn_cancel
            )
        except _DeadlineExceeded as exc:
            turn_cancel.cancel()
            raise AgentTimeoutError(
                f"The plan exceeded its {budget.total_seconds:g}-second limit."
            ) from exc
        finally:
            if callable(end):
                end()
    except BaseException as exc:
        event_sink(TurnFailed(str(exc) or type(exc).__name__))
        raise
    event_sink(AssistantFinished(outcome.answer))
    return outcome
