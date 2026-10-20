"""Multi-turn evaluations of the runtime interactive mode uses, without the terminal UI.

Each case drives JamieApplication and the agent loop the way the REPL does: history carries
between turns, /new resets it, /compact summarizes it, Ctrl+C cancels through a token, and
approvals go through the real permission policy. A case's web fixture stands in for the
network, so web and research tools run offline. Events are recorded but never rendered, so
display code cannot influence what the model sees.
"""

import asyncio
import math
import statistics
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePath
from time import perf_counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.agent.events import (
    AgentEvent,
    AssistantDelta,
    AssistantFinished,
    TurnFailed,
    TurnStarted,
)
from assistant.agent.loop import INVALID_OUTPUT_MESSAGE, AgentError
from assistant.agent_evaluation import ToolCall, _RecordingBackend, _RecordingTools
from assistant.cancellation import CancellationToken, TurnCancelled
from assistant.config import ProjectConfig
from assistant.evaluation import (
    load_case_files,
    new_report_path,
    run_rounds,
)
from assistant.llm.base import LLMBackend, LLMError, LLMResponse, Message
from assistant.research_evaluation import _FixtureWeb
from assistant.runtime import JamieApplication
from assistant.runtime.context import SUMMARY_INSTRUCTION
from assistant.security.approvals import CallbackApprovalProvider
from assistant.tools.registry import (
    create_agent_registry,
    create_workspace_registry,
    register_web_tools,
)
from assistant.tools.scopes import select_tools
from assistant.tools.web import SearchResult

CANCEL_AFTER_SECONDS = 1.0
MALFORMED_ACTION = "Sure, I will read the file now."


def _total_usage(responses: list[LLMResponse], field: str) -> int | None:
    counts = [getattr(response, field) for response in responses]
    return sum(counts) if counts and all(count is not None for count in counts) else None


class InteractiveStep(BaseModel):
    """One user turn, with the interactive action around it and what must hold after it."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prompt: str = Field(min_length=1)
    tools: bool = False  # run through the agent loop with workspace tools
    new_conversation: bool = False  # /new before this turn
    compact: bool = False  # /compact before this turn
    cancel: bool = False  # Ctrl+C shortly after the turn starts
    approve: bool = False  # the user's answer if the policy asks for approval
    malformed_first_action: bool = False  # replace the model's first action with prose
    answer_contains: list[str] = Field(default_factory=list)
    tool_called: str | None = None
    tool_failed: bool = False  # a tool call must fail first (recovery scenarios)
    approval_requested: bool | None = None
    files_after: dict[str, str] = Field(default_factory=dict)  # path -> required content
    files_absent: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        tool_only = (
            self.approve
            or self.malformed_first_action
            or self.tool_called
            or self.tool_failed
            or self.approval_requested is not None
        )
        if tool_only and not self.tools:
            raise ValueError("Tool, approval, and malformed-action checks need tools: true.")
        if self.cancel and (self.tools or self.answer_contains):
            raise ValueError("Cancelled steps are chat turns without answer checks.")
        return self


class WebFixture(BaseModel):
    """Search results and pages (URL to HTML) served instead of the network."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    search_results: list[SearchResult] = Field(default_factory=list)
    pages: dict[str, str] = Field(default_factory=dict)


class InteractiveEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    files: dict[str, str] = Field(default_factory=dict)
    web: WebFixture | None = None  # adds web_search, fetch_url, and research to tool steps
    steps: list[InteractiveStep] = Field(min_length=1)
    context_grows: bool = False  # the measured context must grow on every turn
    all_tools: bool = False
    tool_scope: Literal["all", "files", "read", "web", "macos"] = "all"

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        for name in self.files:
            path = PurePath(name)
            if not name.strip() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Fixture paths must stay within the workspace.")
        return self


class StepResult(BaseModel):
    prompt: str
    answer: str | None
    cancelled: bool
    error: str | None
    tool_calls: list[ToolCall]
    approvals_requested: int
    events: list[str]
    context_tokens: int | None
    passed: bool
    failure_reason: str | None
    latency_seconds: float = 0.0
    first_visible_seconds: float | None = None
    model_calls: int = 0
    parse_retries: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    compactions: int = 0
    initial_input_tokens: int | None = None


class InteractiveRunResult(BaseModel):
    case_id: str
    run: int
    success: bool
    failed_step: int | None
    failure_reason: str | None
    model: str | None
    output_tokens: int | None
    latency_seconds: float
    steps: list[StepResult]
    model_outputs: list[str] = Field(default_factory=list)


class InteractiveEvalReport(BaseModel):
    schema_version: int = 1
    package_version: str = __version__
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "interrupted"] = "running"
    cases: list[InteractiveEvalCase]
    runs_per_case: int
    timeout_seconds: float
    total_duration_seconds: float = 0.0
    results: list[InteractiveRunResult] = Field(default_factory=list)
    environment: dict[str, JsonValue] = Field(default_factory=dict)

    def summary(self) -> dict[str, JsonValue]:
        planned = len(self.cases) * self.runs_per_case
        completed = len(self.results)
        passes = sum(result.success for result in self.results)
        tokens = [r.output_tokens for r in self.results if r.output_tokens is not None]
        steps = [step for result in self.results for step in result.steps]
        latencies = sorted(r.latency_seconds for r in self.results)
        visible = [
            step.first_visible_seconds for step in steps if step.first_visible_seconds is not None
        ]
        initial = [
            step.initial_input_tokens for step in steps if step.initial_input_tokens is not None
        ]
        input_counts = [step.input_tokens for step in steps if step.input_tokens is not None]
        return {
            "models": sorted({r.model for r in self.results if r.model is not None}),
            "cases": len(self.cases),
            "runs_per_case": self.runs_per_case,
            "planned_runs": planned,
            "completed_runs": completed,
            "passes": passes,
            "failures": completed - passes,
            "failed_cases": sorted({r.case_id for r in self.results if not r.success}),
            "median_latency_seconds": (
                statistics.median(r.latency_seconds for r in self.results) if self.results else None
            ),
            "p95_latency_seconds": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
            if latencies
            else None,
            "median_first_visible_seconds": statistics.median(visible) if visible else None,
            "first_visible_reported_turns": len(visible),
            "model_calls": sum(step.model_calls for step in steps),
            "parse_retries": sum(step.parse_retries for step in steps),
            "compactions": sum(step.compactions for step in steps),
            "median_initial_input_tokens": statistics.median(initial) if initial else None,
            "approval_prompts": sum(step.approvals_requested for step in steps),
            "input_tokens": sum(input_counts) if input_counts else None,
            "input_usage_reported_turns": sum(step.input_tokens is not None for step in steps),
            "output_tokens": sum(tokens) if tokens else None,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.status == "completed" and passes == planned,
        }


def load_interactive_cases(path: Path | None = None) -> list[InteractiveEvalCase]:
    return load_case_files(path, InteractiveEvalCase, "interactive_cases", "interactive")


def _events_well_formed(events: list[AgentEvent]) -> bool:
    return (
        bool(events)
        and isinstance(events[0], TurnStarted)
        and isinstance(events[-1], AssistantFinished | TurnFailed)
    )


def _step_failure(
    step: InteractiveStep,
    result: StepResult,
    events: list[AgentEvent],
    workspace: Path,
) -> str | None:
    """Return the first unmet expectation, or None when the step passed."""
    if not _events_well_formed(events):
        return "events_out_of_order"
    if step.cancel:
        return None if result.cancelled else "finished_before_cancel"
    if result.cancelled:
        return "unexpected_cancel"
    if result.error is not None:
        return "turn_error"
    if step.tool_called and not any(c.tool == step.tool_called for c in result.tool_calls):
        return "expected_tool_not_called"
    if step.tool_failed and not any(not call.success for call in result.tool_calls):
        return "no_tool_failure_to_recover_from"
    if step.approval_requested is not None and (
        (result.approvals_requested > 0) != step.approval_requested
    ):
        return "approval_expectation_mismatch"
    for name, content in step.files_after.items():
        target = workspace / name
        if not target.is_file() or content not in target.read_text(encoding="utf-8"):
            return "expected_file_missing"
    if any((workspace / name).exists() for name in step.files_absent):
        return "denied_change_was_made"
    answer = result.answer or ""
    if not all(text.lower() in answer.lower() for text in step.answer_contains):
        return "answer_mismatch"
    return None


def _step_tools(
    step: InteractiveStep,
    case: InteractiveEvalCase,
    workspace: Path,
    backend: "_RequestRecorder",
    approvals: list[object],
) -> _RecordingTools:
    """The REPL's tools for one step: workspace tools, plus fixture-backed web tools."""
    backend.malform_next = step.malformed_first_action
    writes = {"write_file", "edit_file", "move_file", "create_directory", "trash_file"}
    provider = CallbackApprovalProvider(
        lambda request: (
            approvals.append(request)
            or (step.approve and request.tool in writes | {"fetch_url", "research"})
        )
    )
    if case.all_tools:
        fixture = case.web or WebFixture()
        registry = create_agent_registry(
            workspace,
            approvals=provider,
            web=_FixtureWeb(fixture.search_results, fixture.pages),
            backend=backend,
        )
    else:
        registry = create_workspace_registry(workspace, approvals=provider)
    if case.web is not None and not case.all_tools:
        web = _FixtureWeb(case.web.search_results, case.web.pages)
        register_web_tools(registry, web, backend)
    return _RecordingTools(select_tools(registry, case.tool_scope))


async def _run_steps(
    application: JamieApplication,
    backend: "_RequestRecorder",
    case: InteractiveEvalCase,
    workspace: Path,
    timeout_seconds: float,
    max_tokens: int = 1024,
    temperature: float = 0.6,
) -> list[StepResult]:
    history: list[Message] = []
    conversation_id: str | None = None
    cancelled_prompts: list[str] = []
    results: list[StepResult] = []
    previous_context = -1
    for step in case.steps:
        if step.new_conversation:
            history, conversation_id = [], None
        earlier_cancelled = list(cancelled_prompts)
        timed_out: list[bool] = []
        events: list[AgentEvent] = []
        approvals: list[object] = []
        tool_calls: list[ToolCall] = []
        answer = error = None
        cancelled = False
        requests_before = len(backend.requests)
        responses_before = len(backend.responses)
        started = perf_counter()
        first_visible = None

        def observe(event: AgentEvent, events=events, started=started) -> None:
            nonlocal first_visible
            events.append(event)
            if first_visible is None and isinstance(event, AssistantDelta | AssistantFinished):
                if event.text.strip():
                    first_visible = perf_counter() - started

        conversation_id = conversation_id or application.start_conversation()
        token = CancellationToken()
        loop = asyncio.get_running_loop()
        if step.cancel:
            loop.call_later(CANCEL_AFTER_SECONDS, token.cancel)

        # A timeout cancels like Ctrl+C, so llama.cpp stops instead of
        # generating into the next case.
        def expire(token: CancellationToken = token, flag: list = timed_out) -> None:
            flag.append(True)
            token.cancel()

        timer = loop.call_later(timeout_seconds, expire)
        tools = _step_tools(step, case, workspace, backend, approvals) if step.tools else None
        compact_failed = False
        try:
            if step.compact:
                try:
                    history = await application.compact(history, cancel=token)
                except (LLMError, ValueError) as exc:
                    compact_failed = True
                    error = f"Compaction failed: {exc}"
            if not compact_failed:
                # Tool steps take the same run_turn path as the REPL, history included.
                turn = await application.run_turn(
                    step.prompt,
                    history=history,
                    cancel=token,
                    conversation_id=conversation_id,
                    event_sink=observe,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = turn.message
        except TurnCancelled:
            if timed_out:
                error = f"The turn exceeded its {timeout_seconds:g}-second timeout."
            else:
                cancelled = True
                cancelled_prompts.append(step.prompt)
        except (AgentError, LLMError) as exc:
            error = str(exc) or type(exc).__name__
        finally:
            timer.cancel()
            tool_calls = tools.calls if tools is not None else []
        if answer is not None:
            if turn.compacted_history is not None:
                history = list(turn.compacted_history)
            history = [*history, Message("user", step.prompt), Message("assistant", answer)]

        context_tokens = None
        measure = getattr(application, "context_usage", None)
        if case.context_grows and measure is not None:
            try:
                usage = measure(history)
                context_tokens = usage.used if usage is not None else None
            except LLMError:
                context_tokens = None

        result = StepResult(
            prompt=step.prompt,
            answer=answer,
            cancelled=cancelled,
            error=error,
            tool_calls=tool_calls,
            approvals_requested=len(approvals),
            events=[type(event).__name__ for event in events],
            context_tokens=context_tokens,
            passed=False,
            failure_reason=None,
            latency_seconds=perf_counter() - started,
            first_visible_seconds=first_visible,
            model_calls=len(backend.requests) - requests_before,
            parse_retries=sum(
                request[-1].content == INVALID_OUTPUT_MESSAGE
                for request in backend.requests[requests_before:]
            ),
            input_tokens=_total_usage(backend.responses[responses_before:], "input_tokens"),
            output_tokens=_total_usage(backend.responses[responses_before:], "output_tokens"),
            initial_input_tokens=(
                backend.responses[responses_before].input_tokens
                if len(backend.responses) > responses_before
                else None
            ),
            compactions=sum(
                bool(request) and request[0].content == SUMMARY_INSTRUCTION
                for request in backend.requests[requests_before:]
            ),
        )
        reason = (
            "compact_failed" if compact_failed else _step_failure(step, result, events, workspace)
        )
        sent = backend.requests[requests_before:]
        if reason is None and any(
            prompt in message.content
            for prompt in earlier_cancelled
            for request in sent
            for message in request
        ):
            reason = "cancelled_prompt_reached_the_model_again"
        if reason is None and case.context_grows and not step.cancel:
            if context_tokens is None:
                reason = "context_not_reported"
            elif context_tokens <= previous_context:
                reason = "context_did_not_grow"
            else:
                previous_context = context_tokens
        results.append(
            result.model_copy(update={"passed": reason is None, "failure_reason": reason})
        )
        if reason is not None:
            break
    return results


class _RequestRecorder(_RecordingBackend):
    """Also keep every request, so checks can see exactly what the model was sent."""

    def __init__(self, backend: LLMBackend) -> None:
        super().__init__(backend)
        self.requests: list[list[Message]] = []
        self.malform_next = False  # replace the next reply with prose, as models sometimes do

    def __getattr__(self, name):
        if name != "stream_chat":
            raise AttributeError(name)
        stream = getattr(self.backend, name)

        def record(messages, **options):
            self.requests.append(list(messages))
            response = stream(messages, **options)
            self.responses.append(response)
            return response

        return record

    def chat(self, messages: list[Message], **options) -> LLMResponse:
        self.requests.append(list(messages))
        response = super().chat(messages, **options)
        if not self.malform_next:
            return response
        self.malform_next = False
        return LLMResponse(MALFORMED_ACTION, response.model, response.finish_reason, 0)

    def context_usage(self, messages: list[Message]):
        measure = getattr(self.backend, "context_usage", None)
        return measure(messages) if measure is not None else None


def _run_case(
    backend: LLMBackend,
    case: InteractiveEvalCase,
    run: int,
    timeout_seconds: float,
    max_tokens: int = 1024,
    temperature: float = 0.6,
) -> InteractiveRunResult:
    started = perf_counter()
    recorder = _RequestRecorder(backend)
    with tempfile.TemporaryDirectory(prefix="jamie-interactive-eval-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        for name, content in case.files.items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        config = ProjectConfig(memory_path=str(root / "memory.sqlite3"))
        application = JamieApplication(config, backend_factory=lambda: recorder)

        async def drive() -> list[StepResult]:
            async with application:
                return await _run_steps(
                    application, recorder, case, workspace, timeout_seconds, max_tokens, temperature
                )

        steps = asyncio.run(drive())

    failed = next((index for index, step in enumerate(steps) if not step.passed), None)
    tokens = [r.output_tokens for r in recorder.responses if r.output_tokens is not None]
    return InteractiveRunResult(
        case_id=case.id,
        run=run,
        success=failed is None and len(steps) == len(case.steps),
        failed_step=None if failed is None else failed + 1,
        failure_reason=None if failed is None else steps[failed].failure_reason,
        model=next((r.model for r in recorder.responses if r.model), None),
        output_tokens=sum(tokens) if tokens else None,
        latency_seconds=perf_counter() - started,
        steps=steps,
        model_outputs=[response.content for response in recorder.responses],
    )


def run_interactive_evaluation(
    backend: LLMBackend,
    cases: list[InteractiveEvalCase],
    *,
    runs: int,
    timeout_seconds: float,
    output_dir: Path,
    on_result: Callable[[InteractiveRunResult, int], None] | None = None,
    environment: dict[str, JsonValue] | None = None,
    report_prefix: str = "interactive",
    max_tokens: int = 1024,
    temperature: float = 0.6,
) -> tuple[InteractiveEvalReport, Path]:
    started = datetime.now(UTC)
    report = InteractiveEvalReport(
        started_at=started.isoformat(),
        cases=cases,
        runs_per_case=runs,
        timeout_seconds=timeout_seconds,
        environment=environment or {},
    )
    path = new_report_path(output_dir, report_prefix, started)
    run_rounds(
        report,
        path,
        cases,
        runs,
        lambda case, run: _run_case(backend, case, run, timeout_seconds, max_tokens, temperature),
        on_result,
    )
    return report, path
