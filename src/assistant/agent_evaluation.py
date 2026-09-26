"""End-to-end evaluations of the model, agent loop, and read-only tools."""

import statistics
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePath
from time import perf_counter
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.agent.loop import AgentError, ToolResult, run_agent
from assistant.evaluation import (
    load_case_files,
    new_report_path,
    run_rounds,
)
from assistant.llm.base import LLMBackend, LLMError, LLMResponse, Message
from assistant.tools.registry import ToolRegistry, create_read_only_registry


class AgentEvalCase(BaseModel):
    """A workspace, task, and checks for one complete agent run."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    prompt: str = Field(min_length=1)
    files: dict[str, str]
    expected_tool: Literal["read_file", "list_directory", "find_files", "search_files"]
    expected_args: dict[str, JsonValue] = Field(default_factory=dict)
    tool_output_contains: list[str] = Field(min_length=1)
    answer_contains: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        if not self.files:
            raise ValueError("Agent cases require at least one workspace file.")
        for name in self.files:
            path = PurePath(name)
            if not name.strip() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Fixture paths must stay within the workspace.")
        values = [*self.tool_output_contains, *self.answer_contains]
        if any(not value for value in values):
            raise ValueError("Expected text must not be empty.")
        return self


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    success: bool
    output: str


class AgentRunResult(BaseModel):
    case_id: str
    run: int
    answer: str | None
    model: str | None
    output_tokens: int | None
    latency_seconds: float
    model_outputs: list[str]
    tool_calls: list[ToolCall]
    expected_tool_called: bool
    correct_tool_output: bool
    answer_check: bool
    success: bool
    failure_reason: str | None
    error: str | None


class AgentEvalReport(BaseModel):
    schema_version: int = 1
    package_version: str = __version__
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "interrupted"] = "running"
    cases: list[AgentEvalCase]
    runs_per_case: int
    timeout_seconds: float
    total_duration_seconds: float = 0.0
    results: list[AgentRunResult] = Field(default_factory=list)

    def summary(self) -> dict[str, JsonValue]:
        planned = len(self.cases) * self.runs_per_case
        completed = len(self.results)
        passes = sum(result.success for result in self.results)
        tool_calls = sum(result.expected_tool_called for result in self.results)
        tool_outputs = sum(result.correct_tool_output for result in self.results)
        answer_checks = sum(result.answer_check for result in self.results)
        token_counts = [r.output_tokens for r in self.results if r.output_tokens is not None]
        return {
            "models": sorted({r.model for r in self.results if r.model is not None}),
            "cases": len(self.cases),
            "runs_per_case": self.runs_per_case,
            "planned_runs": planned,
            "completed_runs": completed,
            "passes": passes,
            "failures": completed - passes,
            "expected_tool_calls": tool_calls,
            "correct_tool_outputs": tool_outputs,
            "answer_checks": answer_checks,
            "median_latency_seconds": (
                statistics.median(r.latency_seconds for r in self.results) if self.results else None
            ),
            "output_tokens": sum(token_counts) if token_counts else None,
            "token_usage_reported_runs": len(token_counts),
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.status == "completed" and passes == planned,
        }


class _RecordingBackend:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend
        self.responses: list[LLMResponse] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        try:
            response = self.backend.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as exc:
            if exc.response is not None:
                self.responses.append(exc.response)
            raise
        self.responses.append(response)
        return response


class _RecordingTools:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.calls: list[ToolCall] = []

    def instructions(self) -> str:
        return self.registry.instructions()

    def execute(self, tool: str, args: dict[str, Any]) -> ToolResult:
        result = self.registry.execute(tool, args)
        self.calls.append(
            ToolCall(tool=tool, args=args, success=result.success, output=result.output)
        )
        return result


def load_agent_cases(path: Path | None = None) -> list[AgentEvalCase]:
    return load_case_files(path, AgentEvalCase, "agent_cases", "agent")


def _contains_args(actual: dict[str, Any], expected: dict[str, JsonValue]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _run_case(
    backend: LLMBackend,
    case: AgentEvalCase,
    run: int,
    timeout_seconds: float,
) -> AgentRunResult:
    started = perf_counter()
    answer = None
    error = None
    with tempfile.TemporaryDirectory(prefix="jamie-agent-eval-") as directory:
        root = Path(directory)
        for name, content in case.files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        recording_backend = _RecordingBackend(backend)
        tools = _RecordingTools(create_read_only_registry(root))
        try:
            answer = run_agent(
                recording_backend,
                tools,
                case.prompt,
                timeout_seconds=timeout_seconds,
            )
        except (AgentError, LLMError) as exc:
            error = str(exc)

    matching_calls = [
        call
        for call in tools.calls
        if call.tool == case.expected_tool and _contains_args(call.args, case.expected_args)
    ]
    expected_tool_called = bool(matching_calls)
    correct_tool_output = any(
        call.success and all(text in call.output for text in case.tool_output_contains)
        for call in matching_calls
    )
    answer_check = answer is not None and all(text in answer for text in case.answer_contains)
    success = error is None and expected_tool_called and correct_tool_output and answer_check
    if error is not None:
        reason = "agent_error"
    elif not expected_tool_called:
        reason = "expected_tool_not_called"
    elif not correct_tool_output:
        reason = "tool_output_mismatch"
    elif not answer_check:
        reason = "answer_mismatch"
    else:
        reason = None
    responses = recording_backend.responses
    token_counts = [
        response.output_tokens for response in responses if response.output_tokens is not None
    ]
    return AgentRunResult(
        case_id=case.id,
        run=run,
        answer=answer,
        model=next((response.model for response in responses if response.model), None),
        output_tokens=sum(token_counts) if token_counts else None,
        latency_seconds=perf_counter() - started,
        model_outputs=[response.content for response in responses],
        tool_calls=tools.calls,
        expected_tool_called=expected_tool_called,
        correct_tool_output=correct_tool_output,
        answer_check=answer_check,
        success=success,
        failure_reason=reason,
        error=error,
    )


def run_agent_evaluation(
    backend: LLMBackend,
    cases: list[AgentEvalCase],
    *,
    runs: int,
    timeout_seconds: float,
    output_dir: Path,
    on_result: Callable[[AgentRunResult, int], None] | None = None,
) -> tuple[AgentEvalReport, Path]:
    started = datetime.now(UTC)
    report = AgentEvalReport(
        started_at=started.isoformat(),
        cases=cases,
        runs_per_case=runs,
        timeout_seconds=timeout_seconds,
    )
    path = new_report_path(output_dir, "agent", started)
    run_rounds(
        report,
        path,
        cases,
        runs,
        lambda case, run: _run_case(backend, case, run, timeout_seconds),
        on_result,
    )
    return report, path
