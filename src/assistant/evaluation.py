"""Fixed-prompt model evaluations with deterministic scoring and saved results."""

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.llm.base import LLMBackend, LLMError, LLMResponse, Message


class EvaluationError(ValueError):
    """Evaluation inputs or result files could not be read or written."""


class EvalCase(BaseModel):
    """An unchanged prompt and a deterministic JSON or text expectation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    prompt: str = Field(min_length=1)
    expected: JsonValue
    format: Literal["json", "text"] = "json"

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        if self.format == "text" and not isinstance(self.expected, str):
            raise ValueError("Text cases require a string expectation.")
        return self


class RunResult(BaseModel):
    case_id: str
    run: int
    raw_output: str | None
    model: str | None
    finish_reason: str | None
    output_tokens: int | None
    latency_seconds: float
    parse_success: bool | None
    instruction_following: bool
    success: bool
    failure_reason: str | None
    error: str | None


class EvalReport(BaseModel):
    schema_version: int = 1
    package_version: str = __version__
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "interrupted"] = "running"
    cases: list[EvalCase]
    cases_sha256: str
    runs_per_case: int
    temperature: float
    max_tokens: int
    request_timeout_seconds: float | None = None
    total_duration_seconds: float = 0.0
    results: list[RunResult] = Field(default_factory=list)

    def summary(self) -> dict[str, JsonValue]:
        json_results = [result for result in self.results if result.parse_success is not None]
        token_counts = [r.output_tokens for r in self.results if r.output_tokens is not None]
        passes = sum(result.success for result in self.results)
        following = sum(result.instruction_following for result in self.results)
        parses = sum(result.parse_success is True for result in json_results)
        planned = len(self.cases) * self.runs_per_case
        return {
            "models": sorted({r.model for r in self.results if r.model is not None}),
            "cases": len(self.cases),
            "runs_per_case": self.runs_per_case,
            "request_timeout_seconds": self.request_timeout_seconds,
            "planned_runs": planned,
            "completed_runs": len(self.results),
            "passes": passes,
            "failures": len(self.results) - passes,
            "json_attempts": len(json_results),
            "parse_successes": parses,
            "parse_success_rate": parses / len(json_results) if json_results else None,
            "instruction_following_passes": following,
            "instruction_following_rate": following / len(self.results) if self.results else None,
            "median_latency_seconds": (
                statistics.median(r.latency_seconds for r in self.results) if self.results else None
            ),
            "output_tokens": sum(token_counts) if token_counts else None,
            "token_usage_reported_runs": len(token_counts),
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.status == "completed" and passes == planned,
        }


def default_cases_path() -> Path:
    """Use checkout cases, or the copies included in an installed wheel."""
    checkout = Path(__file__).resolve().parents[2] / "evals" / "cases"
    return checkout if checkout.is_dir() else Path(__file__).with_name("eval_cases")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside the finite range.")
    return parsed


def parse_json(text: str) -> JsonValue:
    """Parse the complete response without extracting fences or repairing JSON."""
    return json.loads(
        text,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_object,
    )


def load_cases(path: Path | None = None) -> list[EvalCase]:
    source = path if path is not None else default_cases_path()
    try:
        files = sorted(source.glob("*.json")) if source.is_dir() else [source]
        cases = []
        for file in files:
            data = parse_json(file.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                raise ValueError(f"{file} must contain a non-empty array of cases.")
            cases.extend(EvalCase.model_validate(item) for item in data)
        if not cases:
            raise ValueError("No evaluation cases found.")
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation case IDs must be unique across all files.")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvaluationError(f"Cannot load evaluation cases from {source}: {exc}") from exc
    return sorted(cases, key=lambda case: case.id)


def _matches(actual: JsonValue, expected: JsonValue) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            key in actual and _matches(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _matches(item, wanted) for item, wanted in zip(actual, expected, strict=True)
        )
    return actual == expected


def score_response(
    case: EvalCase, response: LLMResponse | None, *, error: str | None = None
) -> tuple[bool | None, bool, bool, str | None]:
    parse_success = False if case.format == "json" else None
    following = False
    if response is not None:
        if case.format == "json":
            try:
                value = parse_json(response.content)
            except (ValueError, RecursionError):
                pass
            else:
                parse_success = True
                following = _matches(value, case.expected)
        else:
            following = response.content.strip() == case.expected

    if response is None:
        reason = "request_error" if error is not None else "empty_output"
    elif response.finish_reason == "length":
        reason = "token_limit"
    elif not response.content.strip():
        reason = "empty_output"
    elif error is not None:
        reason = "request_error"
    elif parse_success is False:
        reason = "invalid_json"
    elif not following:
        reason = "expectation_mismatch"
    else:
        reason = None
    return parse_success, following, reason is None, reason


def save_report(path: Path, report: EvalReport) -> None:
    """Replace a checkpoint atomically, retaining the previous report on failure."""
    payload = report.model_dump(mode="json")
    payload["summary"] = report.summary()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        raise EvaluationError(f"Cannot save evaluation report to {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_evaluation(
    backend: LLMBackend,
    cases: list[EvalCase],
    *,
    runs: int = 5,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    output_dir: Path = Path("evals/results"),
    request_timeout_seconds: float | None = None,
    on_result: Callable[[RunResult, int], None] | None = None,
) -> tuple[EvalReport, Path]:
    """Run all cases once per round, with fresh conversations and no retries."""
    if not cases or len({case.id for case in cases}) != len(cases):
        raise EvaluationError("Provide non-empty cases with unique IDs.")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise EvaluationError("Runs must be a positive integer.")
    if not math.isfinite(temperature) or temperature < 0:
        raise EvaluationError("Temperature must be finite and non-negative.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise EvaluationError("Max tokens must be a positive integer.")

    started = datetime.now(UTC)
    case_data = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    report = EvalReport(
        started_at=started.isoformat(),
        cases=cases,
        cases_sha256=hashlib.sha256(case_data.encode("utf-8")).hexdigest(),
        runs_per_case=runs,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout_seconds=request_timeout_seconds,
    )
    path = output_dir / f"model-{started.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(f"Cannot create results directory {output_dir}: {exc}") from exc
    began = perf_counter()
    save_report(path, report)
    try:
        for run in range(1, runs + 1):
            for case in cases:
                response = None
                error = None
                request_started = perf_counter()
                try:
                    response = backend.chat(
                        [Message(role="user", content=case.prompt)],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except LLMError as exc:
                    response, error = exc.response, str(exc)
                latency = perf_counter() - request_started
                parsed, following, success, reason = score_response(case, response, error=error)
                result = RunResult(
                    case_id=case.id,
                    run=run,
                    raw_output=response.content if response is not None else None,
                    model=response.model if response is not None else None,
                    finish_reason=response.finish_reason if response is not None else None,
                    output_tokens=response.output_tokens if response is not None else None,
                    latency_seconds=latency,
                    parse_success=parsed,
                    instruction_following=following,
                    success=success,
                    failure_reason=reason,
                    error=error,
                )
                report.results.append(result)
                report.total_duration_seconds = perf_counter() - began
                save_report(path, report)
                if on_result is not None:
                    on_result(result, len(report.results))
    except KeyboardInterrupt:
        report.status = "interrupted"
    else:
        report.status = "completed"
    report.finished_at = datetime.now(UTC).isoformat()
    report.total_duration_seconds = perf_counter() - began
    save_report(path, report)
    return report, path
