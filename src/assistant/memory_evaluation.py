"""Deterministic FTS5 retrieval evaluations for persistent memories."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.evaluation import EvaluationError, parse_json
from assistant.memory import MemoryStore, MemoryStoreError


class MemoryEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    memories: list[str] = Field(min_length=2)
    query: str = Field(min_length=1)
    expected_contains: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_case(self):
        if not self.query.strip() or any(not memory.strip() for memory in self.memories):
            raise ValueError("Memory cases must contain non-blank text.")
        matches = sum(
            self.expected_contains.casefold() in memory.casefold() for memory in self.memories
        )
        if matches != 1:
            raise ValueError("Expected text must identify exactly one seeded memory.")
        return self


class MemoryEvalResult(BaseModel):
    case_id: str
    retrieved: list[str]
    expected_rank: int | None
    success: bool


class MemoryEvalReport(BaseModel):
    schema_version: int = 1
    package_version: str = __version__
    started_at: str
    finished_at: str
    status: Literal["completed"] = "completed"
    cases: list[MemoryEvalCase]
    cases_sha256: str
    results: list[MemoryEvalResult]
    total_duration_seconds: float

    def summary(self) -> dict[str, JsonValue]:
        hits = sum(result.success for result in self.results)
        reciprocal_ranks = [
            1 / result.expected_rank if result.expected_rank is not None else 0
            for result in self.results
        ]
        return {
            "cases": len(self.cases),
            "hits": hits,
            "misses": len(self.results) - hits,
            "recall_at_k": hits / len(self.results) if self.results else None,
            "mean_reciprocal_rank": (
                statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None
            ),
            "success": hits == len(self.cases),
            "total_duration_seconds": self.total_duration_seconds,
        }


def default_memory_cases_path() -> Path:
    checkout = Path(__file__).resolve().parents[2] / "evals" / "memory_cases"
    return checkout if checkout.is_dir() else Path(__file__).with_name("memory_eval_cases")


def load_memory_cases(path: Path | None = None) -> list[MemoryEvalCase]:
    source = path if path is not None else default_memory_cases_path()
    try:
        files = sorted(source.glob("*.json")) if source.is_dir() else [source]
        cases: list[MemoryEvalCase] = []
        for file in files:
            data = parse_json(file.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                raise ValueError(f"{file} must contain a non-empty array of cases.")
            cases.extend(MemoryEvalCase.model_validate(item) for item in data)
        if not cases:
            raise ValueError("No memory evaluation cases found.")
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Memory evaluation case IDs must be unique.")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvaluationError(f"Cannot load memory evaluation cases from {source}: {exc}") from exc
    return sorted(cases, key=lambda case: case.id)


def _save_report(path: Path, report: MemoryEvalReport) -> None:
    payload = report.model_dump(mode="json")
    payload["summary"] = report.summary()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        raise EvaluationError(f"Cannot save memory report to {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_memory_evaluation(
    cases: list[MemoryEvalCase], *, output_dir: Path
) -> tuple[MemoryEvalReport, Path]:
    if not cases:
        raise EvaluationError("At least one memory evaluation case is required.")
    started = datetime.now(UTC)
    began = perf_counter()
    results: list[MemoryEvalResult] = []
    try:
        with tempfile.TemporaryDirectory(prefix="jamie-memory-eval-") as directory:
            for index, case in enumerate(cases):
                store = MemoryStore(Path(directory) / f"case-{index}.sqlite3")
                for memory in case.memories:
                    store.add_memory(memory, kind="evaluation")
                retrieved = store.search_memories(case.query, limit=case.top_k)
                texts = [record.content for record in retrieved]
                rank = next(
                    (
                        position
                        for position, content in enumerate(texts, start=1)
                        if case.expected_contains.casefold() in content.casefold()
                    ),
                    None,
                )
                results.append(
                    MemoryEvalResult(
                        case_id=case.id,
                        retrieved=texts,
                        expected_rank=rank,
                        success=rank is not None,
                    )
                )
    except MemoryStoreError as exc:
        raise EvaluationError(str(exc)) from exc
    finished = datetime.now(UTC)
    case_data = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    report = MemoryEvalReport(
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        cases=cases,
        cases_sha256=hashlib.sha256(case_data.encode()).hexdigest(),
        results=results,
        total_duration_seconds=perf_counter() - began,
    )
    path = output_dir / f"memory-{started.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(f"Cannot create results directory {output_dir}: {exc}") from exc
    _save_report(path, report)
    return report, path
