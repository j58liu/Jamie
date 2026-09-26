"""Deterministic FTS5 retrieval evaluations for persistent memories."""

from __future__ import annotations

import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.evaluation import (
    EvaluationError,
    load_case_files,
    new_report_path,
    save_report,
)
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


def load_memory_cases(path: Path | None = None) -> list[MemoryEvalCase]:
    return load_case_files(path, MemoryEvalCase, "memory_cases", "memory")


def run_memory_evaluation(
    cases: list[MemoryEvalCase], *, output_dir: Path
) -> tuple[MemoryEvalReport, Path]:
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
    report = MemoryEvalReport(
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        cases=cases,
        results=results,
        total_duration_seconds=perf_counter() - began,
    )
    path = new_report_path(output_dir, "memory", started)
    save_report(path, report)
    return report, path
