"""Score the memory classifier on labelled messages with the running model.

What matters is the learn decision (A or B, above the confidence cutoff): a false save
stores something the user never said; a miss forgets something they did. Both are
reported at several cutoffs so the default can be chosen from data.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from assistant.evaluation import EvaluationError, new_report_path, save_report
from assistant.knowledge.extraction import LABELS, LEARN, classify
from assistant.llm.base import LLMBackend

DEFAULT_CASES = Path("evals/memory_classification/cases.json")
CUTOFFS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ClassificationCase(BaseModel):
    message: str
    label: str


class ClassificationResult(BaseModel):
    message: str
    expected: str
    predicted: str
    confidence: float
    seconds: float


class ClassificationReport(BaseModel):
    model: str | None = None
    started_at: str
    results: list[ClassificationResult]

    def summary(self) -> dict:
        total = len(self.results)
        correct = sum(r.predicted == r.expected for r in self.results)
        should = [r for r in self.results if r.expected in LEARN]
        should_not = [r for r in self.results if r.expected not in LEARN]
        cutoffs = {}
        for cutoff in CUTOFFS:

            def learns(r: ClassificationResult, cutoff: float = cutoff) -> bool:
                return r.predicted in LEARN and r.confidence >= cutoff

            cutoffs[f"{cutoff:.1f}"] = {
                "false_saves": sum(learns(r) for r in should_not),
                "missed": sum(not learns(r) for r in should),
            }
        seconds = sorted(r.seconds for r in self.results)
        return {
            "cases": total,
            "label_accuracy": correct / total if total else None,
            "should_learn": len(should),
            "should_not_learn": len(should_not),
            "cutoffs": cutoffs,
            "median_seconds": seconds[total // 2] if total else None,
        }


def load_classification_cases(path: Path | None = None) -> list[ClassificationCase]:
    path = path or DEFAULT_CASES
    try:
        cases = [ClassificationCase.model_validate(c) for c in json.loads(path.read_text())]
    except (OSError, ValueError, ValidationError) as exc:
        raise EvaluationError(f"Cannot read classification cases from {path}: {exc}") from exc
    unknown = sorted({c.label for c in cases} - set(LABELS))
    if unknown or not cases:
        raise EvaluationError(f"Cases need labels from {', '.join(LABELS)}; found {unknown}.")
    return cases


def run_classification_evaluation(
    backend: LLMBackend, cases: list[ClassificationCase], *, output_dir: Path, model: str | None
) -> tuple[ClassificationReport, Path]:
    started = datetime.now(UTC)
    results = []
    for case in cases:
        before = time.perf_counter()
        predicted, confidence = classify(backend, case.message)
        results.append(
            ClassificationResult(
                message=case.message,
                expected=case.label,
                predicted=predicted,
                confidence=round(confidence, 4),
                seconds=round(time.perf_counter() - before, 3),
            )
        )
    report = ClassificationReport(model=model, started_at=started.isoformat(), results=results)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = new_report_path(output_dir, "memory-classification", started)
    save_report(path, report)
    return report, path
