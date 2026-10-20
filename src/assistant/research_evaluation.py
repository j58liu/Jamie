"""Repeatable evaluations for cited web research behavior."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from assistant import __version__
from assistant.agent_evaluation import _RecordingBackend
from assistant.evaluation import (
    load_case_files,
    new_report_path,
    run_rounds,
)
from assistant.llm.base import LLMBackend
from assistant.research import ResearchError, ResearchResult, ResearchWorkflow, citation_urls
from assistant.tools.web import ExtractedPage, FetchedPage, SearchResult, WebError, extract_page


class ClaimExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    answer_contains: str = Field(min_length=1)
    evidence_contains: str = Field(min_length=1)
    supported_by: list[str] = Field(min_length=1)


class ResearchEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    question: str = Field(min_length=1)
    search_results: list[SearchResult] = Field(min_length=2)
    pages: dict[str, str]
    expected_claims: list[ClaimExpectation] = Field(min_length=1)
    min_sources: int = Field(default=2, ge=2, le=8)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.question.strip():
            raise ValueError("Research question must not be blank.")
        result_urls = {result.url for result in self.search_results}
        if not result_urls.issubset(self.pages):
            raise ValueError("Each search result requires a page fixture.")
        if self.min_sources > len(result_urls):
            raise ValueError("Minimum sources exceeds available search results.")
        for claim in self.expected_claims:
            if not set(claim.supported_by).issubset(result_urls):
                raise ValueError("Claim support URLs must be search result fixtures.")
            if not any(claim.evidence_contains in self.pages[url] for url in claim.supported_by):
                raise ValueError("Claim evidence must occur in a supporting page fixture.")
        return self


class ResearchRunResult(BaseModel):
    case_id: str
    run: int
    answer: str | None
    model: str | None
    output_tokens: int | None
    latency_seconds: float
    queries: list[str]
    source_urls: list[str]
    model_outputs: list[str]
    cross_check: str | None
    source_count: int
    citations_exist: bool
    citations_resolve: bool
    claims_supported: bool
    no_fake_urls: bool
    multiple_source_coverage: bool
    cited_urls: list[str]
    success: bool
    failure_reason: str | None
    error: str | None
    timings: dict[str, float] = Field(default_factory=dict)
    evidence_checks: int = 0


class ResearchEvalReport(BaseModel):
    schema_version: int = 1
    package_version: str = __version__
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "interrupted"] = "running"
    cases: list[ResearchEvalCase]
    runs_per_case: int
    total_duration_seconds: float = 0.0
    results: list[ResearchRunResult] = Field(default_factory=list)
    max_workers: int = 3
    fixture_delay_seconds: float = 0
    environment: dict[str, JsonValue] = Field(default_factory=dict)

    def summary(self) -> dict[str, JsonValue]:
        planned = len(self.cases) * self.runs_per_case
        completed = len(self.results)
        passes = sum(result.success for result in self.results)
        return {
            "models": sorted({result.model for result in self.results if result.model}),
            "cases": len(self.cases),
            "runs_per_case": self.runs_per_case,
            "planned_runs": planned,
            "completed_runs": completed,
            "passes": passes,
            "failures": completed - passes,
            "median_source_count": (
                statistics.median(result.source_count for result in self.results)
                if self.results
                else None
            ),
            "citations_exist": sum(result.citations_exist for result in self.results),
            "citations_resolve": sum(result.citations_resolve for result in self.results),
            "claims_supported": sum(result.claims_supported for result in self.results),
            "no_fake_urls": sum(result.no_fake_urls for result in self.results),
            "multiple_source_coverage": sum(
                result.multiple_source_coverage for result in self.results
            ),
            "median_latency_seconds": (
                statistics.median(result.latency_seconds for result in self.results)
                if self.results
                else None
            ),
            "output_tokens": (
                sum(
                    result.output_tokens
                    for result in self.results
                    if result.output_tokens is not None
                )
                if any(result.output_tokens is not None for result in self.results)
                else None
            ),
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.status == "completed" and passes == planned,
        }


class _FixtureWeb:
    """Serve fixed search results and pages instead of the network."""

    def __init__(
        self, search_results: list[SearchResult], pages: dict[str, str], delay_seconds: float = 0
    ) -> None:
        self.search_results = search_results
        self.pages = pages
        self.fetched: dict[str, FetchedPage] = {}
        self.delay_seconds = delay_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        time.sleep(self.delay_seconds)
        return self.search_results[:limit]

    def fetch(self, url: str) -> FetchedPage:
        time.sleep(self.delay_seconds)
        try:
            content = self.pages[url]
        except KeyError as exc:
            raise WebError("Page fixture not found.") from exc
        page = FetchedPage(
            requested_url=url,
            url=url,
            status_code=200,
            content_type="text/html",
            text=content,
            byte_count=len(content.encode()),
            truncated=False,
        )
        self.fetched[url] = page
        return page

    def extract(self, url: str, *, fallback_title: str = "") -> ExtractedPage:
        try:
            page = self.fetched[url]
        except KeyError as exc:
            raise WebError("Page fixture was not fetched.") from exc
        return extract_page(page, fallback_title=fallback_title)


def load_research_cases(path: Path | None = None) -> list[ResearchEvalCase]:
    return load_case_files(path, ResearchEvalCase, "research_cases", "research")


def _claims_supported(case: ResearchEvalCase, result: ResearchResult) -> bool:
    source_by_url = {source.url: source for source in result.sources}
    blocks = [block for block in result.answer.splitlines() if block.strip()]
    for claim in case.expected_claims:
        matching = [block for block in blocks if claim.answer_contains.lower() in block.lower()]
        supported = False
        for block in matching:
            block_urls = set(citation_urls(block))
            for url in block_urls.intersection(claim.supported_by):
                source = source_by_url.get(url)
                if source and claim.evidence_contains.lower() in source.text.lower():
                    supported = True
                    break
            if supported:
                break
        if not supported:
            return False
    return True


def _run_case(
    backend: LLMBackend,
    case: ResearchEvalCase,
    run: int,
    max_workers: int = 3,
    fixture_delay_seconds: float = 0,
) -> ResearchRunResult:
    started = perf_counter()
    answer = None
    result = None
    error = None
    recorder = _RecordingBackend(backend)
    try:
        result = ResearchWorkflow(
            recorder,
            _FixtureWeb(case.search_results, case.pages, fixture_delay_seconds),
            max_workers=max_workers,
        ).run(
            case.question,
            max_sources=len(case.search_results),
            min_sources=case.min_sources,
        )
        answer = result.answer
    except ResearchError as exc:
        error = str(exc)

    cited = citation_urls(answer or "")
    unique_cited = list(dict.fromkeys(cited))
    result_urls = {item.url for item in case.search_results}
    extracted_urls = {source.url for source in result.sources} if result else set()
    citations_exist = bool(cited)
    no_fake_urls = citations_exist and set(cited).issubset(result_urls)
    citations_resolve = citations_exist and set(cited).issubset(extracted_urls)
    hosts = {urlsplit(url).hostname for url in unique_cited}
    multiple_source_coverage = len(unique_cited) >= case.min_sources and len(hosts) >= 2
    claims_supported = result is not None and _claims_supported(case, result)
    success = all(
        (
            error is None,
            citations_exist,
            citations_resolve,
            claims_supported,
            no_fake_urls,
            multiple_source_coverage,
        )
    )
    if error:
        reason = "research_error"
    elif not citations_exist:
        reason = "citations_missing"
    elif not no_fake_urls:
        reason = "fake_url"
    elif not citations_resolve:
        reason = "citation_unresolved"
    elif not claims_supported:
        reason = "claim_unsupported"
    elif not multiple_source_coverage:
        reason = "insufficient_source_coverage"
    else:
        reason = None
    return ResearchRunResult(
        case_id=case.id,
        run=run,
        answer=answer,
        model=result.model
        if result
        else next((r.model for r in recorder.responses if r.model), None),
        output_tokens=sum(r.output_tokens or 0 for r in recorder.responses) or None,
        latency_seconds=perf_counter() - started,
        queries=result.queries if result else [],
        source_urls=[source.url for source in result.sources] if result else [],
        model_outputs=[response.content for response in recorder.responses],
        cross_check=result.cross_check if result else None,
        source_count=len(unique_cited),
        citations_exist=citations_exist,
        citations_resolve=citations_resolve,
        claims_supported=claims_supported,
        no_fake_urls=no_fake_urls,
        multiple_source_coverage=multiple_source_coverage,
        cited_urls=unique_cited,
        success=success,
        failure_reason=reason,
        error=error,
        timings=result.timings if result else {},
        evidence_checks=len(result.evidence_checks) if result else 0,
    )


def run_research_evaluation(
    backend: LLMBackend,
    cases: list[ResearchEvalCase],
    *,
    runs: int,
    output_dir: Path,
    on_result: Callable[[ResearchRunResult, int], None] | None = None,
    max_workers: int = 3,
    fixture_delay_seconds: float = 0,
    environment: dict[str, JsonValue] | None = None,
) -> tuple[ResearchEvalReport, Path]:
    started = datetime.now(UTC)
    report = ResearchEvalReport(
        started_at=started.isoformat(),
        cases=cases,
        runs_per_case=runs,
        max_workers=max_workers,
        fixture_delay_seconds=fixture_delay_seconds,
        environment=environment or {},
    )
    path = new_report_path(output_dir, "research", started)
    run_rounds(
        report,
        path,
        cases,
        runs,
        lambda case, run: _run_case(backend, case, run, max_workers, fixture_delay_seconds),
        on_result,
    )
    return report, path
