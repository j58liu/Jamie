"""Deterministic search, retrieval, cross-checking, and cited synthesis."""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from assistant.evaluation import parse_json
from assistant.llm.base import LLMBackend, LLMError, Message
from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError
from assistant.tools.web import ExtractedPage, FetchedPage, SearchResult, WebError

MAX_QUESTION_CHARACTERS = 2_000
MAX_QUERIES = 4
MAX_SOURCES = 8
SOURCE_EXCERPT_CHARACTERS = 2_500
TOTAL_SOURCE_CHARACTERS = 12_000
_SOURCE_REFERENCE = re.compile(r"\[S([1-9][0-9]*)\]")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((?:<([^>]+)>|([^)]*))\)")
_LINK_TEXT = re.compile(r"\[([^\]]*)\]\((?:<[^>]+>|[^)]*)\)")
_RAW_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class ResearchError(RuntimeError):
    """A research run could not produce a supported cited answer."""


class ResearchWeb(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...

    def fetch(self, url: str) -> FetchedPage: ...

    def extract(self, url: str, *, fallback_title: str = "") -> ExtractedPage: ...


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=MAX_QUERIES)

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        cleaned = [query.strip() for query in queries]
        if any(not query for query in cleaned):
            raise ValueError("Search queries must not be blank.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Search queries must be unique.")
        return cleaned


class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    url: str
    text: str


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    queries: list[str]
    answer: str
    sources: list[ResearchSource]
    cited_source_ids: list[str]
    cross_check: str
    model_outputs: list[str]
    output_tokens: int | None
    model: str | None


def _strip_model_wrapper(content: str) -> str:
    value = content.strip()
    if value.startswith("<think>"):
        end = value.find("</think>")
        if end < 0:
            raise ResearchError("Model search plan has an incomplete think block.")
        value = value[end + len("</think>") :].strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ResearchError("Model search plan has an incomplete JSON fence.")
        if lines[0].strip() not in {"```", "```json"}:
            raise ResearchError("Model search plan uses an unsupported fence.")
        value = "\n".join(lines[1:-1]).strip()
    return value


def _parse_plan(content: str) -> SearchPlan:
    try:
        return SearchPlan.model_validate(parse_json(_strip_model_wrapper(content)))
    except (ValueError, ResearchError) as exc:
        raise ResearchError(f"Model returned an invalid search plan: {exc}") from exc


def _source_payload(sources: list[ResearchSource]) -> str:
    excerpt_limit = min(SOURCE_EXCERPT_CHARACTERS, TOTAL_SOURCE_CHARACTERS // len(sources))
    return json.dumps(
        [
            {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                # Page links would tempt the model to write URLs, which synthesis rejects.
                "text": _LINK_TEXT.sub(r"\1", source.text)[:excerpt_limit],
            }
            for source in sources
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def citation_urls(answer: str) -> list[str]:
    """Return HTTP(S) destinations from Markdown citations in appearance order."""
    urls: list[str] = []
    for match in _MARKDOWN_LINK.finditer(answer):
        destination = match.group(1) if match.group(1) is not None else (match.group(2) or "")
        if destination.startswith(("http://", "https://")):
            urls.append(destination)
    return urls


def _render_answer(
    raw_answer: str, sources: list[ResearchSource], min_sources: int
) -> tuple[str, list[str]]:
    if _RAW_URL.search(raw_answer) or _MARKDOWN_LINK.search(raw_answer):
        raise ResearchError("Model synthesis must cite source IDs instead of writing links.")
    by_id = {source.id: source for source in sources}
    references = _SOURCE_REFERENCE.findall(raw_answer)
    if not references:
        raise ResearchError("Model synthesis contains no source citations.")
    cited_ids = list(dict.fromkeys(f"S{number}" for number in references))
    unknown = [source_id for source_id in cited_ids if source_id not in by_id]
    if unknown:
        raise ResearchError(f"Model synthesis cites unknown sources: {', '.join(unknown)}")
    if len(cited_ids) < min_sources:
        raise ResearchError(f"Model synthesis must cite at least {min_sources} sources.")

    def replace(match: re.Match[str]) -> str:
        source = by_id[f"S{match.group(1)}"]
        label = source.title.replace("[", "").replace("]", "") or source.id
        return f"[{label}](<{source.url}>)"

    return _SOURCE_REFERENCE.sub(replace, raw_answer).strip(), cited_ids


class ResearchWorkflow:
    """Run a bounded research sequence using a model and web client."""

    def __init__(self, backend: LLMBackend, web: ResearchWeb) -> None:
        self._backend = backend
        self._web = web

    def _chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        outputs: list[str],
        usage: list[int],
        models: list[str],
    ) -> str:
        try:
            response = self._backend.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as exc:
            raise ResearchError(str(exc)) from exc
        outputs.append(response.content)
        if response.output_tokens is not None:
            usage.append(response.output_tokens)
        if response.model:
            models.append(response.model)
        return response.content

    def run(
        self,
        question: str,
        *,
        max_queries: int = 3,
        results_per_query: int = 5,
        max_sources: int = 5,
        min_sources: int = 2,
    ) -> ResearchResult:
        question = question.strip()
        if not question:
            raise ResearchError("Research question must not be blank.")
        if len(question) > MAX_QUESTION_CHARACTERS:
            raise ResearchError("Research question is too long.")
        if isinstance(max_queries, bool) or not 1 <= max_queries <= MAX_QUERIES:
            raise ResearchError(f"Search count must be between 1 and {MAX_QUERIES}.")
        if isinstance(max_sources, bool) or not 2 <= max_sources <= MAX_SOURCES:
            raise ResearchError(f"Source count must be between 2 and {MAX_SOURCES}.")
        if isinstance(min_sources, bool) or not 2 <= min_sources <= max_sources:
            raise ResearchError("Minimum sources must be between 2 and the source limit.")
        if isinstance(results_per_query, bool) or not 1 <= results_per_query <= 10:
            raise ResearchError("Results per search must be between 1 and 10.")

        outputs: list[str] = []
        usage: list[int] = []
        models: list[str] = []
        plan_content = self._chat(
            [
                Message(
                    role="system",
                    content=(
                        "Create a concise web search plan. Return only JSON with one key, queries, "
                        f"containing 1 to {max_queries} distinct search query strings."
                    ),
                ),
                Message(role="user", content=question),
            ],
            temperature=0.2,
            max_tokens=512,
            outputs=outputs,
            usage=usage,
            models=models,
        )
        plan = _parse_plan(plan_content)
        queries = plan.queries[:max_queries]

        candidates: list[SearchResult] = []
        seen_urls: set[str] = set()
        search_errors: list[str] = []
        for query in queries:
            try:
                results = self._web.search(query, limit=results_per_query)
            except WebError as exc:
                search_errors.append(str(exc))
                continue
            for result in results:
                if result.url not in seen_urls:
                    candidates.append(result)
                    seen_urls.add(result.url)
        if not candidates:
            detail = f": {'; '.join(search_errors)}" if search_errors else ""
            raise ResearchError(f"Search returned no usable sources{detail}")

        sources: list[ResearchSource] = []
        retrieved_urls: set[str] = set()
        for candidate in candidates:
            if len(sources) >= max_sources:
                break
            try:
                fetched = self._web.fetch(candidate.url)
                extracted = self._web.extract(fetched.url, fallback_title=candidate.title)
            except WebError:
                continue
            if extracted.url in retrieved_urls:
                continue
            retrieved_urls.add(extracted.url)
            sources.append(
                ResearchSource(
                    id=f"S{len(sources) + 1}",
                    title=extracted.title or candidate.title,
                    url=extracted.url,
                    text=extracted.text,
                )
            )
        if len(sources) < min_sources:
            raise ResearchError(
                f"Research retrieved {len(sources)} usable sources; "
                f"at least {min_sources} are required."
            )

        payload = _source_payload(sources)
        cross_check = self._chat(
            [
                Message(
                    role="system",
                    content=(
                        "Cross-check the supplied sources for the question. Identify agreements, "
                        "differences, and unsupported statements. Refer to sources only by IDs "
                        "such as [S1]. Do not write URLs."
                    ),
                ),
                Message(role="user", content=f"Question: {question}\nSources: {payload}"),
            ],
            temperature=0.2,
            max_tokens=768,
            outputs=outputs,
            usage=usage,
            models=models,
        ).strip()

        synthesis_messages = [
            Message(
                role="system",
                content=(
                    "Answer from the supplied evidence only. Put a source ID such as [S1] directly "
                    "after every factual claim. Use at least two different source IDs. Do not "
                    "write URLs or Markdown links. State uncertainty or disagreement plainly."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question: {question}\nCross-check notes: {cross_check}\nSources: {payload}"
                ),
            ),
        ]
        raw_answer = self._chat(
            synthesis_messages,
            temperature=0.2,
            max_tokens=1536,
            outputs=outputs,
            usage=usage,
            models=models,
        )
        try:
            answer, cited_ids = _render_answer(raw_answer, sources, min_sources)
        except ResearchError as first_error:
            retry = self._chat(
                [
                    Message(
                        role="system",
                        content=(
                            "Answer from the supplied evidence only. Put a valid source ID "
                            "directly after every factual claim and cite at least two source IDs. "
                            "Return only the answer. Do not write URLs or Markdown links."
                        ),
                    ),
                    Message(
                        role="user",
                        content=(
                            f"Question: {question}\nCitation issue to correct: {first_error}\n"
                            f"Cross-check notes: {cross_check}\nSources: {payload}"
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=1536,
                outputs=outputs,
                usage=usage,
                models=models,
            )
            answer, cited_ids = _render_answer(retry, sources, min_sources)

        return ResearchResult(
            question=question,
            queries=queries,
            answer=answer,
            sources=sources,
            cited_source_ids=cited_ids,
            cross_check=cross_check,
            model_outputs=outputs,
            output_tokens=sum(usage) if usage else None,
            model=models[0] if models else None,
        )


class ResearchTool:
    """Run the whole research workflow as one agent action, in its own model context.

    The cited report ends the turn as the answer, so a small model never has to retype it.
    """

    name = "research"
    description = (
        "Research a question on the public web and answer it with citations. Use this "
        "whenever the user asks you to research, look up, or find out about a topic."
    )
    risk = PermissionLevel.EXTERNAL
    ends_turn = True

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")

        question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)

    input_schema = Input

    def __init__(self, backend: LLMBackend, web: ResearchWeb) -> None:
        self._workflow = ResearchWorkflow(backend, web)

    def execute(self, arguments: Input) -> str:
        try:
            return self._workflow.run(arguments.question).answer
        except ResearchError as exc:
            raise ToolError(str(exc)) from exc
