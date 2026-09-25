from collections import deque

import pytest

from assistant.agent.loop import run_agent
from assistant.llm.base import LLMResponse
from assistant.research import ResearchError, ResearchWorkflow, citation_urls
from assistant.security import CallbackApprovalProvider
from assistant.tools import (
    ExtractedPage,
    FetchedPage,
    SearchResult,
    ToolRegistry,
    register_web_tools,
)


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024):
        self.calls.append((messages, temperature, max_tokens))
        return LLMResponse(
            self.responses.popleft(),
            model="test-model",
            output_tokens=10,
        )


class FixtureWeb:
    def __init__(self):
        self.results = [
            SearchResult(title="First source", url="https://one.example/fact_(one)", snippet="one"),
            SearchResult(title="Second source", url="https://two.test/fact", snippet="two"),
        ]
        self.pages = {
            result.url: FetchedPage(
                requested_url=result.url,
                url=result.url,
                status_code=200,
                content_type="text/html",
                text="body",
                byte_count=4,
                truncated=False,
            )
            for result in self.results
        }
        self.searches = []
        self.fetches = []

    def search(self, query, *, limit=5):
        self.searches.append((query, limit))
        return self.results[:limit]

    def fetch(self, url):
        self.fetches.append(url)
        return self.pages[url]

    def extract(self, url, *, fallback_title=""):
        text = "First supported fact." if "one.example" in url else "Second supported fact."
        return ExtractedPage(url=url, title=fallback_title, text=text)


def test_research_workflow_searches_cross_checks_and_renders_known_citations():
    backend = ScriptedBackend(
        [
            '{"queries":["first query","second query"]}',
            "The sources support separate facts [S1] [S2].",
            "First supported fact. [S1]\nSecond supported fact. [S2]",
        ]
    )
    web = FixtureWeb()
    result = ResearchWorkflow(backend, web).run("What are the facts?")
    assert result.queries == ["first query", "second query"]
    assert web.searches == [("first query", 5), ("second query", 5)]
    assert web.fetches == ["https://one.example/fact_(one)", "https://two.test/fact"]
    assert citation_urls(result.answer) == [
        "https://one.example/fact_(one)",
        "https://two.test/fact",
    ]
    assert "[S1]" not in result.answer
    assert result.cited_source_ids == ["S1", "S2"]
    assert result.output_tokens == 30
    assert len(backend.calls) == 3


def test_citation_parser_keeps_parentheses_inside_angle_bracket_destination():
    url = "https://example.com/wiki/Subject_(detail)"
    assert citation_urls(f"[Source](<{url}>)") == [url]


def test_research_workflow_retries_synthesis_that_writes_a_url():
    backend = ScriptedBackend(
        [
            '{"queries":["facts"]}',
            "Cross-check [S1] [S2].",
            "A fake citation https://fake.example/claim",
            "First supported fact. [S1]\nSecond supported fact. [S2]",
        ]
    )
    result = ResearchWorkflow(backend, FixtureWeb()).run("What are the facts?")
    assert "fake.example" not in result.answer
    assert len(backend.calls) == 4


def test_research_workflow_rejects_unknown_citations_after_retry():
    backend = ScriptedBackend(
        [
            '{"queries":["facts"]}',
            "Cross-check [S1] [S2].",
            "Unsupported [S9].",
            "Still unsupported [S9].",
        ]
    )
    with pytest.raises(ResearchError, match="unknown sources"):
        ResearchWorkflow(backend, FixtureWeb()).run("What are the facts?")


@pytest.mark.parametrize(
    "plan",
    [
        "not json",
        '{"queries":[]}',
        '{"queries":["same","same"]}',
        '{"queries":["ok"],"extra":true}',
    ],
)
def test_research_workflow_rejects_invalid_search_plans(plan: str):
    backend = ScriptedBackend([plan])
    with pytest.raises(ResearchError, match="invalid search plan"):
        ResearchWorkflow(backend, FixtureWeb()).run("Question?")


def _research_registry(backend, web=None):
    registry = ToolRegistry(approvals=CallbackApprovalProvider(lambda request: True))
    return register_web_tools(registry, web or FixtureWeb(), backend)


def test_research_tool_ends_the_agent_turn_with_the_cited_report():
    backend = ScriptedBackend(
        [
            '{"action":"tool","tool":"research","args":{"question":"What are the facts?"}}',
            '{"queries":["facts"]}',
            "Cross-check [S1] [S2].",
            "First supported fact. [S1]\nSecond supported fact. [S2]",
        ]
    )

    answer = run_agent(backend, _research_registry(backend), "Research the facts.")

    assert citation_urls(answer) == ["https://one.example/fact_(one)", "https://two.test/fact"]
    assert len(backend.calls) == 4  # one agent action; the model never retypes the report


def test_research_failure_is_reported_to_the_model_as_a_tool_failure():
    backend = ScriptedBackend(["not a plan"])

    result = _research_registry(backend).execute("research", {"question": "What are the facts?"})

    assert not result.success and not result.final
    assert "invalid search plan" in result.output


def test_research_sources_reach_the_model_without_page_links():
    class LinkedWeb(FixtureWeb):
        def extract(self, url, *, fallback_title=""):
            text = "A [supported fact](<https://tracker.example/x>) is here."
            return ExtractedPage(url=url, title=fallback_title, text=text)

    backend = ScriptedBackend(
        ['{"queries":["facts"]}', "Cross-check [S1] [S2].", "Fact. [S1] [S2]"]
    )

    ResearchWorkflow(backend, LinkedWeb()).run("What are the facts?")

    cross_check_request = backend.calls[1][0][-1].content
    assert "A supported fact is here." in cross_check_request
    assert "tracker.example" not in cross_check_request
