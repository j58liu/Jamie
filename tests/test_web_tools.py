import json
import socket

import httpx
import pytest

from assistant.security import CallbackApprovalProvider, PermissionLevel
from assistant.tools import WebClient, WebError, create_web_registry, validate_public_url


def public_resolver(host, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def test_search_parses_results_and_unwraps_redirects():
    search_html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fone.example%2Fpage">
        First result
      </a>
      <div class="result__snippet">Useful <b>summary</b>.</div>
    </div>
    <a class="result__a" href="https://two.example/article">Second result</a>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        assert request.url.params["q"] == "test query"
        return httpx.Response(200, text=search_html, headers={"content-type": "text/html"})

    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        results = client.search("test query", limit=2)
    assert [result.url for result in results] == [
        "https://one.example/page",
        "https://two.example/article",
    ]
    assert results[0].title == "First result"
    assert results[0].snippet == "Useful summary."


def test_fetch_then_extract_removes_scripts_and_keeps_readable_text():
    page_html = """
    <html><head><title>Example page</title><style>hidden</style></head>
    <body><script>ignore()</script><main><h1>Heading</h1><p>Hello world.</p></main></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=page_html, headers={"content-type": "text/html; charset=utf-8"}
        )

    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        fetched = client.fetch("https://example.com/page#fragment")
        extracted = client.extract("https://example.com/page")
    assert fetched.url == "https://example.com/page"
    assert extracted.title == "Example page"
    assert "Heading" in extracted.text
    assert "Hello world." in extracted.text
    assert "hidden" not in extracted.text
    assert "ignore" not in extracted.text


def test_fetch_validates_every_redirect_destination():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        with pytest.raises(WebError, match="public address"):
            client.fetch("https://example.com/start")


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/private",
        "http://localhost/private",
        "http://127.0.0.1/private",
        "http://169.254.169.254/latest/meta-data",
        "https://user:password@example.com/private",
    ],
)
def test_public_url_validation_rejects_unsafe_targets(url: str):
    with pytest.raises(WebError):
        validate_public_url(url, public_resolver)


def test_registry_requires_approval_for_network_then_extracts_without_another_prompt():
    search_html = '<a class="result__a" href="https://example.com/page">Example</a>'
    page_html = "<html><head><title>Example</title></head><body><p>Evidence.</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(200, text=search_html)
        return httpx.Response(200, text=page_html, headers={"content-type": "text/html"})

    approvals = []
    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        registry = create_web_registry(
            client=client,
            approvals=CallbackApprovalProvider(lambda request: approvals.append(request) or True),
        )
        search = registry.execute("web_search", {"query": "evidence"})
        fetched = registry.execute("fetch_url", {"url": "https://example.com/page"})
        extracted = registry.execute("extract_page", {"url": "https://example.com/page"})
    assert search.success and fetched.success and extracted.success
    assert json.loads(extracted.output)["text"] == "Example\nEvidence."
    assert [request.tool for request in approvals] == ["web_search", "fetch_url"]
    assert [request.permission for request in approvals] == [
        PermissionLevel.EXTERNAL,
        PermissionLevel.EXTERNAL,
    ]


def test_default_denial_prevents_web_request():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("denied requests must not reach the network")

    with WebClient(transport=httpx.MockTransport(handler)) as client:
        registry = create_web_registry(client=client)
        result = registry.execute("web_search", {"query": "private question"})
    assert not result.success
    assert requests == []


def test_extract_requires_a_page_fetched_in_the_same_session():
    client = WebClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    try:
        registry = create_web_registry(client=client)
        result = registry.execute("extract_page", {"url": "https://example.com/missing"})
    finally:
        client.close()
    assert not result.success
    assert "not been fetched" in result.output
