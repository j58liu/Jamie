import json
import socket

import httpx
import pytest

from assistant.security import CallbackApprovalProvider, PermissionLevel
from assistant.tools import (
    FetchedPage,
    WebClient,
    WebError,
    create_web_registry,
    extract_page,
    validate_public_url,
)
from assistant.tools.web import MAX_TOOL_PAGE_CHARACTERS


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
        assert request.method == "POST"
        assert request.url.host == "html.duckduckgo.com"
        assert request.content == b"q=test+query"
        return httpx.Response(200, text=search_html, headers={"content-type": "text/html"})

    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        results = client.search("test query", limit=2)
    assert [result.url for result in results] == [
        "https://one.example/page",
        "https://two.example/article",
    ]
    assert results[0].title == "First result"
    assert results[0].snippet == "Useful summary."


def test_search_reports_a_bot_check_instead_of_empty_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, text='<div class="anomaly-modal"></div>')

    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        with pytest.raises(WebError, match="bot check"):
            client.search("natural gas")


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
    page = {"url": "https://example.com/page", "title": "Example", "markdown": "Evidence."}
    assert json.loads(fetched.output) == {**page, "truncated": False}
    assert json.loads(extracted.output) == {**page, "truncated": False}
    assert [request.tool for request in approvals] == ["web_search", "fetch_url"]
    assert [request.permission for request in approvals] == [
        PermissionLevel.EXTERNAL,
        PermissionLevel.EXTERNAL,
    ]


def _page(html: str, url: str = "https://docs.example/guide/install") -> FetchedPage:
    return FetchedPage(
        requested_url=url,
        url=url,
        status_code=200,
        content_type="text/html",
        text=html,
        byte_count=len(html),
        truncated=False,
    )


def test_pages_are_read_as_markdown():
    html = """<html><head><title>Install guide</title><script>track()</script></head>
    <body><nav><a href="/">Home</a></nav><main>
      <h1>Install   the tool</h1>
      <p>Run it. See <a href="../faq">the <b>FAQ</b></a>, <a href="#top">top</a>,
         or <a href="javascript:void(0)">nothing</a>.</p>
      <pre><code>pip install example-tool
  example --help</code></pre>
      <ul><li>Fast <code>setup</code></li>
          <li>Offline<ol><li>first</li><li>second</li></ol></li><li> </li></ul>
      <table><tr><th>Name</th><th>Value</th></tr><tr><td>a</td><td>1</td></tr></table>
      <h2></h2><p>2024 was a good year.</p>
    </main></body></html>"""

    page = extract_page(_page(html))

    assert page.title == "Install guide"
    assert page.text == (
        "# Install the tool\n\n"
        "Run it. See [the FAQ](<https://docs.example/faq>), top, or nothing.\n\n"
        "```\npip install example-tool\n  example --help\n```\n\n"
        "- Fast `setup`\n- Offline\n  1. first\n  2. second\n\n"
        "Name | Value\na | 1\n\n"
        "2024 was a good year."
    )


def test_page_tools_cut_long_pages_to_fit_the_model_context():
    long_page = f"<main><p>{'word ' * 3000}</p></main>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=long_page, headers={"content-type": "text/html"})

    approvals = CallbackApprovalProvider(lambda request: True)
    with WebClient(transport=httpx.MockTransport(handler), resolver=public_resolver) as client:
        registry = create_web_registry(client=client, approvals=approvals)
        output = json.loads(registry.execute("fetch_url", {"url": "https://example.com"}).output)

    assert output["truncated"] is True
    assert len(output["markdown"]) == MAX_TOOL_PAGE_CHARACTERS


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
