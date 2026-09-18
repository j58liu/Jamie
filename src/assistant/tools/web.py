"""Bounded web search, retrieval, and text extraction tools."""

from __future__ import annotations

import html
import ipaddress
import json
import socket
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Jamie/0.1 local research assistant"
MAX_QUERY_CHARACTERS = 500
MAX_URL_CHARACTERS = 2_000
MAX_PAGE_BYTES = 2_000_000
MAX_EXTRACTED_CHARACTERS = 30_000
MAX_SEARCH_RESULTS = 10
MAX_REDIRECTS = 5


class WebError(ToolError):
    """A web operation could not safely complete."""


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    snippet: str = ""


class FetchedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_url: str
    url: str
    status_code: int
    content_type: str
    text: str
    byte_count: int
    truncated: bool


class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    title: str
    text: str


class PageStore:
    """Keep fetched pages in memory for a single tool or research session."""

    def __init__(self) -> None:
        self._pages: dict[str, FetchedPage] = {}

    def put(self, page: FetchedPage) -> None:
        self._pages[page.requested_url] = page
        self._pages[page.url] = page

    def get(self, url: str) -> FetchedPage:
        try:
            return self._pages[_normalize_url(url)]
        except KeyError as exc:
            raise WebError("The URL has not been fetched in this session.") from exc


Resolver = Callable[..., Iterable[tuple[object, object, object, object, tuple[object, ...]]]]


def _normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise WebError("URL must not be blank.")
    if len(url) > MAX_URL_CHARACTERS:
        raise WebError("URL is too long.")
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise WebError("URL is invalid.") from exc
    if parts.scheme not in {"http", "https"}:
        raise WebError("Only HTTP and HTTPS URLs are supported.")
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise WebError("URL must contain a host and no user information.")
    host = parts.hostname.lower().rstrip(".")
    default_port = (
        port is None
        or (parts.scheme == "http" and port == 80)
        or (parts.scheme == "https" and port == 443)
    )
    netloc = host if default_port else f"{host}:{port}"
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]" if default_port else f"[{host}]:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


def _is_public_address(address: str) -> bool:
    try:
        value = ipaddress.ip_address(address)
    except ValueError:
        return False
    return value.is_global


def validate_public_url(url: str, resolver: Resolver = socket.getaddrinfo) -> str:
    """Normalize a URL and reject hosts that resolve outside the public internet."""
    normalized = _normalize_url(url)
    parts = urlsplit(normalized)
    host = parts.hostname
    assert host is not None
    if host == "localhost" or host.endswith((".localhost", ".local")):
        raise WebError("URL host must use a public address.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            records = list(resolver(host, parts.port or (443 if parts.scheme == "https" else 80)))
        except OSError as exc:
            raise WebError(f"Cannot resolve URL host: {host}") from exc
        addresses = {str(record[4][0]) for record in records if len(record) >= 5 and record[4]}
        if not addresses or any(not _is_public_address(address) for address in addresses):
            raise WebError("URL host does not resolve only to public addresses.") from None
    else:
        if not literal.is_global:
            raise WebError("URL host must use a public address.")
    return normalized


def _search_target(href: str) -> str | None:
    candidate = html.unescape(href).strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    parts = urlsplit(candidate)
    if parts.hostname and parts.hostname.endswith("duckduckgo.com"):
        redirected = parse_qs(parts.query).get("uddg")
        if redirected:
            candidate = unquote(redirected[0])
    try:
        return _normalize_url(candidate)
    except WebError:
        return None


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._depth = 0
        self._text: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self._capture is not None:
            self._depth += 1
            return
        if tag == "a" and "result__a" in classes:
            self._capture = "title"
            self._href = attributes.get("href") or ""
        elif classes.intersection({"result__snippet", "result-snippet"}):
            self._capture = "snippet"
        else:
            return
        self._depth = 1
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        self._depth -= 1
        if self._depth:
            return
        value = " ".join("".join(self._text).split())
        if self._capture == "title":
            target = _search_target(self._href)
            if target and value:
                self.results.append({"title": value, "url": target, "snippet": ""})
        elif self.results and value:
            self.results[-1]["snippet"] = value
        self._capture = None
        self._text = []


class _PageParser(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template"}
    _BREAKS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title: list[str] = []
        self._text: list[str] = []
        self._focused_depth = 0
        self._focused_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif not self._skip_depth and tag in {"main", "article"}:
            self._focused_depth += 1
            self._text.append("\n")
            self._focused_text.append("\n")
        elif not self._skip_depth and tag == "title":
            self._in_title = True
        elif not self._skip_depth and tag in self._BREAKS:
            self._text.append("\n")
            if self._focused_depth:
                self._focused_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in {"main", "article"} and self._focused_depth:
            self._focused_depth -= 1
            self._text.append("\n")
            self._focused_text.append("\n")
        elif not self._skip_depth and tag == "title":
            self._in_title = False
        elif not self._skip_depth and tag in self._BREAKS:
            self._text.append("\n")
            if self._focused_depth:
                self._focused_text.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
        self._text.append(data)
        if self._focused_depth:
            self._focused_text.append(data)

    def result(self, fallback_title: str = "") -> tuple[str, str]:
        title = " ".join("".join(self._title).split()) or fallback_title
        focused = "".join(self._focused_text)
        raw_text = focused if len(focused.strip()) >= 100 else "".join(self._text)
        lines = [" ".join(line.split()) for line in raw_text.splitlines()]
        text = "\n".join(line for line in lines if line)
        return title, text[:MAX_EXTRACTED_CHARACTERS]


def extract_page(page: FetchedPage, *, fallback_title: str = "") -> ExtractedPage:
    """Extract readable text from one fetched HTML or plain-text page."""
    if page.content_type in {"text/plain", "application/json"}:
        text = page.text[:MAX_EXTRACTED_CHARACTERS]
        title = fallback_title
    else:
        parser = _PageParser()
        try:
            parser.feed(page.text)
            parser.close()
        except Exception as exc:
            raise WebError("Page HTML could not be parsed.") from exc
        title, text = parser.result(fallback_title)
    if not text.strip():
        raise WebError("Page contains no extractable text.")
    return ExtractedPage(url=page.url, title=title, text=text)


class WebClient:
    """Search and fetch public web pages with strict limits and no browser."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = socket.getaddrinfo,
        store: PageStore | None = None,
    ) -> None:
        self._resolver = resolver
        self.store = store if store is not None else PageStore()
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(20.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> WebClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise WebError("Search query must not be blank.")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise WebError("Search query is too long.")
        if isinstance(limit, bool) or not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise WebError(f"Search limit must be between 1 and {MAX_SEARCH_RESULTS}.")
        try:
            response = self._client.get(SEARCH_URL, params={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebError("Web search request failed.") from exc
        parser = _SearchParser()
        parser.feed(response.text)
        results = [SearchResult.model_validate(item) for item in parser.results[:limit]]
        if not results:
            raise WebError("Web search returned no usable results.")
        return results

    def fetch(self, url: str) -> FetchedPage:
        requested = validate_public_url(url, self._resolver)
        target = requested
        for redirect_count in range(MAX_REDIRECTS + 1):
            try:
                with self._client.stream("GET", target) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise WebError("Redirect response has no destination.")
                        if redirect_count == MAX_REDIRECTS:
                            raise WebError("Page exceeded the redirect limit.")
                        target = validate_public_url(urljoin(target, location), self._resolver)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in {
                        "text/html",
                        "application/xhtml+xml",
                        "text/plain",
                        "application/json",
                    }:
                        raise WebError(
                            f"Unsupported page content type: {content_type or 'unknown'}"
                        )
                    body = bytearray()
                    truncated = False
                    for chunk in response.iter_bytes():
                        remaining = MAX_PAGE_BYTES - len(body)
                        if len(chunk) > remaining:
                            body.extend(chunk[:remaining])
                            truncated = True
                            break
                        body.extend(chunk)
                    encoding = response.encoding or "utf-8"
            except WebError:
                raise
            except httpx.HTTPError as exc:
                raise WebError("Page request failed.") from exc
            text = bytes(body).decode(encoding, errors="replace")
            page = FetchedPage(
                requested_url=requested,
                url=target,
                status_code=response.status_code,
                content_type=content_type,
                text=text,
                byte_count=len(body),
                truncated=truncated,
            )
            self.store.put(page)
            return page
        raise WebError("Page exceeded the redirect limit.")

    def extract(self, url: str, *, fallback_title: str = "") -> ExtractedPage:
        return extract_page(self.store.get(url), fallback_title=fallback_title)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebSearchTool:
    name = "web_search"
    description = "Search the public web and return titles, URLs, and snippets."
    risk = PermissionLevel.EXTERNAL

    class Input(_Input):
        query: str = Field(min_length=1, max_length=MAX_QUERY_CHARACTERS)
        limit: int = Field(default=5, ge=1, le=MAX_SEARCH_RESULTS, strict=True)

        @field_validator("query")
        @classmethod
        def validate_query(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("Query must not be blank.")
            return value

    input_schema = Input

    def __init__(self, client: WebClient) -> None:
        self._client = client

    def execute(self, arguments: Input) -> str:
        results = self._client.search(arguments.query, limit=arguments.limit)
        return json.dumps(
            {"results": [result.model_dump(mode="json") for result in results]},
            ensure_ascii=False,
            separators=(",", ":"),
        )


class FetchUrlTool:
    name = "fetch_url"
    description = "Fetch one public HTTP or HTTPS page into the current session."
    risk = PermissionLevel.EXTERNAL

    class Input(_Input):
        url: str = Field(min_length=1, max_length=MAX_URL_CHARACTERS)

    input_schema = Input

    def __init__(self, client: WebClient) -> None:
        self._client = client

    def execute(self, arguments: Input) -> str:
        page = self._client.fetch(arguments.url)
        return json.dumps(
            {
                "url": page.url,
                "status_code": page.status_code,
                "content_type": page.content_type,
                "bytes": page.byte_count,
                "truncated": page.truncated,
            },
            separators=(",", ":"),
        )


class ExtractPageTool:
    name = "extract_page"
    description = "Extract readable text from a page fetched in the current session."
    risk = PermissionLevel.READ

    class Input(_Input):
        url: str = Field(min_length=1, max_length=MAX_URL_CHARACTERS)

    input_schema = Input

    def __init__(self, client: WebClient) -> None:
        self._client = client

    def execute(self, arguments: Input) -> str:
        page = self._client.extract(arguments.url)
        return json.dumps(page.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
