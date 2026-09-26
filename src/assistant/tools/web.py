"""Bounded web search, retrieval, and text extraction tools."""

from __future__ import annotations

import html
import ipaddress
import json
import re
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
# ponytail: one fixed cut, about 6k tokens of a 32k context; add paging if pages need more.
MAX_TOOL_PAGE_CHARACTERS = 24_000
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
    if not url.strip():
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


_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_BLOCKS = {
    "address",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "header",
    "nav",
    "p",
    "section",
    "table",
}
_EMPTY_MARKER = re.compile(r"(?:[-*]|\d+\.|#{1,6})")
_LIST_MARKER = re.compile(r"(?:-|\d+\.) ")


def _tidy_markdown(raw: str) -> str:
    """Collapse HTML whitespace; keep list indentation and code blocks as written."""
    lines: list[str] = []
    fenced = False
    for line in raw.splitlines():
        if line.strip() == "```":
            fenced = not fenced
            lines.append("```")
        elif fenced:
            lines.append(line.rstrip())
        else:
            body = " ".join(line.split())
            if _EMPTY_MARKER.fullmatch(body):
                continue  # a list item or heading with no text
            indent = len(line) - len(line.lstrip()) if _LIST_MARKER.match(body) else 0
            lines.append(" " * indent + body)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class _MarkdownParser(HTMLParser):
    """Convert readable HTML to Markdown headings, lists, links, code, and paragraphs."""

    _SKIP = {"script", "style", "noscript", "svg", "template"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self._title: list[str] = []
        self._text: list[str] = []
        self._focused_depth = 0
        self._focused_text: list[str] = []
        self._lists: list[int | None] = []  # the next number for <ol>, None for <ul>
        self._cells = 0
        self._pre = 0
        self._link: list[str] | None = None  # text of the open link
        self._href = ""

    def _emit(self, text: str) -> None:
        if self._link is not None:
            self._link.append(text)  # markup inside a link collapses into its text
            return
        self._text.append(text)
        if self._focused_depth:
            self._focused_text.append(text)

    def _target(self, href: str | None) -> str | None:
        if not href or href.startswith("#"):
            return None
        url = urljoin(self._base_url, href.strip())
        return url if urlsplit(url).scheme in {"http", "https"} else None

    def _close_link(self) -> None:
        text = " ".join("".join(self._link or []).split()).replace("[", "").replace("]", "")
        self._link = None
        if text:
            self._emit(f"[{text}](<{self._href}>)")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif self._skip_depth:
            return
        elif tag == "title":
            self._in_title = True
        elif tag in {"main", "article"}:
            self._focused_depth += 1
            self._emit("\n\n")
        elif tag in _HEADINGS:
            self._emit("\n\n" + "#" * _HEADINGS[tag] + " ")
        elif tag in _BLOCKS:
            self._emit("\n\n")
        elif tag == "br":
            self._emit("\n")
        elif tag in {"ul", "ol"}:
            self._emit("" if self._lists else "\n")  # a nested list continues its item
            self._lists.append(1 if tag == "ol" else None)
        elif tag == "li":
            number = self._lists[-1] if self._lists else None
            if number is not None:
                self._lists[-1] = number + 1
            marker = "- " if number is None else f"{number}. "
            self._emit("\n" + "  " * max(len(self._lists) - 1, 0) + marker)
        elif tag == "tr":
            self._cells = 0
            self._emit("\n")
        elif tag in {"td", "th"}:
            self._emit(" | " if self._cells else "")
            self._cells += 1
        elif tag == "pre":
            self._pre += 1
            self._emit("\n\n```\n")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "a" and self._link is None:
            target = self._target(dict(attrs).get("href"))
            if target is not None:
                self._href = target
                self._link = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(self._skip_depth - 1, 0)
        elif self._skip_depth:
            return
        elif tag == "title":
            self._in_title = False
        elif tag in {"main", "article"} and self._focused_depth:
            self._emit("\n\n")
            self._focused_depth -= 1
        elif tag in _HEADINGS or tag in _BLOCKS:
            self._emit("\n\n")
        elif tag in {"ul", "ol"} and self._lists:
            self._lists.pop()
            self._emit("" if self._lists else "\n\n")
        elif tag == "pre" and self._pre:
            self._pre -= 1
            self._emit("\n```\n\n")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "a" and self._link is not None:
            self._close_link()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
            return
        self._emit(data if self._pre else re.sub(r"\s+", " ", data))

    def result(self, fallback_title: str = "") -> tuple[str, str]:
        if self._link is not None:
            self._close_link()
        title = " ".join("".join(self._title).split()) or fallback_title
        focused = _tidy_markdown("".join(self._focused_text))
        text = focused if len(focused) >= 100 else _tidy_markdown("".join(self._text))
        return title, text[:MAX_EXTRACTED_CHARACTERS]


def extract_page(page: FetchedPage, *, fallback_title: str = "") -> ExtractedPage:
    """Convert one fetched HTML page to Markdown; plain-text pages pass through."""
    if page.content_type in {"text/plain", "application/json"}:
        text = page.text[:MAX_EXTRACTED_CHARACTERS]
        title = fallback_title
    else:
        parser = _MarkdownParser(page.url)
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
        try:
            # Submit like DuckDuckGo's own form; GET requests for longer queries draw bot checks.
            response = self._client.post(SEARCH_URL, data={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebError("Web search request failed.") from exc
        if response.status_code == 202:  # DuckDuckGo answers with a bot-check page
            raise WebError("DuckDuckGo blocked the search with a bot check. Try again later.")
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
    description = "Read one public HTTP or HTTPS page and return its content as Markdown."
    risk = PermissionLevel.EXTERNAL

    class Input(_Input):
        url: str = Field(min_length=1, max_length=MAX_URL_CHARACTERS)

    input_schema = Input

    def __init__(self, client: WebClient) -> None:
        self._client = client

    def execute(self, arguments: Input) -> str:
        return _page_output(extract_page(self._client.fetch(arguments.url)))


class ExtractPageTool:
    name = "extract_page"
    description = "Return the Markdown of a page already fetched in the current session."
    risk = PermissionLevel.READ

    class Input(_Input):
        url: str = Field(min_length=1, max_length=MAX_URL_CHARACTERS)

    input_schema = Input

    def __init__(self, client: WebClient) -> None:
        self._client = client

    def execute(self, arguments: Input) -> str:
        return _page_output(self._client.extract(arguments.url))


def _page_output(page: ExtractedPage) -> str:
    return json.dumps(
        {
            "url": page.url,
            "title": page.title,
            "markdown": page.text[:MAX_TOOL_PAGE_CHARACTERS],
            "truncated": len(page.text) > MAX_TOOL_PAGE_CHARACTERS,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
