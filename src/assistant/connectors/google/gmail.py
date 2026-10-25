"""Gmail REST connector returning concise, typed message data."""

from __future__ import annotations

import base64
import html
import re
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import formataddr, getaddresses, parseaddr

from pydantic import BaseModel

from assistant.connectors.base import Capability, ConnectionStatus, ConnectorInfo
from assistant.connectors.errors import ProviderError
from assistant.connectors.google.auth import GMAIL_COMPOSE, GMAIL_READONLY, GoogleSession

API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_BODY_CHARACTERS = 4000
MAX_SEARCH_RESULTS = 10
SUMMARY_HEADERS = ("From", "Subject", "Date")

OPERATIONS = (
    "gmail_search_messages",
    "gmail_get_message",
    "gmail_get_thread",
    "gmail_create_draft",
    "gmail_get_draft",
    "gmail_send_draft",
)


class MessageSummary(BaseModel):
    id: str
    thread_id: str
    sender: str
    subject: str
    snippet: str
    received_at: str


class MessageDetail(MessageSummary):
    to: list[str] = []
    cc: list[str] | None = None
    body: str
    truncated: bool | None = None


class ThreadSummary(BaseModel):
    id: str
    message_count: int
    messages: list[MessageSummary]
    latest: MessageDetail


class Draft(BaseModel):
    id: str
    message_id: str | None = None
    thread_id: str | None = None
    to: list[str]
    subject: str
    body: str
    in_reply_to: str | None = None


class SentMessage(BaseModel):
    id: str
    thread_id: str | None = None
    sent: bool = True


def _headers(payload: dict) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in payload.get("headers", []) if "name" in h}


def _received_at(message: dict) -> str:
    try:
        stamp = datetime.fromtimestamp(int(message["internalDate"]) / 1000, UTC)
    except (KeyError, ValueError):
        return ""
    return stamp.isoformat().replace("+00:00", "Z")


def _address(value: str) -> str:
    name, address = parseaddr(value)
    return formataddr((name, address)) if address else value.strip()


def _addresses(value: str | None) -> list[str]:
    if not value:
        return []
    return [formataddr(pair) if pair[0] else pair[1] for pair in getaddresses([value]) if pair[1]]


def _decode(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def strip_html(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", markup)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def trim_quoted(text: str) -> str:
    """Drop quoted history: '>' lines and everything after an 'On ... wrote:' line."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^On .+wrote:$", stripped) or stripped.startswith("-----Original Message"):
            break
        if stripped.startswith(">"):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_body(payload: dict) -> str:
    plain: list[str] = []
    markup: list[str] = []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(_decode(data))
        elif data and mime == "text/html":
            markup.append(_decode(data))
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    if plain:
        return "\n".join(plain)
    return strip_html("\n".join(markup))


def _summary(message: dict) -> MessageSummary:
    headers = _headers(message.get("payload", {}))
    return MessageSummary(
        id=message["id"],
        thread_id=message.get("threadId", ""),
        sender=_address(headers.get("from", "")),
        subject=headers.get("subject", "(no subject)"),
        snippet=html.unescape(message.get("snippet", "")),
        received_at=_received_at(message),
    )


def _detail(message: dict) -> MessageDetail:
    payload = message.get("payload", {})
    headers = _headers(payload)
    body = trim_quoted(extract_body(payload))
    truncated = len(body) > MAX_BODY_CHARACTERS
    return MessageDetail(
        **_summary(message).model_dump(),
        to=_addresses(headers.get("to")),
        cc=_addresses(headers.get("cc")) or None,
        body=body[:MAX_BODY_CHARACTERS],
        truncated=True if truncated else None,
    )


def _draft(draft: dict) -> Draft:
    message = draft.get("message", {})
    headers = _headers(message.get("payload", {}))
    return Draft(
        id=draft["id"],
        message_id=message.get("id"),
        thread_id=message.get("threadId"),
        to=_addresses(headers.get("to")),
        subject=headers.get("subject", ""),
        body=extract_body(message.get("payload", {})).strip()[:MAX_BODY_CHARACTERS],
        in_reply_to=headers.get("in-reply-to"),
    )


class GmailConnector:
    """Search, read, draft, and send Gmail messages through one Google session."""

    name = "gmail"

    def __init__(self, session: GoogleSession) -> None:
        self._session = session

    def info(self) -> ConnectorInfo:
        status = self._session.status()
        capabilities: set[Capability] = set()
        if status is ConnectionStatus.CONNECTED:
            if self._session.has_scope(GMAIL_READONLY):
                capabilities.add(Capability.READ)
            if self._session.has_scope(GMAIL_COMPOSE):
                capabilities |= {Capability.DRAFT, Capability.SEND}
            if not capabilities:
                status = ConnectionStatus.INSUFFICIENT_SCOPE
        return ConnectorInfo(self.name, status, frozenset(capabilities), OPERATIONS)

    def search_messages(self, query: str, max_results: int = 5) -> list[MessageSummary]:
        self._session.require_scope(GMAIL_READONLY, "read")
        listing = self._session.request(
            "GET",
            f"{API}/messages",
            params={"q": query, "maxResults": min(max_results, MAX_SEARCH_RESULTS)},
        )
        summaries = []
        for item in listing.get("messages", []):
            message = self._session.request(
                "GET",
                f"{API}/messages/{item['id']}",
                params=[("format", "metadata")]
                + [("metadataHeaders", name) for name in SUMMARY_HEADERS],
            )
            summaries.append(_summary(message))
        return summaries

    def get_message(self, message_id: str) -> MessageDetail:
        self._session.require_scope(GMAIL_READONLY, "read")
        message = self._session.request(
            "GET", f"{API}/messages/{message_id}", params={"format": "full"}
        )
        return _detail(message)

    def get_thread(self, thread_id: str) -> ThreadSummary:
        self._session.require_scope(GMAIL_READONLY, "read")
        thread = self._session.request(
            "GET", f"{API}/threads/{thread_id}", params={"format": "full"}
        )
        messages = thread.get("messages", [])
        if not messages:
            raise ProviderError("The thread has no messages.")
        return ThreadSummary(
            id=thread.get("id", thread_id),
            message_count=len(messages),
            messages=[_summary(message) for message in messages],
            latest=_detail(messages[-1]),
        )

    def create_draft(
        self,
        to: list[str],
        subject: str,
        body: str,
        *,
        cc: list[str] | None = None,
        reply_to_message_id: str | None = None,
    ) -> Draft:
        self._session.require_scope(GMAIL_COMPOSE, "compose")
        email = EmailMessage()
        email["To"] = ", ".join(to)
        if cc:
            email["Cc"] = ", ".join(cc)
        thread_id = None
        in_reply_to = None
        if reply_to_message_id:
            original = self._session.request(
                "GET",
                f"{API}/messages/{reply_to_message_id}",
                params=[("format", "metadata")]
                + [("metadataHeaders", n) for n in ("Message-ID", "References", "Subject")],
            )
            headers = _headers(original.get("payload", {}))
            thread_id = original.get("threadId")
            in_reply_to = headers.get("message-id")
            if in_reply_to:
                email["In-Reply-To"] = in_reply_to
                references = headers.get("references", "")
                email["References"] = f"{references} {in_reply_to}".strip()
            if not subject.strip():
                subject = headers.get("subject", "")
            if subject and not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
        email["Subject"] = subject
        email.set_content(body)
        raw = base64.urlsafe_b64encode(email.as_bytes()).decode("ascii")
        message: dict[str, str] = {"raw": raw}
        if thread_id:
            message["threadId"] = thread_id
        created = self._session.request("POST", f"{API}/drafts", json={"message": message})
        return Draft(
            id=created["id"],
            message_id=created.get("message", {}).get("id"),
            thread_id=created.get("message", {}).get("threadId", thread_id),
            to=list(to),
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
        )

    def get_draft(self, draft_id: str) -> Draft:
        self._session.require_scope(GMAIL_COMPOSE, "compose")
        return _draft(
            self._session.request("GET", f"{API}/drafts/{draft_id}", params={"format": "full"})
        )

    def send_draft(self, draft_id: str) -> SentMessage:
        self._session.require_scope(GMAIL_COMPOSE, "compose")
        sent = self._session.request("POST", f"{API}/drafts/send", json={"id": draft_id})
        return SentMessage(id=sent.get("id", ""), thread_id=sent.get("threadId"))
