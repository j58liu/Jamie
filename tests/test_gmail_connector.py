import base64
import email
import json

import pytest

from assistant.connectors import Connectors, PermissionScopeError
from assistant.connectors.google.auth import GMAIL_COMPOSE, GMAIL_READONLY
from assistant.connectors.google.gmail import (
    MAX_BODY_CHARACTERS,
    GmailConnector,
    strip_html,
    trim_quoted,
)
from assistant.security import ApprovalRequest, CallbackApprovalProvider, PermissionLevel
from assistant.tools import ToolRegistry
from assistant.tools.integrations import register_personal_tools
from assistant.ui.approvals import SESSION_APPROVABLE, TerminalApprovalProvider

API = "/gmail/v1/users/me"


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def metadata(message_id="m1", subject="Meeting next week"):
    return {
        "id": message_id,
        "threadId": "t1",
        "snippet": "Are you available Tuesday?",
        "internalDate": "1792765800000",
        "payload": {
            "headers": [
                {"name": "From", "value": "Alex Kim <alex@example.com>"},
                {"name": "Subject", "value": subject},
                {"name": "Message-ID", "value": "<abc@mail.example.com>"},
            ]
        },
    }


def full(body_parts):
    message = metadata()
    message["payload"].update({"mimeType": "multipart/alternative", "parts": body_parts})
    message["payload"]["headers"].append({"name": "To", "value": "me@example.com"})
    return message


@pytest.fixture
def gmail(google_api):
    return google_api, GmailConnector(google_api.session([GMAIL_READONLY, GMAIL_COMPOSE]))


def test_search_returns_concise_summaries(gmail):
    api, connector = gmail
    api.on("GET", f"{API}/messages", {"messages": [{"id": "m1"}]})
    api.on("GET", f"{API}/messages/m1", metadata())
    results = connector.search_messages("from:alex meeting", 50)
    assert [r.model_dump() for r in results] == [
        {
            "id": "m1",
            "thread_id": "t1",
            "sender": "Alex Kim <alex@example.com>",
            "subject": "Meeting next week",
            "snippet": "Are you available Tuesday?",
            "received_at": "2026-10-23T14:30:00Z",
        }
    ]
    listing = api.sent("GET", f"{API}/messages")[0]
    assert listing.url.params["maxResults"] == "10"
    detail = api.sent("GET", f"{API}/messages/m1")[0]
    assert detail.url.params["format"] == "metadata"


def test_message_body_drops_quoted_history(gmail):
    api, connector = gmail
    text = "Tuesday 2pm for 30 minutes?\n\nOn Mon, Oct 19, 2026 Me wrote:\n> old text"
    api.on(
        "GET", f"{API}/messages/m1", full([{"mimeType": "text/plain", "body": {"data": b64(text)}}])
    )
    message = connector.get_message("m1")
    assert message.body == "Tuesday 2pm for 30 minutes?"
    assert message.to == ["me@example.com"]
    assert message.truncated is None


def test_html_only_body_is_stripped_and_long_bodies_truncated(gmail):
    api, connector = gmail
    markup = "<html><style>p{}</style><p>Hello&nbsp;there</p>" + "x" * 5000 + "</html>"
    api.on(
        "GET",
        f"{API}/messages/m1",
        full([{"mimeType": "text/html", "body": {"data": b64(markup)}}]),
    )
    message = connector.get_message("m1")
    assert message.body.startswith("Hello there")
    assert len(message.body) == MAX_BODY_CHARACTERS
    assert message.truncated is True


def test_strip_and_trim_helpers():
    assert strip_html("<b>a</b><br>b") == "a\nb"
    assert trim_quoted("hi\n> quoted\nbye") == "hi\nbye"


def test_thread_includes_body_of_latest_message_only(gmail):
    api, connector = gmail
    older = metadata("m0", "Meeting next week")
    latest = full([{"mimeType": "text/plain", "body": {"data": b64("Latest body")}}])
    api.on("GET", f"{API}/threads/t1", {"id": "t1", "messages": [older, latest]})
    thread = connector.get_thread("t1")
    assert thread.message_count == 2
    assert [m.id for m in thread.messages] == ["m0", "m1"]
    assert thread.latest.body == "Latest body"
    assert "body" not in thread.messages[0].model_dump()


def test_reply_draft_joins_the_thread(gmail):
    api, connector = gmail
    api.on("GET", f"{API}/messages/m1", metadata())
    api.on("POST", f"{API}/drafts", {"id": "r-1", "message": {"id": "dm", "threadId": "t1"}})
    draft = connector.create_draft(
        ["alex@example.com"], "", "Tuesday works.", reply_to_message_id="m1"
    )
    assert draft.id == "r-1" and draft.subject == "Re: Meeting next week"
    sent = json.loads(api.sent("POST", f"{API}/drafts")[0].content)
    assert sent["message"]["threadId"] == "t1"
    raw = base64.urlsafe_b64decode(sent["message"]["raw"])
    parsed = email.message_from_bytes(raw)
    assert parsed["In-Reply-To"] == "<abc@mail.example.com>"
    assert parsed["References"] == "<abc@mail.example.com>"
    assert parsed["To"] == "alex@example.com"
    assert parsed.get_payload().strip() == "Tuesday works."


def test_drafting_without_compose_scope_sends_nothing(google_api):
    connector = GmailConnector(google_api.session([GMAIL_READONLY]))
    with pytest.raises(PermissionScopeError, match="compose"):
        connector.create_draft(["alex@example.com"], "Hi", "Body")
    assert not google_api.requests


def _registry(connector, approvals):
    registry = ToolRegistry(approvals=approvals)
    import datetime

    return register_personal_tools(registry, Connectors(tz=datetime.UTC, gmail=connector))


def test_draft_requires_approval_and_denial_sends_nothing(gmail):
    api, connector = gmail
    requests: list[ApprovalRequest] = []
    registry = _registry(connector, CallbackApprovalProvider(lambda r: requests.append(r) or False))
    result = registry.execute("gmail_create_draft", {"to": ["alex@example.com"], "body": "Hi"})
    assert result.denied
    assert requests[0].permission is PermissionLevel.EXTERNAL
    assert requests[0].arguments["body"] == "Hi"
    assert not api.sent("POST", f"{API}/drafts")


def test_every_send_is_confirmed_and_session_approval_is_not_offered(gmail):
    from io import StringIO

    from rich.console import Console

    api, connector = gmail
    api.on("POST", f"{API}/drafts/send", {"id": "sent-1", "threadId": "t1"})
    answers = iter(["a", "y", "y"])
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return next(answers)

    console = Console(file=StringIO(), width=100)
    registry = _registry(connector, TerminalApprovalProvider(console, ask=ask))
    assert PermissionLevel.SEND not in SESSION_APPROVABLE
    first = registry.execute("gmail_send_draft", {"draft_id": "r-1"})
    second = registry.execute("gmail_send_draft", {"draft_id": "r-1"})
    assert first.success and second.success
    assert len(asked) == 3  # "a" was rejected, then each send asked again
    assert "[a] allow session" not in console.file.getvalue()
    assert json.loads(first.output) == {"id": "sent-1", "thread_id": "t1", "sent": True}


def test_ids_are_validated_before_any_request(gmail):
    api, connector = gmail
    registry = _registry(connector, CallbackApprovalProvider(lambda r: True))
    result = registry.execute("gmail_get_message", {"message_id": "../drafts/send"})
    assert not result.success and "Invalid arguments" in result.output
    assert not api.requests
