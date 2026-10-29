import base64
import hashlib
import json
import urllib.parse

import httpx
import pytest

from assistant.connectors import (
    AuthExpiredError,
    ConnectionStatus,
    ConnectorError,
    NotFoundError,
    PermissionScopeError,
    RateLimitedError,
)
from assistant.connectors.credentials import InMemoryCredentialStore
from assistant.connectors.google.auth import (
    GMAIL_COMPOSE,
    GMAIL_READONLY,
    SCOPE_LEVELS,
    GoogleClientConfig,
    GoogleSession,
    authorization_url,
    make_pkce,
)


def test_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = make_pkce()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    assert challenge == expected.decode()


def test_authorization_url_requests_offline_pkce_consent():
    url = authorization_url(
        GoogleClientConfig("cid"), SCOPE_LEVELS["read"], "http://127.0.0.1:5000", "state-1", "chal"
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["code_challenge_method"] == ["S256"]
    assert query["access_type"] == ["offline"]
    assert query["scope"][0].split() == list(SCOPE_LEVELS["read"])
    assert GMAIL_COMPOSE not in query["scope"][0]


class FakeReceiver:
    redirect_uri = "http://127.0.0.1:5555"

    def __init__(self, reply):
        self.reply = reply
        self.closed = False

    def wait(self, timeout):
        return self.reply

    def close(self):
        self.closed = True


def _session(handler, store=None, clock=lambda: 1000.0):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleSession(
        GoogleClientConfig("cid", "csecret"),
        store or InMemoryCredentialStore(),
        http=http,
        clock=clock,
    )


def test_connect_exchanges_code_with_verifier_and_stores_only_refresh_token():
    sent = []

    def handler(request):
        sent.append(urllib.parse.parse_qs(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "access_token": "a",
                "refresh_token": "r",
                "expires_in": 3600,
                "scope": f"{GMAIL_READONLY}",
            },
        )

    store = InMemoryCredentialStore()
    session = _session(handler, store)
    state = {}

    def open_browser(url):
        state.update(urllib.parse.parse_qs(urllib.parse.urlparse(url).query))
        receiver.reply = {"code": "c-1", "state": state["state"][0]}

    receiver = FakeReceiver({})
    granted = session.connect(
        SCOPE_LEVELS["read"], open_browser=open_browser, receiver_factory=lambda: receiver
    )
    assert granted == {GMAIL_READONLY}
    assert receiver.closed
    assert sent[0]["code_verifier"][0]
    assert sent[0]["redirect_uri"] == [FakeReceiver.redirect_uri]
    record = json.loads(store.get("google", "default"))
    assert record == {"refresh_token": "r", "scopes": [GMAIL_READONLY]}
    assert session.status() is ConnectionStatus.CONNECTED


def test_connect_rejects_state_mismatch_and_saves_nothing():
    store = InMemoryCredentialStore()
    session = _session(lambda request: httpx.Response(500), store)
    with pytest.raises(ConnectorError, match="unexpected state"):
        session.connect(
            SCOPE_LEVELS["read"],
            open_browser=lambda url: None,
            receiver_factory=lambda: FakeReceiver({"code": "c", "state": "forged"}),
        )
    assert store.get("google", "default") is None


def test_status_reflects_configuration_and_stored_credentials():
    assert _session(lambda r: httpx.Response(500)).status() is ConnectionStatus.NOT_CONNECTED
    unconfigured = GoogleSession(None, InMemoryCredentialStore())
    assert unconfigured.status() is ConnectionStatus.UNAVAILABLE


def test_access_token_is_refreshed_when_expired(google_api):
    now = [1000.0]
    session = google_api.session([GMAIL_READONLY])
    session._clock = lambda: now[0]
    google_api.on("GET", "/ping", {"ok": True})
    session.request("GET", "https://example.googleapis.com/ping")
    session.request("GET", "https://example.googleapis.com/ping")
    assert len(google_api.sent("POST", "/token")) == 1
    now[0] += 4000
    session.request("GET", "https://example.googleapis.com/ping")
    assert len(google_api.sent("POST", "/token")) == 2
    assert google_api.requests[-1].headers["Authorization"] == "Bearer access-1"


def test_401_retries_once_then_reports_expired(google_api):
    session = google_api.session([GMAIL_READONLY])
    google_api.on("GET", "/ping", (401, {"error": {"message": "bad"}}))
    with pytest.raises(AuthExpiredError, match="jamie connect google"):
        session.request("GET", "https://example.googleapis.com/ping")
    assert len(google_api.sent("GET", "/ping")) == 2


def test_revoked_refresh_token_reports_expired(google_api):
    session = google_api.session([GMAIL_READONLY])
    google_api.on("POST", "/token", (400, {"error": "invalid_grant"}))
    with pytest.raises(AuthExpiredError):
        session.access_token()


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (
            403,
            {"error": {"errors": [{"reason": "insufficientPermissions"}], "message": "x"}},
            PermissionScopeError,
        ),
        (429, {"error": {"message": "slow down"}}, RateLimitedError),
        (404, {"error": {"message": "gone"}}, NotFoundError),
    ],
)
def test_provider_errors_are_typed(google_api, status, body, error):
    session = google_api.session([GMAIL_READONLY])
    google_api.on("GET", "/ping", (status, body))
    with pytest.raises(error):
        session.request("GET", "https://example.googleapis.com/ping")


def test_missing_scope_names_the_level_to_request(google_api):
    session = google_api.session([GMAIL_READONLY])
    with pytest.raises(PermissionScopeError, match="--scopes compose") as raised:
        session.require_scope(GMAIL_COMPOSE, "compose")
    assert raised.value.scope == GMAIL_COMPOSE


def test_disconnect_revokes_and_deletes(google_api):
    session = google_api.session([GMAIL_READONLY])
    google_api.on("POST", "/revoke", {})
    session.disconnect()
    assert session.status() is ConnectionStatus.NOT_CONNECTED
    assert google_api.sent("POST", "/revoke")
