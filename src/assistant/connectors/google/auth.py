"""Google desktop OAuth (loopback + PKCE) and an authorized request session.

Only the refresh token and the granted scopes are stored, in the credential store
(Keychain on macOS). Access tokens live in memory and are refreshed when they expire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

import httpx

from assistant.connectors.base import ConnectionStatus
from assistant.connectors.credentials import CredentialStore
from assistant.connectors.errors import (
    AlreadyExistsError,
    AuthExpiredError,
    ConnectorError,
    NotConnectedError,
    NotFoundError,
    PermissionScopeError,
    ProviderError,
    RateLimitedError,
)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
CALENDAR_EVENTS_READONLY = "https://www.googleapis.com/auth/calendar.events.readonly"
CALENDAR_FREEBUSY = "https://www.googleapis.com/auth/calendar.freebusy"
CALENDAR_EVENTS = "https://www.googleapis.com/auth/calendar.events"

# Each named level adds to the one before it; request more only when it is needed.
SCOPE_LEVELS: dict[str, tuple[str, ...]] = {
    "read": (GMAIL_READONLY, CALENDAR_EVENTS_READONLY, CALENDAR_FREEBUSY),
    "compose": (GMAIL_READONLY, GMAIL_COMPOSE, CALENDAR_EVENTS_READONLY, CALENDAR_FREEBUSY),
    "calendar-write": (GMAIL_READONLY, GMAIL_COMPOSE, CALENDAR_EVENTS, CALENDAR_FREEBUSY),
}

# A broader scope that also satisfies a narrower one.
IMPLIED_BY = {
    CALENDAR_EVENTS_READONLY: {CALENDAR_EVENTS},
    CALENDAR_FREEBUSY: {CALENDAR_EVENTS, CALENDAR_EVENTS_READONLY},
}

CREDENTIAL_SERVICE = "google"
CREDENTIAL_ACCOUNT = "default"
REFRESH_MARGIN_SECONDS = 60
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class GoogleClientConfig:
    """A Google Cloud "Desktop app" OAuth client."""

    client_id: str
    client_secret: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> GoogleClientConfig:
        try:
            data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NotConnectedError(f"Cannot read Google client file {path}: {exc}") from exc
        section = data.get("installed") or data
        if not isinstance(section, dict) or not section.get("client_id"):
            raise NotConnectedError("Google client file has no installed.client_id.")
        return cls(section["client_id"], section.get("client_secret"))


def make_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def authorization_url(
    client: GoogleClientConfig,
    scopes: tuple[str, ...],
    redirect_uri: str,
    state: str,
    challenge: str,
) -> str:
    query = {
        "client_id": client.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(query)}"


class Receiver(Protocol):
    redirect_uri: str

    def wait(self, timeout: float) -> dict[str, str]: ...

    def close(self) -> None: ...


class LoopbackReceiver:
    """Accept exactly one OAuth redirect on 127.0.0.1 at a random port."""

    def __init__(self) -> None:
        result: dict[str, str] = {}
        done = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                query = urllib.parse.urlparse(self.path).query
                result.update({k: v[0] for k, v in urllib.parse.parse_qs(query).items()})
                body = b"Jamie is connected. You can close this tab."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                done.set()

            def log_message(self, *args) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._result, self._done = result, done
        self.redirect_uri = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._done.is_set():
            self._server.handle_request()

    def wait(self, timeout: float) -> dict[str, str]:
        if not self._done.wait(timeout):
            raise ConnectorError("Timed out waiting for Google sign-in.")
        return dict(self._result)

    def close(self) -> None:
        self._done.set()
        self._server.server_close()


def _token_request(http: httpx.Client, data: dict[str, str]) -> dict:
    try:
        response = http.post(TOKEN_URL, data=data, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise ProviderError(f"Cannot reach Google: {exc}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        error = payload.get("error", "")
        if error in {"invalid_grant", "unauthorized_client"}:
            raise AuthExpiredError(
                "Google sign-in expired or was revoked. Run: jamie connect google"
            )
        raise ProviderError(f"Google token request failed ({response.status_code}) {error}")
    return payload


class GoogleSession:
    """Hold Google credentials and send authorized, error-mapped API requests."""

    def __init__(
        self,
        client: GoogleClientConfig | None,
        store: CredentialStore,
        *,
        http: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._store = store
        self._http = http if http is not None else httpx.Client()
        self._clock = clock
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._record: dict | None = None
        self._loaded = False

    # -- stored credentials -------------------------------------------------------

    def _load(self) -> dict | None:
        if not self._loaded:
            raw = self._store.get(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
            try:
                self._record = json.loads(raw) if raw else None
            except ValueError:
                self._record = None
            self._loaded = True
        return self._record

    @property
    def granted_scopes(self) -> frozenset[str]:
        record = self._load()
        return frozenset(record.get("scopes", ())) if record else frozenset()

    def has_scope(self, scope: str) -> bool:
        granted = self.granted_scopes
        return scope in granted or bool(IMPLIED_BY.get(scope, set()) & granted)

    def require_scope(self, scope: str, level: str) -> None:
        if not self.has_scope(scope):
            name = scope.rsplit("/", 1)[-1]
            raise PermissionScopeError(
                f"Google access {name!r} was not granted. "
                f"Ask the user to run: jamie connect google --scopes {level}",
                scope=scope,
            )

    def status(self) -> ConnectionStatus:
        if self._client is None:
            return ConnectionStatus.UNAVAILABLE
        return ConnectionStatus.CONNECTED if self._load() else ConnectionStatus.NOT_CONNECTED

    def save(self, token: dict, requested: tuple[str, ...]) -> None:
        refresh = token.get("refresh_token") or (self._load() or {}).get("refresh_token")
        if not refresh:
            raise AuthExpiredError("Google did not return a refresh token; try connecting again.")
        scopes = token.get("scope", "").split() or list(requested)
        record = {"refresh_token": refresh, "scopes": sorted(set(scopes))}
        self._store.set(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT, json.dumps(record))
        self._record, self._loaded = record, True
        if token.get("access_token"):
            self._access_token = token["access_token"]
            self._expires_at = self._clock() + float(token.get("expires_in", 0))

    def disconnect(self) -> None:
        record = self._load()
        if record:
            try:
                self._http.post(
                    REVOKE_URL, data={"token": record["refresh_token"]}, timeout=TIMEOUT_SECONDS
                )
            except httpx.HTTPError:
                pass  # the local credential is removed regardless
        self._store.delete(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        self._record, self._access_token, self._expires_at = None, None, 0.0

    # -- OAuth flow ---------------------------------------------------------------

    def connect(
        self,
        scopes: tuple[str, ...],
        *,
        open_browser: Callable[[str], object],
        receiver_factory: Callable[[], Receiver] = LoopbackReceiver,
        timeout: float = 300,
    ) -> frozenset[str]:
        if self._client is None:
            raise NotConnectedError("Set [google] client_secrets_path in assistant.toml first.")
        verifier, challenge = make_pkce()
        state = secrets.token_urlsafe(24)
        receiver = receiver_factory()
        try:
            open_browser(
                authorization_url(self._client, scopes, receiver.redirect_uri, state, challenge)
            )
            reply = receiver.wait(timeout)
        finally:
            receiver.close()
        if reply.get("state") != state:
            raise ConnectorError("Google sign-in returned an unexpected state; nothing saved.")
        if "error" in reply or "code" not in reply:
            raise ConnectorError(f"Google sign-in failed: {reply.get('error', 'no code')}")
        data = {
            "client_id": self._client.client_id,
            "code": reply["code"],
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": receiver.redirect_uri,
        }
        if self._client.client_secret:
            data["client_secret"] = self._client.client_secret
        self.save(_token_request(self._http, data), scopes)
        return self.granted_scopes

    # -- requests -----------------------------------------------------------------

    def _refresh(self) -> str:
        record = self._load()
        if self._client is None or not record:
            raise NotConnectedError(
                "Google is not connected. Ask the user to run: jamie connect google"
            )
        data = {
            "client_id": self._client.client_id,
            "refresh_token": record["refresh_token"],
            "grant_type": "refresh_token",
        }
        if self._client.client_secret:
            data["client_secret"] = self._client.client_secret
        token = _token_request(self._http, data)
        self._access_token = token["access_token"]
        self._expires_at = self._clock() + float(token.get("expires_in", 3600))
        return self._access_token

    def access_token(self) -> str:
        if self._access_token and self._clock() < self._expires_at - REFRESH_MARGIN_SECONDS:
            return self._access_token
        return self._refresh()

    def request(self, method: str, url: str, **kwargs) -> dict:
        response = self._send(method, url, self.access_token(), **kwargs)
        if response.status_code == 401:
            response = self._send(method, url, self._refresh(), **kwargs)
            if response.status_code == 401:
                raise AuthExpiredError(
                    "Google rejected the stored sign-in. Ask the user to run: jamie connect google"
                )
        return _checked(response)

    def _send(self, method: str, url: str, token: str, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        try:
            return self._http.request(
                method, url, headers=headers, timeout=TIMEOUT_SECONDS, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot reach Google: {exc}") from exc


def _checked(response: httpx.Response) -> dict:
    try:
        payload = response.json() if response.content else {}
    except ValueError:
        payload = {}
    if response.status_code < 400:
        return payload if isinstance(payload, dict) else {"items": payload}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else str(error)
    reasons = {
        detail.get("reason", "")
        for detail in (error.get("errors", []) if isinstance(error, dict) else [])
    }
    if isinstance(error, dict):
        reasons |= {d.get("reason", "") for d in error.get("details", []) if isinstance(d, dict)}
    if response.status_code == 429 or reasons & {"rateLimitExceeded", "userRateLimitExceeded"}:
        raise RateLimitedError("Google rate limit reached; try again shortly.")
    if response.status_code == 403 and (
        reasons & {"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}
        or "insufficient" in message.lower()
    ):
        raise PermissionScopeError("The Google sign-in lacks permission for this operation.")
    if response.status_code == 404:
        raise NotFoundError("Not found in the Google account.")
    if response.status_code == 409:
        raise AlreadyExistsError("An item with this id already exists.")
    raise ProviderError(f"Google API error {response.status_code}: {message[:200]}")
