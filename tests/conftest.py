import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch):
    """Tests use fakes and mock transports; a real connection means a test reached out."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"Tests must not open network connections: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect)


class FakeGoogleApi:
    """Route Google API requests to handlers and record what was sent."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], object] = {}
        self.requests: list = []

    def on(self, method: str, path: str, response) -> None:
        """``response`` is a JSON-able value, a (status, body) pair, or a callable."""
        self.routes[(method, path)] = response

    def sent(self, method: str, path: str) -> list:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def handler(self, request):
        import json

        import httpx

        self.requests.append(request)
        route = self.routes.get((request.method, request.url.path))
        if route is None:
            return httpx.Response(404, json={"error": {"message": "no route"}})
        if callable(route):
            route = route(request)
        status, body = route if isinstance(route, tuple) else (200, route)
        return httpx.Response(
            status, content=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )


@pytest.fixture
def google_api():
    """A connected GoogleSession backed by FakeGoogleApi; pass the scopes it holds."""
    import json

    import httpx

    from assistant.connectors.credentials import InMemoryCredentialStore
    from assistant.connectors.google.auth import (
        CREDENTIAL_ACCOUNT,
        CREDENTIAL_SERVICE,
        GoogleClientConfig,
        GoogleSession,
    )

    api = FakeGoogleApi()
    api.on("POST", "/token", {"access_token": "access-1", "expires_in": 3600})

    def make(scopes):
        store = InMemoryCredentialStore()
        store.set(
            CREDENTIAL_SERVICE,
            CREDENTIAL_ACCOUNT,
            json.dumps({"refresh_token": "refresh-1", "scopes": list(scopes)}),
        )
        http = httpx.Client(transport=httpx.MockTransport(api.handler))
        return GoogleSession(GoogleClientConfig("client-1"), store, http=http, clock=lambda: 1000.0)

    api.session = make
    return api
