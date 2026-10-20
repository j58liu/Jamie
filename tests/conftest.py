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


@pytest.fixture
def confirm_every_tool(monkeypatch: pytest.MonkeyPatch):
    """Exercise the approval path: confirm every non-read tool but web search."""
    from assistant.security import policy

    confirmed = {
        tool
        for tool, level in policy.DEFAULT_TOOL_PERMISSIONS.items()
        if level is not policy.PermissionLevel.READ and tool != "web_search"
    }
    monkeypatch.setattr(policy, "CONFIRM_REQUIRED", frozenset(confirmed))


@pytest.fixture(autouse=True)
def isolated_app_install(monkeypatch: pytest.MonkeyPatch, tmp_path_factory):
    """Tests never read the real Jamie.app or write a real login item."""
    from assistant.ui import login_item

    root = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(login_item, "APP_LOCATIONS", (root / "Applications" / "Jamie.app",))
    monkeypatch.setattr(login_item, "LOGIN_AGENT", root / "LaunchAgents" / "jamie.plist")
    return root
