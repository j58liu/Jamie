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
