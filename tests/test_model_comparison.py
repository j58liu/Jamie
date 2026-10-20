import runpy
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-local-models.py"


def test_comparison_refuses_an_occupied_port_without_starting_a_process(tmp_path, monkeypatch):
    module = runpy.run_path(str(SCRIPT))

    def start(*args, **kwargs):
        raise AssertionError("An existing server must not be replaced.")

    monkeypatch.setattr(module["subprocess"], "Popen", start)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError), module["test_server"](["unused"], port, tmp_path / "log"):
            raise AssertionError("An occupied server was accepted.")


def test_comparison_stops_its_own_server_after_readiness_failure(tmp_path, monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    stopped = []

    class Process:
        def poll(self):
            return 0 if stopped else None

        def terminate(self):
            stopped.append("terminate")

        def wait(self, *, timeout):
            stopped.append("wait")

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return SimpleNamespace(status_code=503)

    clock = iter((0, 91))
    monkeypatch.setattr(module["subprocess"], "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(module["httpx"], "Client", lambda **kwargs: Client())
    monkeypatch.setattr(module["time"], "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="did not become ready"):
        with module["test_server"](["unused"], 0, tmp_path / "log"):
            raise AssertionError("An unready server was accepted.")
    assert stopped == ["terminate", "wait"]
