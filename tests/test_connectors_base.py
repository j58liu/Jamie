import json

import pytest
from pydantic import BaseModel

from assistant.connectors import ConnectorError, NotFoundError
from assistant.connectors.base import ConnectorInput, ConnectorTool, to_json
from assistant.connectors.credentials import (
    CredentialError,
    InMemoryCredentialStore,
    KeychainCredentialStore,
)
from assistant.security.policy import PermissionLevel, PermissionPolicy
from assistant.tools import CommandResult, ToolError, ToolRegistry


class Item(BaseModel):
    id: str
    note: str | None = None


def test_to_json_is_compact_and_drops_empty_fields():
    assert to_json([Item(id="a"), Item(id="b", note="x")]) == '[{"id":"a"},{"id":"b","note":"x"}]'


def test_connector_errors_are_tool_errors_reported_to_the_model():
    class Failing(ConnectorTool):
        name = "gmail_get_message"
        description = "fails"
        risk = PermissionLevel.READ

        class Input(ConnectorInput):
            message_id: str

        input_schema = Input

        def run(self, arguments):
            raise NotFoundError("Not found in the Google account.")

    assert issubclass(ConnectorError, ToolError)
    registry = ToolRegistry(policy=PermissionPolicy())
    registry.register(Failing())
    result = registry.execute("gmail_get_message", {"message_id": "x"})
    assert not result.success
    assert result.output == "Not found in the Google account."


def test_in_memory_store_round_trip():
    store = InMemoryCredentialStore()
    assert store.get("google", "default") is None
    store.set("google", "default", "secret")
    assert store.get("google", "default") == "secret"
    store.delete("google", "default")
    assert store.get("google", "default") is None


class KeychainRunner:
    def __init__(self, stdout="", error=None):
        self.stdout, self.error = stdout, error
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append((arguments, input_text))
        if self.error:
            raise ToolError(self.error)
        return CommandResult(stdout=self.stdout, stderr="")


def test_keychain_reads_generic_password():
    runner = KeychainRunner(stdout="token-value\n")
    assert KeychainCredentialStore(runner).get("google", "default") == "token-value"
    arguments, _ = runner.calls[0]
    assert arguments[:2] == ["/usr/bin/security", "find-generic-password"]
    assert "com.jamie.assistant.google" in arguments


def test_keychain_missing_item_is_none():
    runner = KeychainRunner(error="macOS command failed (44): could not be found")
    assert KeychainCredentialStore(runner).get("google", "default") is None


def test_keychain_write_keeps_secret_off_the_command_line():
    runner = KeychainRunner()
    KeychainCredentialStore(runner).set("google", "default", 'se"cret')
    arguments, stdin = runner.calls[0]
    assert arguments == ["/usr/bin/security", "-i"]
    assert 'se\\"cret' in stdin
    assert all("cret" not in argument for argument in arguments)


def test_keychain_rejects_multiline_secrets():
    with pytest.raises(CredentialError):
        KeychainCredentialStore(KeychainRunner()).set("google", "default", "a\nb")


def test_keychain_reports_other_failures():
    runner = KeychainRunner(error="macOS command failed (51): user interaction not allowed")
    with pytest.raises(CredentialError):
        KeychainCredentialStore(runner).get("google", "default")
    assert json.dumps  # keep json import used for readability of other tests
