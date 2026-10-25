"""Secret storage for connector credentials.

On macOS secrets live in the login Keychain, written through ``security -i`` so the
secret is passed on stdin rather than on a visible command line.
"""

from __future__ import annotations

from typing import Protocol

from assistant.tools.base import ToolError
from assistant.tools.macos import CommandRunner, SubprocessCommandRunner

KEYCHAIN_SERVICE_PREFIX = "com.jamie.assistant"
SECURITY = "/usr/bin/security"
TIMEOUT_SECONDS = 15


class CredentialStore(Protocol):
    def get(self, service: str, account: str) -> str | None: ...

    def set(self, service: str, account: str, secret: str) -> None: ...

    def delete(self, service: str, account: str) -> None: ...


class CredentialError(ToolError):
    """The credential store could not be read or written."""


class InMemoryCredentialStore:
    """Process-local store for tests and platforms without Keychain."""

    def __init__(self) -> None:
        self._secrets: dict[tuple[str, str], str] = {}

    def get(self, service: str, account: str) -> str | None:
        return self._secrets.get((service, account))

    def set(self, service: str, account: str, secret: str) -> None:
        self._secrets[(service, account)] = secret

    def delete(self, service: str, account: str) -> None:
        self._secrets.pop((service, account), None)


def _quote(value: str) -> str:
    if any(character in value for character in "\x00\r\n"):
        raise CredentialError("Credential values must be a single line.")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class KeychainCredentialStore:
    """Generic passwords in the macOS login Keychain."""

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessCommandRunner()

    @staticmethod
    def _service(service: str) -> str:
        return f"{KEYCHAIN_SERVICE_PREFIX}.{service}"

    def get(self, service: str, account: str) -> str | None:
        try:
            result = self._runner.run(
                [
                    SECURITY,
                    "find-generic-password",
                    "-s",
                    self._service(service),
                    "-a",
                    account,
                    "-w",
                ],
                timeout=TIMEOUT_SECONDS,
            )
        except ToolError as exc:
            if "could not be found" in str(exc) or "(44)" in str(exc):
                return None
            raise CredentialError(f"Cannot read Keychain: {exc}") from exc
        secret = result.stdout.rstrip("\n")
        return secret or None

    def set(self, service: str, account: str, secret: str) -> None:
        command = (
            f"add-generic-password -U -s {_quote(self._service(service))} "
            f"-a {_quote(account)} -w {_quote(secret)}\n"
        )
        try:
            self._runner.run([SECURITY, "-i"], input_text=command, timeout=TIMEOUT_SECONDS)
        except ToolError as exc:
            raise CredentialError(f"Cannot write Keychain: {exc}") from exc

    def delete(self, service: str, account: str) -> None:
        try:
            self._runner.run(
                [SECURITY, "delete-generic-password", "-s", self._service(service), "-a", account],
                timeout=TIMEOUT_SECONDS,
            )
        except ToolError as exc:
            if "could not be found" in str(exc) or "(44)" in str(exc):
                return
            raise CredentialError(f"Cannot delete from Keychain: {exc}") from exc


def default_credential_store() -> CredentialStore:
    import sys

    return KeychainCredentialStore() if sys.platform == "darwin" else InMemoryCredentialStore()
