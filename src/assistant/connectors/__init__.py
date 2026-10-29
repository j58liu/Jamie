"""Connectors adapt personal-application APIs to Jamie's tool interface.

Connector: makes provider calls. Connector tool (assistant.tools.integrations): exposes
one typed operation. Registry: validates arguments. Security policy: decides whether the
action may run. Agent: chooses the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import tzinfo

from assistant.connectors.base import (
    Capability,
    ConnectionStatus,
    Connector,
    ConnectorInfo,
    ConnectorTool,
)
from assistant.connectors.errors import (
    AuthExpiredError,
    ConnectorError,
    NotConnectedError,
    NotFoundError,
    PermissionScopeError,
    ProviderError,
    RateLimitedError,
    UnavailableError,
)


@dataclass
class Connectors:
    """The connectors available to one Jamie process; any of them may be absent."""

    tz: tzinfo
    gmail: object | None = None
    calendar: object | None = None
    reminders: object | None = None
    google: object | None = None  # the shared GoogleSession, for connect/disconnect
    extra: list[Connector] = field(default_factory=list)

    def all(self) -> list[Connector]:
        found = [c for c in (self.gmail, self.calendar, self.reminders) if c is not None]
        return [*found, *self.extra]  # type: ignore[list-item]


def build_connectors(config, *, store=None, runner=None) -> Connectors:
    """Create connectors from configuration; nothing contacts a provider here."""
    from pathlib import Path

    from assistant.connectors.apple.reminders import RemindersConnector
    from assistant.connectors.credentials import default_credential_store
    from assistant.connectors.google.auth import GoogleClientConfig, GoogleSession
    from assistant.connectors.google.calendar import CalendarConnector
    from assistant.connectors.google.gmail import GmailConnector
    from assistant.connectors.timeutil import local_zone

    tz = local_zone(config.timezone)
    store = store if store is not None else default_credential_store()
    client = None
    if config.google.client_secrets_path:
        try:
            client = GoogleClientConfig.from_file(Path(config.google.client_secrets_path))
        except NotConnectedError:
            client = None
    elif config.google.client_id:
        client = GoogleClientConfig(config.google.client_id, config.google.client_secret)
    session = GoogleSession(client, store)
    helper = (
        Path(config.reminders.helper_path).expanduser() if config.reminders.helper_path else None
    )
    return Connectors(
        tz=tz,
        gmail=GmailConnector(session) if config.google.enabled else None,
        calendar=CalendarConnector(session, tz) if config.google.enabled else None,
        reminders=RemindersConnector(helper, runner=runner, tz=tz)
        if config.reminders.enabled
        else None,
        google=session,
    )


__all__ = [
    "AuthExpiredError",
    "Capability",
    "ConnectionStatus",
    "Connector",
    "ConnectorError",
    "ConnectorInfo",
    "ConnectorTool",
    "Connectors",
    "NotConnectedError",
    "NotFoundError",
    "PermissionScopeError",
    "ProviderError",
    "RateLimitedError",
    "UnavailableError",
    "build_connectors",
]
