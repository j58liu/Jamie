"""Typed connector failures; each is a ToolError so the registry reports it to the model."""

from assistant.tools.base import ToolError


class ConnectorError(ToolError):
    """A connector could not complete a provider operation."""


class NotConnectedError(ConnectorError):
    """The account has not been connected, or its credentials were removed."""


class AuthExpiredError(ConnectorError):
    """Stored credentials were rejected; the user must reconnect."""


class PermissionScopeError(ConnectorError):
    """The connected account did not grant a scope this operation needs."""

    def __init__(self, message: str, scope: str | None = None) -> None:
        super().__init__(message)
        self.scope = scope


class RateLimitedError(ConnectorError):
    """The provider asked Jamie to slow down."""


class NotFoundError(ConnectorError):
    """The requested item does not exist or is not visible to this account."""


class ProviderError(ConnectorError):
    """The provider failed or returned a response Jamie could not use."""


class UnavailableError(ConnectorError):
    """The connector cannot run on this machine (platform, missing helper, permission)."""
