"""Approval requests for tool actions that policy does not allow automatically."""

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from assistant.agent.events import ApprovalRequested, emit_event
from assistant.security.policy import PermissionLevel, PolicyAction, PolicyDecision


class ApprovalStatus(StrEnum):
    """How an action satisfied its policy decision."""

    NOT_REQUIRED = "not_required"
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalError(RuntimeError):
    """An approval provider failed or returned an invalid answer."""


class ApprovalRequest(BaseModel):
    """A confirmation request bound to one tool and normalized argument set."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    tool: str
    permission: PermissionLevel
    arguments: dict[str, JsonValue]


class ApprovalProvider(Protocol):
    """Present a trusted confirmation prompt outside the model conversation."""

    def request(self, approval: ApprovalRequest) -> bool: ...


class DenyAllApprovals:
    """Fail closed when no user-facing approval provider is configured."""

    def request(self, approval: ApprovalRequest) -> bool:
        return False


class CallbackApprovalProvider:
    """Adapt a user-interface callback to the approval provider contract."""

    def __init__(self, callback: Callable[[ApprovalRequest], bool]) -> None:
        self._callback = callback

    def request(self, approval: ApprovalRequest) -> bool:
        return self._callback(approval)


def authorize(
    decision: PolicyDecision,
    arguments: dict[str, JsonValue],
    provider: ApprovalProvider,
) -> ApprovalStatus:
    """Apply a trusted policy decision and ask the provider only when required."""
    if decision.action is PolicyAction.ALLOW:
        return ApprovalStatus.NOT_REQUIRED

    request = ApprovalRequest(
        tool=decision.tool,
        permission=decision.permission,
        arguments=arguments,
    )
    emit_event(ApprovalRequested(request.model_copy(deep=True)))
    try:
        approved = provider.request(request)
    except Exception as exc:
        raise ApprovalError("The approval provider failed.") from exc
    if not isinstance(approved, bool):
        # CallbackApprovalProvider wraps arbitrary UI code; "no" is truthy.
        raise ApprovalError("The approval provider must return a boolean.")
    return ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
