"""Permission policy, trusted approvals, and tool audit records."""

from assistant.security.approvals import (
    ApprovalError,
    ApprovalProvider,
    ApprovalRequest,
    ApprovalStatus,
    CallbackApprovalProvider,
    DenyAllApprovals,
    authorize,
)
from assistant.security.audit import (
    AuditError,
    AuditEvent,
    AuditPhase,
    AuditSink,
    InMemoryAuditLog,
    JsonlAuditLog,
    redact_arguments,
)
from assistant.security.policy import (
    DEFAULT_TOOL_PERMISSIONS,
    PermissionLevel,
    PermissionPolicy,
    PolicyAction,
    PolicyDecision,
    PolicyError,
)

__all__ = [
    "DEFAULT_TOOL_PERMISSIONS",
    "ApprovalError",
    "ApprovalProvider",
    "ApprovalRequest",
    "ApprovalStatus",
    "AuditError",
    "AuditEvent",
    "AuditPhase",
    "AuditSink",
    "CallbackApprovalProvider",
    "DenyAllApprovals",
    "InMemoryAuditLog",
    "JsonlAuditLog",
    "PermissionLevel",
    "PermissionPolicy",
    "PolicyAction",
    "PolicyDecision",
    "PolicyError",
    "redact_arguments",
    "authorize",
]
