"""Fixed tool permissions and allow-or-confirm policy decisions."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class PermissionLevel(StrEnum):
    """Security classes assigned by application policy, outside the model."""

    READ = "READ"
    SAFE_WRITE = "SAFE_WRITE"
    SENSITIVE = "SENSITIVE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXTERNAL = "EXTERNAL"
    SEND = "SEND"  # sends a message as the user; confirmed every time


class PolicyAction(StrEnum):
    """The permission engine's decision before a tool can execute."""

    ALLOW = "allow"
    CONFIRM = "confirm"


class PolicyError(ValueError):
    """A tool is missing a policy or declares a conflicting permission."""


@dataclass(frozen=True)
class PolicyDecision:
    """A policy result derived only from trusted application configuration."""

    tool: str
    permission: PermissionLevel
    action: PolicyAction


DEFAULT_TOOL_PERMISSIONS: Mapping[str, PermissionLevel] = MappingProxyType(
    {
        "read_file": PermissionLevel.READ,
        "list_directory": PermissionLevel.READ,
        "find_files": PermissionLevel.READ,
        "search_files": PermissionLevel.READ,
        "extract_page": PermissionLevel.READ,
        "open_application": PermissionLevel.SAFE_WRITE,
        "reveal_in_finder": PermissionLevel.SAFE_WRITE,
        "clipboard_write": PermissionLevel.SAFE_WRITE,
        "move_file": PermissionLevel.SAFE_WRITE,
        "write_file": PermissionLevel.SAFE_WRITE,
        "edit_file": PermissionLevel.SAFE_WRITE,
        "create_directory": PermissionLevel.SAFE_WRITE,
        "shell": PermissionLevel.SENSITIVE,
        "clipboard_read": PermissionLevel.SENSITIVE,
        "run_shortcut": PermissionLevel.SENSITIVE,
        "run_applescript": PermissionLevel.SENSITIVE,
        "web_search": PermissionLevel.EXTERNAL,
        "fetch_url": PermissionLevel.EXTERNAL,
        "research": PermissionLevel.EXTERNAL,
        "delete_file": PermissionLevel.DESTRUCTIVE,
        "trash_file": PermissionLevel.SAFE_WRITE,
        # Personal-application connectors. Reads run without asking; anything that
        # changes or sends data outside this Mac asks first.
        "connectors_status": PermissionLevel.READ,
        "gmail_search_messages": PermissionLevel.READ,
        "gmail_get_message": PermissionLevel.READ,
        "gmail_get_thread": PermissionLevel.READ,
        "gmail_get_draft": PermissionLevel.READ,
        "gmail_create_draft": PermissionLevel.EXTERNAL,
        "gmail_send_draft": PermissionLevel.SEND,
        "calendar_list_events": PermissionLevel.READ,
        "calendar_check_availability": PermissionLevel.READ,
        "calendar_get_event": PermissionLevel.READ,
        "calendar_create_event": PermissionLevel.EXTERNAL,
        "calendar_update_event": PermissionLevel.EXTERNAL,
        "reminders_list": PermissionLevel.READ,
        "reminders_create": PermissionLevel.SAFE_WRITE,
        "reminders_complete": PermissionLevel.SAFE_WRITE,
    }
)

# Contacts the network but runs without asking. Only the model-written query is sent,
# to DuckDuckGo; it can contain text from the conversation or files the model read.
ALLOWED_WITHOUT_APPROVAL = frozenset({"web_search"})


class PermissionPolicy:
    """Classify tools and require confirmation for every non-read action but web search."""

    def __init__(self, permissions: Mapping[str, PermissionLevel] | None = None) -> None:
        source = DEFAULT_TOOL_PERMISSIONS if permissions is None else permissions
        self._permissions = MappingProxyType(dict(source))

    @property
    def permissions(self) -> Mapping[str, PermissionLevel]:
        return self._permissions

    def permission_for(self, tool: str) -> PermissionLevel:
        try:
            return self._permissions[tool]
        except KeyError as exc:
            raise PolicyError(f"Tool {tool!r} has no permission policy.") from exc

    def validate_declaration(self, tool: str, declared: PermissionLevel) -> None:
        configured = self.permission_for(tool)
        if declared is not configured:
            raise PolicyError(
                f"Tool {tool!r} declares {declared.value}, but policy requires {configured.value}."
            )

    def decide(self, tool: str) -> PolicyDecision:
        permission = self.permission_for(tool)
        allowed = permission is PermissionLevel.READ or tool in ALLOWED_WITHOUT_APPROVAL
        action = PolicyAction.ALLOW if allowed else PolicyAction.CONFIRM
        return PolicyDecision(tool=tool, permission=permission, action=action)
