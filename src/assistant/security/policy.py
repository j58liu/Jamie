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
        "send_email": PermissionLevel.EXTERNAL,
        "web_search": PermissionLevel.EXTERNAL,
        "fetch_url": PermissionLevel.EXTERNAL,
        "research": PermissionLevel.EXTERNAL,
        "delete_file": PermissionLevel.DESTRUCTIVE,
        "trash_file": PermissionLevel.SAFE_WRITE,
    }
)

# The user chose to confirm only actions that cannot be taken back: destructive ones,
# email that reaches other people, and arbitrary shell commands (not registered today).
# Everything else, including AppleScript, clipboard reads, and network requests carrying
# model-written text, runs without asking; edits stay reversible with /undo.
CONFIRM_REQUIRED = frozenset({"send_email", "shell"})


class PermissionPolicy:
    """Classify tools and require confirmation only for irreversible actions."""

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
        confirm = permission is PermissionLevel.DESTRUCTIVE or tool in CONFIRM_REQUIRED
        action = PolicyAction.CONFIRM if confirm else PolicyAction.ALLOW
        return PolicyDecision(tool=tool, permission=permission, action=action)
