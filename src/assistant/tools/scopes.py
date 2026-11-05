"""Explicit tool sets; restricting model context never changes permission policy."""

from enum import StrEnum


class ToolScope(StrEnum):
    AUTO = "auto"  # Jamie picks the tools for each message (see assistant.routing)
    ALL = "all"
    FILES = "files"
    READ = "read"
    WEB = "web"
    MACOS = "macos"
    PERSONAL = "personal"
    TASKS = "tasks"  # Jamie's own background work and schedules
    BROWSER = "browser"  # websites and logged-in accounts


READ_TOOLS = {"read_file", "list_directory", "find_files", "search_files"}
# Files anywhere on the Mac; see assistant.tools.mac_files.
MAC_READ_TOOLS = {"mac_search_files", "mac_recent_files", "mac_list_folder", "mac_read_file"}
MAC_FILE_TOOLS = MAC_READ_TOOLS | {
    "mac_open_file",
    "mac_copy_file",
    "mac_move_file",
    "mac_create_folder",
}
SCOPE_TOOLS = {
    ToolScope.READ: READ_TOOLS | MAC_READ_TOOLS,
    ToolScope.FILES: READ_TOOLS
    | MAC_FILE_TOOLS
    | {
        "write_file",
        "edit_file",
        "move_file",
        "create_directory",
        "trash_file",
    },
    ToolScope.WEB: {"web_search", "fetch_url", "extract_page", "research"},
    ToolScope.MACOS: READ_TOOLS
    | {
        "open_application",
        "reveal_in_finder",
        "clipboard_read",
        "clipboard_write",
        "run_shortcut",
        "run_applescript",
    },
    # Only connector tools, so a small model sees a short, relevant tool list.
    ToolScope.PERSONAL: {
        "connectors_status",
        "gmail_search_messages",
        "gmail_get_message",
        "gmail_get_thread",
        "gmail_create_draft",
        "gmail_get_draft",
        "gmail_send_draft",
        "calendar_list_events",
        "calendar_check_availability",
        "calendar_get_event",
        "calendar_create_event",
        "calendar_update_event",
        "reminders_list",
        "reminders_create",
        "reminders_complete",
    },
}

SCOPE_TOOLS[ToolScope.TASKS] = {
    "jamie_work_status",
    "jamie_start_background",
    "jamie_cancel_work",
    "jamie_schedule",
}

SCOPE_TOOLS[ToolScope.BROWSER] = {
    "browser_open",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_select",
    "browser_check",
    "browser_scroll",
    "browser_back",
    "browser_upload",
    "browser_submit",
    "browser_screenshot",
    "browser_show",
    "mac_search_files",
    "web_search",
}

# Multi-step personal workflows (search, read, check, draft) need more model actions.
SCOPE_MAX_STEPS = {ToolScope.PERSONAL: 16, ToolScope.BROWSER: 30}


# Scheduled tasks run while nobody is watching: they may read and search, never act.
UNATTENDED_TOOLS = READ_TOOLS | MAC_READ_TOOLS | SCOPE_TOOLS[ToolScope.WEB]


def select_tools(registry, scope: ToolScope | str):
    scope = ToolScope(scope)
    if scope in (ToolScope.ALL, ToolScope.AUTO):
        return registry
    return registry.subset(SCOPE_TOOLS[scope])


def select_scopes(registry, scopes) -> object:
    """Tools for several scopes at once; "all" or an unknown scope keeps every tool."""
    names: set[str] = set()
    for scope in scopes:
        try:
            scope = ToolScope(scope)
        except ValueError:
            return registry
        if scope in (ToolScope.ALL, ToolScope.AUTO):
            return registry
        names |= SCOPE_TOOLS[scope]
    return registry.subset(names) if names else registry


def max_steps_for(scopes) -> int | None:
    steps = [SCOPE_MAX_STEPS[ToolScope(s)] for s in scopes if s in SCOPE_MAX_STEPS]
    return max(steps) if steps else None
