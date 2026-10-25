"""Explicit tool sets; restricting model context never changes permission policy."""

from enum import StrEnum


class ToolScope(StrEnum):
    ALL = "all"
    FILES = "files"
    READ = "read"
    WEB = "web"
    MACOS = "macos"
    PERSONAL = "personal"


READ_TOOLS = {"read_file", "list_directory", "find_files", "search_files"}
SCOPE_TOOLS = {
    ToolScope.READ: READ_TOOLS,
    ToolScope.FILES: READ_TOOLS
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

# Multi-step personal workflows (search, read, check, draft) need more model actions.
SCOPE_MAX_STEPS = {ToolScope.PERSONAL: 16}


def select_tools(registry, scope: ToolScope | str):
    scope = ToolScope(scope)
    return registry if scope is ToolScope.ALL else registry.subset(SCOPE_TOOLS[scope])
