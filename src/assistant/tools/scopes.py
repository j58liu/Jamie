"""Explicit tool sets; restricting model context never changes permission policy."""

from enum import StrEnum


class ToolScope(StrEnum):
    ALL = "all"
    FILES = "files"
    READ = "read"
    WEB = "web"
    MACOS = "macos"


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
}


# Scheduled tasks run while nobody is watching: they may read and search, never act.
UNATTENDED_TOOLS = READ_TOOLS | SCOPE_TOOLS[ToolScope.WEB]


def select_tools(registry, scope: ToolScope | str):
    scope = ToolScope(scope)
    return registry if scope is ToolScope.ALL else registry.subset(SCOPE_TOOLS[scope])
