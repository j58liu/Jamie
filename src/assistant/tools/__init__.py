"""Safe tools available to Jamie's agent loop."""

from assistant.tools.base import Risk, Tool, ToolError
from assistant.tools.changes import WorkspaceChanges
from assistant.tools.editing import (
    CreateDirectoryTool,
    EditFileTool,
    MoveFileTool,
    TrashFileTool,
    WriteFileTool,
)
from assistant.tools.filesystem import (
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
)
from assistant.tools.macos import (
    ClipboardReadTool,
    ClipboardWriteTool,
    CommandResult,
    CommandRunner,
    OpenApplicationTool,
    RevealInFinderTool,
    RunAppleScriptTool,
    RunShortcutTool,
    SubprocessCommandRunner,
)
from assistant.tools.registry import (
    ToolRegistry,
    create_macos_registry,
    create_read_only_registry,
    create_workspace_registry,
)

__all__ = [
    "ClipboardReadTool",
    "ClipboardWriteTool",
    "CommandResult",
    "CommandRunner",
    "CreateDirectoryTool",
    "EditFileTool",
    "FindFilesTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "OpenApplicationTool",
    "ReadFileTool",
    "Risk",
    "RevealInFinderTool",
    "RunAppleScriptTool",
    "RunShortcutTool",
    "SearchFilesTool",
    "SubprocessCommandRunner",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "TrashFileTool",
    "WorkspaceChanges",
    "WriteFileTool",
    "create_macos_registry",
    "create_read_only_registry",
    "create_workspace_registry",
]
