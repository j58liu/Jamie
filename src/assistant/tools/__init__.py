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
from assistant.tools.registry import (
    ToolRegistry,
    create_read_only_registry,
    create_workspace_registry,
)

__all__ = [
    "CreateDirectoryTool",
    "EditFileTool",
    "FindFilesTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadFileTool",
    "Risk",
    "SearchFilesTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "TrashFileTool",
    "WorkspaceChanges",
    "WriteFileTool",
    "create_read_only_registry",
    "create_workspace_registry",
]
