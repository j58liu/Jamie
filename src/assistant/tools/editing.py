"""Approved workspace editing tools with snapshots and reversible changes."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError
from assistant.tools.changes import WorkspaceChanges

MAX_WRITE_CHARACTERS = 2_000_000


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _atomic_write(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except OSError as exc:
        raise ToolError(f"Cannot write file: {path.name}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class WriteFileTool:
    name = "write_file"
    description = "Create or replace one UTF-8 text file inside the workspace."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: Path
        content: str = Field(max_length=MAX_WRITE_CHARACTERS)

    input_schema = Input

    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes

    def execute(self, arguments: Input) -> str:
        path = self._changes.resolve(arguments.path, must_exist=False)
        if path.exists() and not path.is_file():
            raise ToolError(f"Path is not a file: {arguments.path}")
        if not path.parent.is_dir():
            raise ToolError(f"Parent directory does not exist: {arguments.path.parent}")
        try:
            self._changes.snapshot(path)
            _atomic_write(path, arguments.content)
        except OSError as exc:
            raise ToolError(f"Cannot write file: {arguments.path}") from exc
        self._changes.require_file_read(path)
        return (
            f"Wrote {len(arguments.content)} characters to {arguments.path.as_posix()}."
            f"\n\nDiff:\n{self._changes.diff()}"
        )


class EditFileTool:
    name = "edit_file"
    description = "Replace exact text in one UTF-8 file inside the workspace."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: Path
        old_text: str = Field(min_length=1, max_length=MAX_WRITE_CHARACTERS)
        new_text: str = Field(max_length=MAX_WRITE_CHARACTERS)
        replace_all: bool = False

    input_schema = Input

    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes

    def execute(self, arguments: Input) -> str:
        path = self._changes.resolve(arguments.path, must_exist=True)
        if not path.is_file():
            raise ToolError(f"Path is not a file: {arguments.path}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ToolError(f"File is not valid UTF-8 text: {arguments.path}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot read file: {arguments.path}") from exc
        occurrences = content.count(arguments.old_text)
        if occurrences == 0:
            raise ToolError("The exact old_text was not found.")
        if occurrences > 1 and not arguments.replace_all:
            raise ToolError(
                f"The exact old_text occurs {occurrences} times; set replace_all=true "
                "to replace every occurrence."
            )
        replacement_count = occurrences if arguments.replace_all else 1
        updated = content.replace(
            arguments.old_text,
            arguments.new_text,
            -1 if arguments.replace_all else 1,
        )
        if len(updated) > MAX_WRITE_CHARACTERS:
            raise ToolError("The edited file would exceed the supported text size.")
        try:
            self._changes.snapshot(path)
            _atomic_write(path, updated)
        except OSError as exc:
            raise ToolError(f"Cannot edit file: {arguments.path}") from exc
        self._changes.require_file_read(path)
        return (
            f"Replaced {replacement_count} occurrence(s) in {arguments.path.as_posix()}."
            f"\n\nDiff:\n{self._changes.diff()}"
        )


class MoveFileTool:
    name = "move_file"
    description = "Move one file or directory to an unused path inside the workspace."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        source: Path
        destination: Path

    input_schema = Input

    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes

    def execute(self, arguments: Input) -> str:
        source = self._changes.resolve(arguments.source, must_exist=True)
        destination = self._changes.resolve(arguments.destination, must_exist=False)
        if source == self._changes.root:
            raise ToolError("The workspace root cannot be moved.")
        if destination.exists():
            raise ToolError(f"Destination already exists: {arguments.destination}")
        if not destination.parent.is_dir():
            raise ToolError(f"Parent directory does not exist: {arguments.destination.parent}")
        try:
            self._changes.snapshot(source, destination)
            source.rename(destination)
        except OSError as exc:
            raise ToolError(f"Cannot move {arguments.source} to {arguments.destination}.") from exc
        if destination.is_file():
            self._changes.require_file_read(destination)
        else:
            self._changes.require_directory_check(destination)
        self._changes.require_directory_check(source.parent)
        return (
            f"Moved {arguments.source.as_posix()} to {arguments.destination.as_posix()}."
            f"\n\nDiff:\n{self._changes.diff()}"
        )


class CreateDirectoryTool:
    name = "create_directory"
    description = "Create a directory inside the workspace."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: Path
        parents: bool = False

    input_schema = Input

    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes

    def execute(self, arguments: Input) -> str:
        path = self._changes.resolve(arguments.path, must_exist=False)
        if path == self._changes.root:
            raise ToolError("The workspace root already exists.")
        if path.exists():
            raise ToolError(f"Path already exists: {arguments.path}")
        snapshot_path = path
        if arguments.parents:
            while snapshot_path.parent != self._changes.root and not snapshot_path.parent.exists():
                snapshot_path = snapshot_path.parent
        try:
            self._changes.snapshot(snapshot_path)
            path.mkdir(parents=arguments.parents)
        except OSError as exc:
            raise ToolError(f"Cannot create directory: {arguments.path}") from exc
        self._changes.require_directory_check(path)
        return f"Created directory {arguments.path.as_posix()}.\n\nDiff:\n{self._changes.diff()}"


class TrashFileTool:
    name = "trash_file"
    description = "Remove one file or directory from the workspace with an undo snapshot."
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: Path

    input_schema = Input

    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes

    def execute(self, arguments: Input) -> str:
        path = self._changes.resolve(arguments.path, must_exist=True)
        if path == self._changes.root:
            raise ToolError("The workspace root cannot be trashed.")
        parent = path.parent
        try:
            self._changes.snapshot(path)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                raise ToolError(f"Unsupported filesystem entry: {arguments.path}")
        except OSError as exc:
            raise ToolError(f"Cannot trash path: {arguments.path}") from exc
        self._changes.require_directory_check(parent)
        return f"Trashed {arguments.path.as_posix()}.\n\nDiff:\n{self._changes.diff()}"
