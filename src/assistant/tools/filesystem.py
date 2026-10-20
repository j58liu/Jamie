"""Read-only filesystem tools restricted to one workspace root."""

from __future__ import annotations

from itertools import islice
from pathlib import Path, PurePath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from assistant.tools.base import Risk, ToolError

MAX_OUTPUT_ITEMS = 2_000
MAX_SEARCH_LINE_LENGTH = 1_000


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _workspace_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Workspace root cannot be resolved: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError("Workspace root must be a directory.")
    return resolved


def _resolve(root: Path, requested: Path) -> Path:
    if requested.is_absolute():
        raise ToolError("Paths must be relative to the workspace root.")
    try:
        resolved = (root / requested).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolError(
            f"Path does not exist or cannot be accessed: {requested}. "
            "Use find_files to search for it by name."
        ) from exc
    if not resolved.is_relative_to(root):
        raise ToolError("Path resolves outside the workspace root.")
    return resolved


def _validate_pattern(pattern: str) -> str:
    if not pattern.strip():
        raise ValueError("Pattern must not be blank.")
    path = PurePath(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Pattern must stay within the requested directory.")
    return pattern


def _display_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _matching_paths(root: Path, directory: Path, pattern: str, *, directories: bool = False):
    """Files (and directories, if asked) under ``directory``, skipping hidden folders.

    Hidden folders such as .git and .venv would otherwise fill the results.
    """
    try:
        candidates = sorted(directory.rglob(pattern), key=lambda path: path.as_posix())
    except (OSError, ValueError) as exc:
        raise ToolError(f"Cannot search with pattern {pattern!r}.") from exc
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        if any(part.startswith(".") for part in candidate.relative_to(directory).parts[:-1]):
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_relative_to(root) and (resolved.is_file() or directories):
            yield resolved


class ReadFileTool:
    name = "read_file"
    description = "Read a range of lines from a UTF-8 text file inside the workspace."
    risk = Risk.READ

    class Input(_Input):
        path: Path
        offset: int = Field(default=0, ge=0, strict=True)
        limit: int = Field(default=400, ge=1, le=MAX_OUTPUT_ITEMS, strict=True)

    input_schema = Input

    def __init__(self, root: Path | None = None) -> None:
        self._root = _workspace_root(root if root is not None else Path.cwd())

    def execute(self, arguments: Input) -> str:
        path = _resolve(self._root, arguments.path)
        if not path.is_file():
            raise ToolError(f"Path is not a file: {arguments.path}")
        try:
            with path.open(encoding="utf-8") as file:
                return "".join(islice(file, arguments.offset, arguments.offset + arguments.limit))
        except UnicodeError as exc:
            raise ToolError(f"File is not valid UTF-8 text: {arguments.path}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot read file: {arguments.path}") from exc


class ListDirectoryTool:
    name = "list_directory"
    description = "List immediate entries in a directory inside the workspace."
    risk = Risk.READ

    class Input(_Input):
        path: Path = Path(".")
        limit: int = Field(default=400, ge=1, le=MAX_OUTPUT_ITEMS, strict=True)

    input_schema = Input

    def __init__(self, root: Path | None = None) -> None:
        self._root = _workspace_root(root if root is not None else Path.cwd())

    def execute(self, arguments: Input) -> str:
        path = _resolve(self._root, arguments.path)
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {arguments.path}")
        try:
            entries = sorted(path.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            raise ToolError(f"Cannot list directory: {arguments.path}") from exc

        lines = []
        for entry in entries[: arguments.limit]:
            suffix = "@" if entry.is_symlink() else "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        if len(entries) > arguments.limit:
            lines.append("[truncated]")
        return "\n".join(lines) if lines else "Directory is empty."


class FindFilesTool:
    name = "find_files"
    description = (
        "Find files and folders recursively by a glob pattern inside the workspace. "
        "Folders end with /."
    )
    risk = Risk.READ

    class Input(_Input):
        path: Path = Path(".")
        pattern: str
        limit: int = Field(default=200, ge=1, le=MAX_OUTPUT_ITEMS, strict=True)

        @field_validator("pattern")
        @classmethod
        def validate_pattern(cls, value: str) -> str:
            return _validate_pattern(value)

    input_schema = Input

    def __init__(self, root: Path | None = None) -> None:
        self._root = _workspace_root(root if root is not None else Path.cwd())

    def execute(self, arguments: Input) -> str:
        path = _resolve(self._root, arguments.path)
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {arguments.path}")

        matches = []
        truncated = False
        for candidate in _matching_paths(self._root, path, arguments.pattern, directories=True):
            if len(matches) == arguments.limit:
                truncated = True
                break
            suffix = "/" if candidate.is_dir() else ""
            matches.append(f"{_display_path(self._root, candidate)}{suffix}")
        if truncated:
            matches.append("[truncated]")
        return "\n".join(matches) if matches else "No files found."


class SearchFilesTool:
    name = "search_files"
    description = "Search UTF-8 text files recursively for a literal query."
    risk = Risk.READ

    class Input(_Input):
        query: str = Field(min_length=1)
        path: Path = Path(".")
        pattern: str = "*"
        case_sensitive: bool = True
        limit: int = Field(default=100, ge=1, le=MAX_OUTPUT_ITEMS, strict=True)

        @field_validator("query")
        @classmethod
        def validate_query(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("Query must not be blank.")
            return value

        @field_validator("pattern")
        @classmethod
        def validate_pattern(cls, value: str) -> str:
            return _validate_pattern(value)

    input_schema = Input

    def __init__(self, root: Path | None = None) -> None:
        self._root = _workspace_root(root if root is not None else Path.cwd())

    def execute(self, arguments: Input) -> str:
        path = _resolve(self._root, arguments.path)
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {arguments.path}")

        query = arguments.query if arguments.case_sensitive else arguments.query.casefold()
        matches = []
        for candidate in _matching_paths(self._root, path, arguments.pattern):
            try:
                with candidate.open(encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        text = line if arguments.case_sensitive else line.casefold()
                        if query not in text:
                            continue
                        display = line.rstrip("\r\n")
                        if len(display) > MAX_SEARCH_LINE_LENGTH:
                            display = f"{display[:MAX_SEARCH_LINE_LENGTH]}…"
                        match = f"{_display_path(self._root, candidate)}:{line_number}:{display}"
                        if len(matches) == arguments.limit:
                            return "\n".join([*matches, "[truncated]"])
                        matches.append(match)
            except (OSError, UnicodeError):
                continue
        return "\n".join(matches) if matches else "No matches found."
