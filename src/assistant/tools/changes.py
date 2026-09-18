"""Per-turn workspace snapshots, diffs, verification, and undo."""

from __future__ import annotations

import difflib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from uuid import uuid4

from assistant.tools.base import ToolError


@dataclass(frozen=True)
class _SnapshotEntry:
    path: Path
    kind: str
    backup: Path | None


@dataclass
class _Turn:
    id: str = field(default_factory=lambda: uuid4().hex)
    entries: dict[Path, _SnapshotEntry] = field(default_factory=dict)
    pending: set[tuple[str, Path]] = field(default_factory=set)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _has_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    return path.is_dir() and any(item.is_symlink() for item in path.rglob("*"))


def _decode(content: bytes) -> list[str] | None:
    try:
        return content.decode("utf-8").splitlines()
    except UnicodeError:
        return None


def _file_diff(path: str, before: bytes | None, after: bytes | None) -> str:
    old = _decode(before) if before is not None else []
    new = _decode(after) if after is not None else []
    if old is None or new is None:
        return f"Binary file changed: {path}"
    difference = "\n".join(
        difflib.unified_diff(
            old,
            new,
            fromfile=f"a/{path}" if before is not None else "/dev/null",
            tofile=f"b/{path}" if after is not None else "/dev/null",
            lineterm="",
        )
    )
    if difference:
        return difference
    if before is None and after is not None:
        return f"+{path}"
    if before is not None and after is None:
        return f"-{path}"
    return ""


class WorkspaceChanges:
    """Track reversible filesystem changes in isolated turns."""

    def __init__(self, root: Path) -> None:
        try:
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Workspace root cannot be resolved: {exc}") from exc
        if not resolved.is_dir():
            raise ValueError("Workspace root must be a directory.")
        self.root = resolved
        self._storage = tempfile.TemporaryDirectory(prefix="jamie-snapshots-")
        self._storage_path = Path(self._storage.name)
        self._active: _Turn | None = None
        self._history: list[_Turn] = []

    def resolve(self, requested: Path, *, must_exist: bool) -> Path:
        if requested.is_absolute() or ".." in PurePath(requested).parts:
            raise ToolError("Paths must stay within the workspace root.")
        cursor = self.root
        for part in requested.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ToolError("Editing through symbolic links is not allowed.")
        candidate = self.root / requested
        try:
            if candidate.exists() or candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(self.root):
                    raise ToolError("Path resolves outside the workspace root.")
                return resolved
            parent_candidate = candidate.parent
            missing_parents = []
            while not parent_candidate.exists() and parent_candidate != self.root:
                missing_parents.append(parent_candidate.name)
                parent_candidate = parent_candidate.parent
            parent = parent_candidate.resolve(strict=True)
        except ToolError:
            raise
        except (OSError, RuntimeError) as exc:
            raise ToolError(f"Path cannot be accessed: {requested}") from exc
        if must_exist:
            raise ToolError(f"Path does not exist: {requested}")
        if not parent.is_relative_to(self.root):
            raise ToolError("Path resolves outside the workspace root.")
        for part in reversed(missing_parents):
            parent /= part
        return parent / candidate.name

    def relative(self, path: Path) -> Path:
        return path.relative_to(self.root)

    def begin_turn(self) -> None:
        if self._active is not None:
            self.end_turn()
        self._active = _Turn()

    def _turn(self) -> _Turn:
        if self._active is None:
            self.begin_turn()
        assert self._active is not None
        return self._active

    def snapshot(self, *paths: Path) -> None:
        """Capture each path once before the current turn changes it."""
        turn = self._turn()
        turn_dir = self._storage_path / turn.id
        turn_dir.mkdir(exist_ok=True)
        for path in paths:
            relative = self.relative(path)
            if relative in turn.entries:
                continue
            if _has_symlink(path):
                raise ToolError("Directories containing symbolic links cannot be changed.")
            if not path.exists():
                entry = _SnapshotEntry(relative, "absent", None)
            else:
                backup = turn_dir / f"{len(turn.entries):04d}"
                if path.is_file():
                    shutil.copy2(path, backup)
                    entry = _SnapshotEntry(relative, "file", backup)
                elif path.is_dir():
                    shutil.copytree(path, backup)
                    entry = _SnapshotEntry(relative, "directory", backup)
                else:
                    raise ToolError(f"Unsupported filesystem entry: {relative}")
            turn.entries[relative] = entry

    def require_file_read(self, path: Path) -> None:
        self._turn().pending.add(("file", self.relative(path)))

    def require_directory_check(self, path: Path) -> None:
        self._turn().pending.add(("directory", self.relative(path)))

    def observe(self, tool: str, arguments: dict[str, object], output: str) -> None:
        """Clear verification requirements satisfied by a successful read tool."""
        if self._active is None:
            return
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str):
            return
        try:
            path = self.resolve(Path(raw_path), must_exist=True)
        except ToolError:
            return
        relative = self.relative(path)
        if tool == "read_file":
            self._active.pending.discard(("file", relative))
        elif tool == "list_directory":
            self._active.pending.discard(("directory", relative))
        elif tool == "search_files":
            for kind, pending in tuple(self._active.pending):
                if kind == "file" and f"{pending.as_posix()}:" in output:
                    self._active.pending.discard((kind, pending))

    def verification_message(self) -> str | None:
        if self._active is None or not self._active.pending:
            return None
        checks = []
        for kind, path in sorted(
            self._active.pending, key=lambda item: (item[1].as_posix(), item[0])
        ):
            if kind == "file":
                checks.append(f'read_file with path="{path.as_posix()}"')
            else:
                display = path.as_posix() or "."
                checks.append(f'list_directory with path="{display}"')
        return (
            "You changed the workspace but have not verified the final state. "
            "Use the available read tools to complete these checks before finishing: "
            + "; ".join(checks)
            + "."
        )

    def _entry_diff(self, entry: _SnapshotEntry, seen: set[Path]) -> list[str]:
        current = self.root / entry.path
        differences = []
        if current.is_symlink():
            seen.add(entry.path)
            return [f"Symbolic link changed: {entry.path.as_posix()}"]
        if entry.kind in {"file", "absent"} and not current.is_dir():
            if entry.path in seen:
                return []
            before = entry.backup.read_bytes() if entry.kind == "file" else None
            after = current.read_bytes() if current.is_file() else None
            if before != after:
                differences.append(_file_diff(entry.path.as_posix(), before, after))
            seen.add(entry.path)
            return differences

        before_files: dict[Path, bytes] = {}
        if entry.kind == "directory" and entry.backup is not None:
            for file in entry.backup.rglob("*"):
                if file.is_file():
                    before_files[entry.path / file.relative_to(entry.backup)] = file.read_bytes()
        after_files: dict[Path, bytes] = {}
        if current.is_dir():
            for file in current.rglob("*"):
                if file.is_file() and not file.is_symlink():
                    after_files[entry.path / file.relative_to(current)] = file.read_bytes()
        for path in sorted(
            before_files.keys() | after_files.keys(), key=lambda item: item.as_posix()
        ):
            if path in seen or before_files.get(path) == after_files.get(path):
                continue
            differences.append(
                _file_diff(path.as_posix(), before_files.get(path), after_files.get(path))
            )
            seen.add(path)
        if not differences and entry.kind != ("directory" if current.is_dir() else "absent"):
            suffix = "/" if current.is_dir() or entry.kind == "directory" else ""
            sign = "+" if entry.kind == "absent" else "-"
            differences.append(f"{sign}{entry.path.as_posix()}{suffix}")
        return differences

    def _diff_turn(self, turn: _Turn) -> str:
        seen: set[Path] = set()
        chunks = []
        for entry in turn.entries.values():
            chunks.extend(chunk for chunk in self._entry_diff(entry, seen) if chunk)
        return "\n\n".join(chunks) if chunks else "No changes in this turn."

    def diff(self) -> str:
        turn = (
            self._active
            if self._active is not None
            else self._history[-1]
            if self._history
            else None
        )
        return self._diff_turn(turn) if turn is not None else "No changes to show."

    def end_turn(self) -> None:
        if self._active is None:
            return
        turn = self._active
        self._active = None
        if self._diff_turn(turn) != "No changes in this turn.":
            self._history.append(turn)
        else:
            shutil.rmtree(self._storage_path / turn.id, ignore_errors=True)

    def undo(self) -> str:
        if self._active is not None:
            self.end_turn()
        if not self._history:
            return "Nothing to undo."
        turn = self._history[-1]
        entries = list(turn.entries.values())
        try:
            for entry in sorted(entries, key=lambda item: len(item.path.parts), reverse=True):
                _remove(self.root / entry.path)
            for entry in sorted(entries, key=lambda item: len(item.path.parts)):
                destination = self.root / entry.path
                if entry.kind == "absent":
                    _remove(destination)
                elif entry.kind == "file" and entry.backup is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry.backup, destination)
                elif entry.kind == "directory" and entry.backup is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(entry.backup, destination, dirs_exist_ok=True)
        except OSError as exc:
            raise ToolError("The most recent turn could not be fully restored.") from exc
        self._history.pop()
        shutil.rmtree(self._storage_path / turn.id, ignore_errors=True)
        return "Undid the most recent turn."

    def handle_command(self, command: str) -> str | None:
        if command == "/diff":
            return self.diff()
        if command == "/undo":
            return self.undo()
        return None
