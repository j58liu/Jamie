"""Bottom status line; built from in-memory state and one small file read per render."""

from __future__ import annotations

from pathlib import Path

from assistant.ui.commands import ReplState

SEPARATOR = " │ "


def git_branch(start: Path) -> str | None:
    """Read the branch from .git/HEAD without running git; None when unknown."""
    for directory in (start, *start.parents):
        git = directory / ".git"
        if not git.exists():
            continue
        try:
            if git.is_file():  # worktree or submodule: "gitdir: <path>"
                git = directory / git.read_text().removeprefix("gitdir:").strip()
            head = (git / "HEAD").read_text().strip()
        except OSError:
            return None
        if head.startswith("ref: refs/heads/"):
            return head.removeprefix("ref: refs/heads/")
        return head[:7] or None  # detached HEAD
    return None


def short_model(model: str) -> str:
    return model.rsplit("/", 1)[-1].split(":", 1)[0].removesuffix("-GGUF")


def _count(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    return f"{tokens / 1000:.1f}k" if tokens < 10_000 else f"{tokens // 1000}k"


def _home(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}".removesuffix("/.")
    except ValueError:
        return str(path)


def status_line(state: ReplState) -> str:
    """Never raises: a broken field is dropped rather than breaking the prompt."""
    fields = []
    try:
        if state.model:
            fields.append(short_model(state.model))
        usage = state.context
        if usage is None:
            fields.append(f"ctx ~{_count(state.estimated_tokens())}")
        else:
            fields.append(f"ctx {_count(usage.used)}/{_count(usage.maximum)}")
        if state.thinking is not None:
            fields.append(f"thinking {'on' if state.thinking else 'off'}")
        cwd = Path.cwd()
        fields.append(_home(cwd))
        branch = git_branch(cwd)
        if branch:
            fields.append(branch)
    except Exception:  # noqa: BLE001 - the prompt must keep working
        pass
    return f" {SEPARATOR.join(fields)} "
