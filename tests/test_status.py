import io
from pathlib import Path

import pytest
from rich.console import Console

from assistant.llm.base import Message
from assistant.ui import status
from assistant.ui.commands import ReplState
from assistant.ui.status import git_branch, status_line


def _state(**fields) -> ReplState:
    return ReplState(Console(file=io.StringIO()), **fields)


def test_git_branch_reads_head_from_parent_directories(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feature/status\n")
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)

    assert git_branch(nested) == "feature/status"


def test_git_branch_handles_worktrees_and_detached_head(tmp_path: Path):
    gitdir = tmp_path / "main.git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("0123456789abcdef\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")

    assert git_branch(worktree) == "0123456"


def test_git_branch_is_none_outside_a_repository_or_when_unreadable(tmp_path: Path):
    assert git_branch(Path("/")) is None
    (tmp_path / ".git").mkdir()  # no HEAD file
    assert git_branch(tmp_path) is None


def test_status_line_shows_model_context_thinking_cwd_and_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "Projects" / "jamie"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(repo)
    state = _state(
        model="empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M",
        history=[Message("user", "x" * 12_800)],
        thinking=False,
    )

    assert status_line(state) == (
        " Qwen3.8-9B-Distill │ ctx ~3.2k │ thinking off │ ~/Projects/jamie │ main "
    )


def test_status_line_omits_unknown_fields_and_survives_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    assert status_line(_state()).startswith(" ctx ~0 │ ")

    def broken(path: Path) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(status, "git_branch", broken)
    assert status_line(_state(model="m")).startswith(" m │ ctx ~0 │ ")
