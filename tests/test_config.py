from pathlib import Path

import pytest

from assistant.config import ConfigurationError, load_config


def test_missing_default_config_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    assert load_config().color is True
    assert list(tmp_path.iterdir()) == []


def test_default_config_is_loaded_from_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assistant.toml").write_text("color = false\n", encoding="utf-8")

    assert load_config().color is False


def test_memory_path_is_validated_without_creating_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assistant.toml").write_text(
        'color = true\nmemory_path = "state/memory.sqlite3"\n', encoding="utf-8"
    )
    settings = load_config()
    assert settings.resolved_memory_path() == Path("state/memory.sqlite3")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["assistant.toml"]


def test_explicit_config_overrides_working_directory_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assistant.toml").write_text("invalid TOML", encoding="utf-8")
    explicit_config = tmp_path / "custom.toml"
    explicit_config.write_text("color = false\n", encoding="utf-8")

    assert load_config(explicit_config).color is False


def test_explicit_missing_config_fails(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "content",
    [
        "color = [",
        'color = "false"',
        "color = 1",
        "unknown = true",
        "[output]\ncolor = true",
        'memory_path = ""',
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str):
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_non_utf8_config_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.toml"
    path.write_bytes(b"\xff")

    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_config(path)


def test_directory_is_not_a_config_file(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_config(tmp_path)
