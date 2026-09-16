from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from assistant import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)


def test_version_matches_installed_package_and_does_not_load_config(tmp_path: Path):
    (tmp_path / "assistant.toml").write_text("invalid TOML", encoding="utf-8")

    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"assistant {version('j-cli')}"


def test_help_exposes_doctor_and_version():
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "--version" in result.output


def test_doctor_checks_only_python_and_configuration(tmp_path: Path):
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Python version" in result.output
    assert "Project configuration" in result.output
    assert result.output.count("PASS") == 2
    assert "FAIL" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_doctor_accepts_explicit_config(tmp_path: Path):
    path = tmp_path / "custom.toml"
    path.write_text("color = false\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor", "--config", str(path)])

    assert result.exit_code == 0
    assert result.output.count("PASS") == 2
    assert "\x1b[" not in result.output


@pytest.mark.parametrize("content", ["color = [", 'color = "false"', "unknown = true"])
def test_doctor_reports_invalid_default_config(tmp_path: Path, content: str):
    (tmp_path / "assistant.toml").write_text(content, encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "Project configuration" in result.output
    assert "FAIL" in result.output
    assert "Traceback" not in result.output


def test_doctor_reports_missing_explicit_config():
    result = runner.invoke(cli.app, ["doctor", "--config", "missing.toml"])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert "FAIL" in result.output


@pytest.mark.parametrize(
    ("python_version", "expected_exit"),
    [((3, 11, 9), 1), ((3, 12, 0), 0), ((3, 13, 2), 0)],
)
def test_doctor_checks_python_version(
    monkeypatch: pytest.MonkeyPatch, python_version: tuple[int, int, int], expected_exit: int
):
    monkeypatch.setattr(cli, "sys", SimpleNamespace(version_info=python_version))

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == expected_exit
    assert ".".join(map(str, python_version)) in result.output
    assert "requires 3.12+" in result.output
    assert ("FAIL" in result.output) == (expected_exit == 1)
