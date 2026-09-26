import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "start-local-model.sh"


@pytest.fixture
def launcher_environment(tmp_path: Path) -> dict[str, str]:
    binary = tmp_path / "llama-server"
    binary.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@"\nexit "${FAKE_SERVER_EXIT:-0}"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return {**os.environ, "PATH": str(tmp_path)}


def run_launcher(environment: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", str(LAUNCHER), *args],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_launch_uses_cached_model_offline_and_local_gpu(launcher_environment):
    result = run_launcher(launcher_environment)

    assert result.returncode == 0
    args = result.stdout.rstrip("\0").split("\0")
    assert args[:2] == ["-hf", "unsloth/Qwen3.5-2B-GGUF:Q4_K_M"]
    assert "--offline" in args
    assert "--no-mmproj" in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8080"
    assert args[args.index("--gpu-layers") + 1] == "all"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--ctx-size") + 1] == "8192"
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--cache-ram") + 1] == "512"
    assert args[args.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    assert "--perf" in args
    assert "first-download steps in README.md" in result.stderr


def test_local_model_path_preserves_spaces_without_hugging_face(launcher_environment, tmp_path):
    model = tmp_path / "local model.gguf"
    model.write_bytes(b"GGUF")

    result = run_launcher(launcher_environment, str(model))

    assert result.returncode == 0
    args = result.stdout.rstrip("\0").split("\0")
    assert args[:2] == ["--model", str(model)]
    assert "-hf" not in args
    assert "--offline" in args


@pytest.mark.parametrize("model_kind", ["missing", "directory"])
def test_invalid_model_path_has_useful_error(launcher_environment, tmp_path, model_kind):
    model = tmp_path / "invalid.gguf"
    if model_kind == "directory":
        model.mkdir()

    result = run_launcher(launcher_environment, str(model))

    assert result.returncode == 1
    assert result.stdout == ""
    assert "model file does not exist or is not readable" in result.stderr
    assert str(model) in result.stderr


def test_missing_server_has_installation_hint(tmp_path):
    result = run_launcher({**os.environ, "PATH": str(tmp_path)})

    assert result.returncode == 1
    assert "llama-server is not installed or is not on PATH" in result.stderr
    assert "brew install llama.cpp" in result.stderr


@pytest.mark.parametrize("help_flag", ["--help", "-h"])
def test_help_does_not_require_server_or_model(tmp_path, help_flag):
    result = run_launcher({**os.environ, "PATH": str(tmp_path)}, help_flag)

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "does not download model files" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("args", [("--host",), ("one.gguf", "two.gguf")])
def test_unsupported_arguments_show_usage(launcher_environment, args):
    result = run_launcher(launcher_environment, *args)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr


def test_server_failure_preserves_exit_status_and_cache_hint(launcher_environment):
    result = run_launcher({**launcher_environment, "FAKE_SERVER_EXIT": "7"})

    assert result.returncode == 7
    assert "If the cache is missing" in result.stderr
