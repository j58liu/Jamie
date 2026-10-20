"""Reproducible everyday-task benchmarks using the shared runtime and fixed fixtures."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import httpx
from pydantic import JsonValue

from assistant.evaluation import load_case_files
from assistant.interactive_evaluation import InteractiveEvalCase, run_interactive_evaluation
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.tools.scopes import ToolScope


def _command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout + result.stderr).strip() if result.returncode == 0 else None


def capture_environment(backend: LlamaCppBackend, generation: dict) -> dict[str, JsonValue]:
    """Record model identity, actual server properties, code, and local hardware."""
    info = backend.server_info()
    model = Path(info.model_path) if info.model_path else None
    digest = None
    if model is not None and model.is_file():
        with model.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    with httpx.Client(trust_env=False, timeout=5) as client:
        props = client.get(f"{backend.server_url}/props")
        props.raise_for_status()
        server_settings = props.json()
    binary = shutil.which("llama-server")
    return {
        "model": {**asdict(info), "file_sha256": digest},
        "server_settings": server_settings,
        "server_build": server_settings.get("build_info"),
        "local_server_binary_version": _command([binary, "--version"]) if binary else None,
        "server_version_note": "Local binary version; verify it matches the running server.",
        "generation": generation,
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "chip": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": _command(["sysctl", "-n", "hw.memsize"]),
        },
        "python_version": platform.python_version(),
        "git_revision": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "tracked_diff_sha256": hashlib.sha256(
            (_command(["git", "diff", "HEAD"]) or "").encode()
        ).hexdigest(),
        "timing": (
            "End-to-end task latency includes setup and failed attempts. "
            "First visible text is measured at the event sink, not the first reasoning token."
        ),
        "cache": (
            "Server remains warm between trials; each task gets a fresh workspace and database."
        ),
    }


class _BufferedChatBackend:
    """Measure buffered presentation with the same inference and context behavior."""

    def __init__(self, backend):
        self.backend = backend

    def chat(self, messages, **options):
        return self.backend.chat(messages, **options)

    def __getattr__(self, name):
        if name == "stream_chat":
            raise AttributeError(name)
        return getattr(self.backend, name)


def run_benchmark(
    backend,
    *,
    runs: int = 3,
    cases: Path | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    timeout_seconds: float = 300,
    output_dir: Path = Path("evals/results"),
    on_result=None,
    environment: dict[str, JsonValue] | None = None,
    scope: ToolScope = ToolScope.ALL,
    visible_streaming: bool = True,
):
    loaded = load_case_files(cases, InteractiveEvalCase, "benchmark_cases", "benchmark")
    loaded = [
        case.model_copy(update={"all_tools": True, "tool_scope": scope.value}) for case in loaded
    ]
    options = {"max_tokens": max_tokens, "temperature": temperature}
    metadata = capture_environment(backend, options) if environment is None else dict(environment)
    metadata["tool_scope"] = scope.value
    metadata["visible_streaming"] = visible_streaming
    metadata["task_cases_sha256"] = hashlib.sha256(
        json.dumps(
            [case.model_dump(exclude={"all_tools", "tool_scope"}) for case in loaded],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    metadata["cases_sha256"] = hashlib.sha256(
        json.dumps([case.model_dump() for case in loaded], sort_keys=True).encode()
    ).hexdigest()
    return run_interactive_evaluation(
        backend if visible_streaming else _BufferedChatBackend(backend),
        loaded,
        runs=runs,
        timeout_seconds=timeout_seconds,
        output_dir=output_dir,
        on_result=on_result,
        environment=metadata,
        report_prefix="benchmark",
        max_tokens=max_tokens,
        temperature=temperature,
    )
