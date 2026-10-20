"""Compare cached models/settings on fixed tasks; own and clean up only the test server."""

import argparse
import json
import math
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from assistant.benchmark import capture_environment, run_benchmark
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.tools.scopes import ToolScope


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    model_repo: str = Field(min_length=1)
    context_size: int = Field(ge=2048, le=131072)
    max_tokens: int = Field(ge=1)
    cache_ram_mib: int = Field(default=512, ge=0)

    def arguments(self, port: int) -> list[str]:
        return [
            "llama-server",
            "-hf",
            self.model_repo,
            "--offline",
            "--no-mmproj",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-layers",
            "all",
            "--flash-attn",
            "on",
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--cache-ram",
            str(self.cache_ram_mib),
            "--jinja",
            "--reasoning-format",
            "deepseek",
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--temp",
            "0.6",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--perf",
        ]


@contextmanager
def test_server(arguments: list[str], port: int, log_path: Path):
    """Never stop or reconfigure an existing model server."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    with log_path.open("w") as log:
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            with httpx.Client(trust_env=False, timeout=1) as client:
                deadline = time.monotonic() + 90
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Test server exited; see {log_path}")
                    try:
                        response = client.get(f"http://127.0.0.1:{port}/health")
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Test server did not become ready; see {log_path}")
                    time.sleep(0.2)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("evals/inference_profiles.json"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results/inference"))
    options = parser.parse_args()
    if (
        options.runs < 1
        or not 1024 <= options.port <= 65535
        or not math.isfinite(options.timeout)
        or options.timeout <= 0
    ):
        parser.error("Use positive runs/timeout and a port between 1024 and 65535.")
    profiles = [Profile.model_validate(item) for item in json.loads(options.profiles.read_text())]
    if not profiles or len({profile.id for profile in profiles}) != len(profiles):
        parser.error("Profiles must be nonempty and have unique IDs.")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    with tempfile.TemporaryDirectory(prefix="jamie-model-comparison-") as directory:
        for profile in profiles:
            print(f"Starting {profile.id}", flush=True)
            arguments = profile.arguments(options.port)
            with test_server(arguments, options.port, Path(directory) / f"{profile.id}.log"):
                with LlamaCppBackend(f"http://127.0.0.1:{options.port}") as backend:
                    environment = capture_environment(
                        backend, {"max_tokens": profile.max_tokens, "temperature": 0.6}
                    )
                    environment["launch_arguments"] = arguments
                    environment["profile"] = profile.model_dump()
                    report, path = run_benchmark(
                        backend,
                        runs=options.runs,
                        max_tokens=profile.max_tokens,
                        timeout_seconds=options.timeout,
                        scope=ToolScope.FILES,
                        output_dir=options.output_dir / profile.id,
                        environment=environment,
                        on_result=lambda result, count: print(
                            f"{count}: {result.case_id}: {result.success} "
                            f"({result.latency_seconds:.2f}s)",
                            flush=True,
                        ),
                    )
                    comparison.append(
                        {
                            "profile": profile.model_dump(),
                            "report": str(path),
                            "summary": report.summary(),
                        }
                    )
                    (options.output_dir / "comparison.json").write_text(
                        json.dumps(comparison, indent=2) + "\n"
                    )
                    print(json.dumps(comparison[-1]["summary"]), flush=True)
                    if report.status == "interrupted":
                        raise SystemExit(130)


if __name__ == "__main__":
    main()
