"""CLI commands for fixed model evaluations."""

import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assistant.config import ConfigurationError, load_config
from assistant.evaluation import EvalReport, EvaluationError, RunResult, load_cases, run_evaluation
from assistant.llm.llamacpp import LlamaCppBackend

app = typer.Typer(help="Evaluate the local model with fixed prompts.", no_args_is_help=True)


def print_summary(console: Console, report: EvalReport, path: Path) -> None:
    summary = report.summary()
    models = summary["models"]
    console.print(
        ", ".join(models) if models else "Model not reported", markup=False, highlight=False
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Value")
    completed = summary["completed_runs"]
    table.add_row("Cases", str(summary["cases"]))
    table.add_row("Runs/case", str(summary["runs_per_case"]))
    timeout = summary["request_timeout_seconds"]
    table.add_row("Read timeout", f"{timeout:g}s" if timeout is not None else "Not reported")
    table.add_row("Completed", f"{completed}/{summary['planned_runs']}")
    table.add_row("Passes", f"{summary['passes']}/{completed}")
    table.add_row("Failures", str(summary["failures"]))
    for label, key, successes, attempts in (
        (
            "JSON validity",
            "parse_success_rate",
            summary["parse_successes"],
            summary["json_attempts"],
        ),
        (
            "Protocol parse success",
            "protocol_parse_success_rate",
            summary["protocol_parse_successes"],
            summary["protocol_attempts"],
        ),
        (
            "Instruction following",
            "instruction_following_rate",
            summary["instruction_following_passes"],
            completed,
        ),
    ):
        if attempts == 0 and label == "Protocol parse success":
            continue
        rate = summary[key]
        table.add_row(label, f"{rate:.1%} ({successes}/{attempts})" if rate is not None else "N/A")
    latency = summary["median_latency_seconds"]
    table.add_row("Median latency", f"{latency:.2f}s" if latency is not None else "N/A")
    tokens = summary["output_tokens"]
    table.add_row(
        "Output tokens",
        f"{tokens:,}" if tokens is not None else "Unknown",
    )
    table.add_row("Token counts reported", f"{summary['token_usage_reported_runs']}/{completed}")
    table.add_row("Total duration", f"{summary['total_duration_seconds']:.2f}s")
    outcome = "PASS" if summary["success"] else "FAIL"
    table.add_row("Result", "INTERRUPTED" if report.status == "interrupted" else outcome)
    console.print(table)
    console.print(f"Report: {path}", markup=False, highlight=False)


@app.command("model")
def evaluate_model(
    runs: Annotated[int, typer.Option(min=1, help="Number of runs per case.")] = 5,
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for timestamped JSON reports.")
    ] = Path("evals/results"),
    temperature: Annotated[float, typer.Option(min=0.0, help="Sampling temperature.")] = 0.6,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Maximum number of generated tokens per request.")
    ] = 1024,
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Measure parse success, instruction following, latency, and token usage."""
    if not math.isfinite(temperature):
        raise typer.BadParameter("Temperature must be finite.", param_hint="--temperature")
    try:
        settings = load_config(config)
        loaded = load_cases(cases)
        console = Console(no_color=not settings.color)
        progress = Console(stderr=True, no_color=not settings.color)
        total = len(loaded) * runs

        def on_result(result: RunResult, completed: int) -> None:
            outcome = "PASS" if result.success else f"FAIL ({result.failure_reason})"
            progress.print(
                f"[{completed}/{total}] {result.case_id} run {result.run}: "
                f"{outcome}, {result.latency_seconds:.2f}s",
                markup=False,
                highlight=False,
            )

        progress.print(f"Running {len(loaded)} fixed cases, {runs} runs per case.")
        backend = LlamaCppBackend()
        report, path = run_evaluation(
            backend,
            loaded,
            runs=runs,
            temperature=temperature,
            max_tokens=max_tokens,
            output_dir=output_dir,
            request_timeout_seconds=backend.read_timeout_seconds,
            on_result=on_result,
        )
    except (ConfigurationError, EvaluationError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    print_summary(console, report, path)
    if report.status == "interrupted":
        raise typer.Exit(code=130)
    if not report.summary()["success"]:
        raise typer.Exit(code=1)
