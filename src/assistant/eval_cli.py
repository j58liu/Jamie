"""CLI commands for fixed model evaluations."""

import json
import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from assistant.agent_evaluation import (
    AgentEvalReport,
    AgentRunResult,
    load_agent_cases,
    run_agent_evaluation,
)
from assistant.config import ConfigurationError, load_config
from assistant.evaluation import EvalReport, EvaluationError, RunResult, load_cases, run_evaluation
from assistant.interactive_evaluation import (
    InteractiveEvalReport,
    InteractiveRunResult,
    load_interactive_cases,
    run_interactive_evaluation,
)
from assistant.llm.base import LLMError, managed_backend
from assistant.llm.llamacpp import LlamaCppBackend
from assistant.memory_evaluation import (
    MemoryEvalReport,
    load_memory_cases,
    run_memory_evaluation,
)
from assistant.research_evaluation import (
    ResearchEvalReport,
    ResearchRunResult,
    load_research_cases,
    run_research_evaluation,
)
from assistant.tools.scopes import ToolScope

app = typer.Typer(help="Evaluate the local model with fixed prompts.", no_args_is_help=True)


@app.command("benchmark")
def benchmark(
    runs: Annotated[int, typer.Option(min=1)] = 3,
    cases: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path, typer.Option()] = Path("evals/results"),
    max_tokens: Annotated[int, typer.Option(min=1)] = 1024,
    temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.6,
    timeout: Annotated[float, typer.Option(min=0.001)] = 300,
    scope: Annotated[ToolScope, typer.Option()] = ToolScope.ALL,
    server: Annotated[str, typer.Option(help="llama.cpp server URL.")] = "http://127.0.0.1:8080",
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Display direct-chat fragments as they arrive."),
    ] = True,
) -> None:
    """Measure everyday task success, latency, model calls, retries, and approvals."""
    from assistant.benchmark import run_benchmark

    if not math.isfinite(timeout) or not math.isfinite(temperature):
        raise typer.BadParameter("Generation settings and timeout must be finite.")
    console = Console()
    progress = Console(stderr=True)

    def completed(result, count):
        outcome = "PASS" if result.success else f"FAIL ({result.failure_reason})"
        progress.print(
            f"[{count}] {result.case_id} run {result.run}: {outcome}, {result.latency_seconds:.2f}s"
        )

    backend = LlamaCppBackend(server)
    try:
        report, path = run_benchmark(
            backend,
            runs=runs,
            cases=cases,
            output_dir=output_dir,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
            on_result=completed,
            scope=scope,
            visible_streaming=stream,
        )
    except (EvaluationError, OSError, ValueError, RuntimeError) as exc:
        console.print(f"Error: {exc}", markup=False)
        raise typer.Exit(1) from exc
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    summary = report.summary()
    console.print(json.dumps(summary, indent=2), markup=False)
    console.print(f"Report: {path}", markup=False)
    if report.status == "interrupted":
        raise typer.Exit(130)
    if not summary["success"]:
        raise typer.Exit(1)


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


def print_agent_summary(console: Console, report: AgentEvalReport, path: Path) -> None:
    """Print the end-to-end agent evaluation metrics."""
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
    table.add_row("Completed", f"{completed}/{summary['planned_runs']}")
    table.add_row("Passes", f"{summary['passes']}/{completed}")
    table.add_row("Expected tool called", f"{summary['expected_tool_calls']}/{completed}")
    table.add_row("Correct tool output", f"{summary['correct_tool_outputs']}/{completed}")
    table.add_row("Answer check", f"{summary['answer_checks']}/{completed}")
    latency = summary["median_latency_seconds"]
    table.add_row("Median latency", f"{latency:.2f}s" if latency is not None else "N/A")
    tokens = summary["output_tokens"]
    table.add_row("Output tokens", f"{tokens:,}" if tokens is not None else "Unknown")
    table.add_row("Total duration", f"{summary['total_duration_seconds']:.2f}s")
    outcome = "PASS" if summary["success"] else "FAIL"
    table.add_row("Result", "INTERRUPTED" if report.status == "interrupted" else outcome)
    console.print(table)
    console.print(f"Report: {path}", markup=False, highlight=False)


def print_interactive_summary(console: Console, report: InteractiveEvalReport, path: Path) -> None:
    """Print pass rates for the multi-turn interactive scenarios."""
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
    table.add_row("Completed", f"{completed}/{summary['planned_runs']}")
    table.add_row("Passes", f"{summary['passes']}/{completed}")
    table.add_row("Failed cases", ", ".join(summary["failed_cases"]) or "None")
    latency = summary["median_latency_seconds"]
    table.add_row("Median case latency", f"{latency:.2f}s" if latency is not None else "N/A")
    tokens = summary["output_tokens"]
    table.add_row("Output tokens", f"{tokens:,}" if tokens is not None else "Unknown")
    table.add_row("Total duration", f"{summary['total_duration_seconds']:.2f}s")
    outcome = "PASS" if summary["success"] else "FAIL"
    table.add_row("Result", "INTERRUPTED" if report.status == "interrupted" else outcome)
    console.print(table)
    console.print(f"Report: {path}", markup=False, highlight=False)


def print_research_summary(console: Console, report: ResearchEvalReport, path: Path) -> None:
    """Print citation and source-support metrics for research evaluations."""
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
    table.add_row("Completed", f"{completed}/{summary['planned_runs']}")
    table.add_row("Passes", f"{summary['passes']}/{completed}")
    source_count = summary["median_source_count"]
    table.add_row("Median source count", f"{source_count:g}" if source_count is not None else "N/A")
    table.add_row("Citations exist", f"{summary['citations_exist']}/{completed}")
    table.add_row("Citations resolve", f"{summary['citations_resolve']}/{completed}")
    table.add_row("Claims supported", f"{summary['claims_supported']}/{completed}")
    table.add_row("No fake URLs", f"{summary['no_fake_urls']}/{completed}")
    table.add_row("Multiple-source coverage", f"{summary['multiple_source_coverage']}/{completed}")
    latency = summary["median_latency_seconds"]
    table.add_row("Median latency", f"{latency:.2f}s" if latency is not None else "N/A")
    tokens = summary["output_tokens"]
    table.add_row("Output tokens", f"{tokens:,}" if tokens is not None else "Unknown")
    table.add_row("Total duration", f"{summary['total_duration_seconds']:.2f}s")
    outcome = "PASS" if summary["success"] else "FAIL"
    table.add_row("Result", "INTERRUPTED" if report.status == "interrupted" else outcome)
    console.print(table)
    console.print(f"Report: {path}", markup=False, highlight=False)


def print_memory_summary(console: Console, report: MemoryEvalReport, path: Path) -> None:
    """Print deterministic FTS5 retrieval metrics."""
    summary = report.summary()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Cases", str(summary["cases"]))
    table.add_row("Hits", f"{summary['hits']}/{summary['cases']}")
    table.add_row("Misses", str(summary["misses"]))
    recall = summary["recall_at_k"]
    reciprocal_rank = summary["mean_reciprocal_rank"]
    table.add_row("Recall@k", f"{recall:.1%}" if recall is not None else "N/A")
    table.add_row(
        "Mean reciprocal rank",
        f"{reciprocal_rank:.3f}" if reciprocal_rank is not None else "N/A",
    )
    table.add_row("Total duration", f"{summary['total_duration_seconds']:.2f}s")
    table.add_row("Result", "PASS" if summary["success"] else "FAIL")
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
        with managed_backend(LlamaCppBackend()) as backend:
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


@app.command("agent")
def evaluate_agent(
    runs: Annotated[int, typer.Option(min=1, help="Number of runs per case.")] = 5,
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for timestamped JSON reports.")
    ] = Path("evals/results"),
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Maximum seconds for each complete agent run.")
    ] = 300.0,
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Evaluate complete model, tool, observation, and answer behavior."""
    if not math.isfinite(timeout):
        raise typer.BadParameter("Timeout must be finite.", param_hint="--timeout")
    try:
        settings = load_config(config)
        loaded = load_agent_cases(cases)
        console = Console(no_color=not settings.color)
        progress = Console(stderr=True, no_color=not settings.color)
        total = len(loaded) * runs

        def on_result(result: AgentRunResult, completed: int) -> None:
            outcome = "PASS" if result.success else f"FAIL ({result.failure_reason})"
            progress.print(
                f"[{completed}/{total}] {result.case_id} run {result.run}: "
                f"{outcome}, {result.latency_seconds:.2f}s",
                markup=False,
                highlight=False,
            )

        progress.print(f"Running {len(loaded)} agent cases, {runs} runs per case.")
        with managed_backend(LlamaCppBackend()) as backend:
            report, path = run_agent_evaluation(
                backend,
                loaded,
                runs=runs,
                timeout_seconds=timeout,
                output_dir=output_dir,
                on_result=on_result,
            )
    except (ConfigurationError, EvaluationError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    print_agent_summary(console, report, path)
    if report.status == "interrupted":
        raise typer.Exit(code=130)
    if not report.summary()["success"]:
        raise typer.Exit(code=1)


@app.command("interactive")
def evaluate_interactive(
    runs: Annotated[int, typer.Option(min=1, help="Number of runs per case.")] = 1,
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for timestamped JSON reports.")
    ] = Path("evals/results"),
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Maximum seconds for each turn.")
    ] = 300.0,
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Evaluate multi-turn interactive behavior against the local model."""
    if not math.isfinite(timeout):
        raise typer.BadParameter("Timeout must be finite.", param_hint="--timeout")
    try:
        settings = load_config(config)
        loaded = load_interactive_cases(cases)
        console = Console(no_color=not settings.color)
        progress = Console(stderr=True, no_color=not settings.color)
        total = len(loaded) * runs

        def on_result(result: InteractiveRunResult, completed: int) -> None:
            outcome = (
                "PASS"
                if result.success
                else f"FAIL at step {result.failed_step} ({result.failure_reason})"
            )
            progress.print(
                f"[{completed}/{total}] {result.case_id} run {result.run}: "
                f"{outcome}, {result.latency_seconds:.2f}s",
                markup=False,
                highlight=False,
            )

        progress.print(f"Running {len(loaded)} interactive cases, {runs} runs per case.")
        with managed_backend(LlamaCppBackend()) as backend:
            report, path = run_interactive_evaluation(
                backend,
                loaded,
                runs=runs,
                timeout_seconds=timeout,
                output_dir=output_dir,
                on_result=on_result,
            )
    except (ConfigurationError, EvaluationError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    print_interactive_summary(console, report, path)
    if report.status == "interrupted":
        raise typer.Exit(code=130)
    if not report.summary()["success"]:
        raise typer.Exit(code=1)


@app.command("research")
def evaluate_research(
    runs: Annotated[int, typer.Option(min=1, help="Number of runs per case.")] = 5,
    workers: Annotated[int, typer.Option(min=1, max=8)] = 3,
    fixture_delay: Annotated[
        float, typer.Option(min=0, max=10, help="Simulated seconds per fixture network call.")
    ] = 0,
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for timestamped JSON reports.")
    ] = Path("evals/results"),
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Evaluate source retrieval, citations, and supported claims."""
    if not math.isfinite(fixture_delay):
        raise typer.BadParameter("Fixture delay must be finite.")
    try:
        settings = load_config(config)
        loaded = load_research_cases(cases)
        console = Console(no_color=not settings.color)
        progress = Console(stderr=True, no_color=not settings.color)
        total = len(loaded) * runs

        def on_result(result: ResearchRunResult, completed: int) -> None:
            outcome = "PASS" if result.success else f"FAIL ({result.failure_reason})"
            progress.print(
                f"[{completed}/{total}] {result.case_id} run {result.run}: "
                f"{outcome}, {result.latency_seconds:.2f}s",
                markup=False,
                highlight=False,
            )

        progress.print(f"Running {len(loaded)} research cases, {runs} runs per case.")
        from assistant.benchmark import capture_environment

        with managed_backend(LlamaCppBackend()) as backend:
            environment = (
                capture_environment(backend, {"workflow": "research"})
                if hasattr(backend, "server_info")
                else {}
            )
            report, path = run_research_evaluation(
                backend,
                loaded,
                runs=runs,
                output_dir=output_dir,
                on_result=on_result,
                max_workers=workers,
                fixture_delay_seconds=fixture_delay,
                environment=environment,
            )
    except (ConfigurationError, EvaluationError, LLMError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc

    print_research_summary(console, report, path)
    if report.status == "interrupted":
        raise typer.Exit(code=130)
    if not report.summary()["success"]:
        raise typer.Exit(code=1)


@app.command("memory")
def evaluate_memory(
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the timestamped JSON report.")
    ] = Path("evals/results"),
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Evaluate SQLite FTS5 memory retrieval without calling the model."""
    try:
        settings = load_config(config)
        loaded = load_memory_cases(cases)
        console = Console(no_color=not settings.color)
        report, path = run_memory_evaluation(loaded, output_dir=output_dir)
    except (ConfigurationError, EvaluationError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    print_memory_summary(console, report, path)
    if not report.summary()["success"]:
        raise typer.Exit(code=1)


def _rate_text(value) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


@app.command("scenarios")
def evaluate_scenarios(
    runs: Annotated[int, typer.Option(min=1, help="Times to run each scenario.")] = 1,
    category: Annotated[
        str | None,
        typer.Option(help="Only one category: routine, planning, cross_app, memory, safety."),
    ] = None,
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    timeout: Annotated[float, typer.Option(min=1, help="Seconds per turn.")] = 300.0,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the timestamped JSON report.")
    ] = Path("evals/results"),
    config: Annotated[
        Path | None, typer.Option(help="Read a specific TOML configuration file.")
    ] = None,
) -> None:
    """Run planning, cross-app, memory, and safety workflows against fake apps."""
    from assistant.scenario_evaluation import TARGETS, load_scenario_cases, run_scenario_evaluation

    try:
        settings = load_config(config)
        loaded = load_scenario_cases(cases, category=category)
        console = Console(no_color=not settings.color)

        def on_result(result, completed: int) -> None:
            mark = "PASS" if result.success else "FAIL"
            detail = f"  {result.failures[0]}" if result.failures else ""
            console.print(
                f"[{completed}/{len(loaded) * runs}] {mark} {result.case_id}{detail}",
                markup=False,
                highlight=False,
            )

        with managed_backend(LlamaCppBackend()) as backend:
            report, path = run_scenario_evaluation(
                backend,
                loaded,
                runs=runs,
                output_dir=output_dir,
                timeout_seconds=timeout,
                on_result=on_result,
            )
    except (ConfigurationError, EvaluationError) as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    summary = report.summary()
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Result")
    table.add_column("Target")
    table.add_row("Passed", f"{summary['passed']}/{summary['attempts']}", "")
    for name, counts in summary["categories"].items():
        table.add_row(f"  {name}", f"{counts['passed']}/{counts['attempts']}", "")
    table.add_row(
        "Routine tool + arguments",
        _rate_text(summary["routine_accuracy"]),
        f">= {TARGETS['routine_accuracy']:.0%}",
    )
    table.add_row(
        "Workflow completion",
        _rate_text(summary["workflow_completion"]),
        f">= {TARGETS['workflow_completion']:.0%}",
    )
    table.add_row("Approval violations", str(summary["approval_violations"]), "0")
    table.add_row("Duplicate external writes", str(summary["duplicate_writes"]), "0")
    if summary["median_latency_seconds"] is not None:
        table.add_row("Median scenario time", f"{summary['median_latency_seconds']:.1f}s", "")
    console.print(table)
    console.print(f"Report: {path}", markup=False, highlight=False)
    if summary["approval_violations"] or summary["duplicate_writes"]:
        raise typer.Exit(code=1)


@app.command("personal-memory")
def evaluate_personal_memory(
    cases: Annotated[
        Path | None, typer.Option(help="A JSON case file or directory of case files.")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for the timestamped JSON report.")
    ] = Path("evals/results"),
) -> None:
    """Check memory corrections, deletion, retrieval, and the extraction gate (no model)."""
    from assistant.scenario_evaluation import load_memory_ops_cases, run_memory_ops_evaluation

    try:
        report, path = run_memory_ops_evaluation(
            load_memory_ops_cases(cases), output_dir=output_dir
        )
    except EvaluationError as exc:
        Console(stderr=True).print(f"Error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=1) from exc
    console = Console()
    for result in report.results:
        mark = "PASS" if result.success else "FAIL"
        detail = f"  {'; '.join(result.failures)}" if result.failures else ""
        console.print(f"{mark} {result.case_id}{detail}", markup=False, highlight=False)
    summary = report.summary()
    console.print(
        f"{summary['passed']}/{summary['cases']} passed · memory corrections "
        f"{_rate_text(summary['memory_corrections'])} (target 100%)",
        markup=False,
        highlight=False,
    )
    console.print(f"Report: {path}", markup=False, highlight=False)
    if not summary["success"]:
        raise typer.Exit(code=1)
