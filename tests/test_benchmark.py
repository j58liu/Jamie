import json
from pathlib import Path

from assistant.benchmark import run_benchmark
from assistant.llm.base import LLMResponse


class Model:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        assert options == {"temperature": 0.2, "max_tokens": 128}
        return LLMResponse("SQLite", "test-model", "stop", 2, input_tokens=12)


def test_benchmark_repeats_fixed_tasks_and_records_metrics_and_environment(tmp_path: Path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "pass",
                    "steps": [{"prompt": "Which database?", "answer_contains": ["SQLite"]}],
                },
                {
                    "id": "fail",
                    "steps": [{"prompt": "Which database?", "answer_contains": ["Postgres"]}],
                },
            ]
        )
    )
    model = Model()
    report, path = run_benchmark(
        model,
        cases=cases,
        runs=3,
        max_tokens=128,
        temperature=0.2,
        environment={"model_sha256": "abc", "hardware": {"chip": "test"}},
        output_dir=tmp_path,
    )
    saved = json.loads(path.read_text())
    summary = report.summary()
    assert summary["passes"] == summary["failures"] == 3
    assert summary["model_calls"] == model.calls == 6
    assert summary["input_tokens"] == 72
    assert summary["input_usage_reported_turns"] == 6
    assert summary["first_visible_reported_turns"] == 6
    assert summary["p95_latency_seconds"] >= summary["median_latency_seconds"] > 0
    assert summary["parse_retries"] == summary["approval_prompts"] == 0
    assert len(saved["environment"]["cases_sha256"]) == 64
    assert saved["environment"]["hardware"]["chip"] == "test"
    assert path.name.startswith("benchmark-")
