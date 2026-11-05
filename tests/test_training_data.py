import json

from assistant.agent.protocol import parse_action
from assistant.training.dataset import build
from assistant.training.harness import Plan, TrajectoryError, run_plan
from assistant.training.worlds import build_world


def test_builds_valid_chat_examples_from_real_tool_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(tmp_path / "jamie" / "memory.sqlite3"))
    report = build(tmp_path / "data", worlds=1)
    assert report["problems"] == {}
    rows = [
        json.loads(line)
        for name in ("train", "valid")
        for line in (tmp_path / "data" / f"{name}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == report["train"] + report["valid"] > 100
    agent = [r for r in rows if r["messages"][0]["content"].startswith("Available tools")]
    assert agent
    for row in agent:
        messages = row["messages"]
        assert messages[-1]["role"] == "assistant"
        parse_action(messages[-1]["content"])  # every target is a valid action
        for earlier in messages[1:-1]:
            if earlier["role"] == "assistant":
                parse_action(earlier["content"])
    # Routing targets are a single option letter, as the classifier reads them.
    letters = [r["messages"][-1]["content"] for r in rows if "Answer with the letter" in
               r["messages"][-1 - 1]["content"]]  # fmt: skip
    assert letters and all(len(letter) == 1 for letter in letters)


def test_a_failing_step_is_rejected_instead_of_taught(tmp_path):
    import random

    import pytest

    world = build_world(random.Random(1), tmp_path)
    plan = Plan(("files",), "read it", [("mac_read_file", {"path": "~/nope.txt"})], "It says x")
    with pytest.raises(TrajectoryError):
        run_plan(world, plan)
