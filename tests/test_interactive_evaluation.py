import asyncio
import io
import json
import re
import time
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from assistant import cli, eval_cli, interactive_evaluation
from assistant.agent.events import discard_event
from assistant.agent.loop import INVALID_OUTPUT_MESSAGE, run_agent
from assistant.cancellation import current_token
from assistant.config import ProjectConfig
from assistant.evaluation import EvaluationError
from assistant.interactive_evaluation import (
    InteractiveEvalCase,
    InteractiveStep,
    load_interactive_cases,
    run_interactive_evaluation,
)
from assistant.llm.base import ContextUsage, LLMResponse, Message
from assistant.research import VERIFICATION_INSTRUCTIONS
from assistant.runtime import JamieApplication
from assistant.runtime.application import COMPACT_REQUEST
from assistant.tools.registry import create_read_only_registry
from assistant.ui.renderer import EventRenderer


class ScriptedModel:
    """A deterministic stand-in that can only answer from the messages it is sent."""

    def __init__(self, *, forget_history: bool = False) -> None:
        self.forget_history = forget_history
        self.requests: list[list[Message]] = []

    def chat(self, messages, *, temperature=0.6, max_tokens=1024, thinking=None):
        self.requests.append(list(messages))
        if '"action"' in messages[0].content:
            return LLMResponse(self._action(messages), "scripted", "stop", 5)
        return LLMResponse(self._reply(messages), "scripted", "stop", 5)

    def context_usage(self, messages):
        return ContextUsage(sum(len(m.content) for m in messages), 100_000)

    def _reply(self, messages: list[Message]) -> str:
        prompt = messages[-1].content
        system = messages[0].content if messages[0].role == "system" else ""
        if system.startswith("Create a concise web search plan"):
            return '{"queries":["boiling point of water"]}'
        if system.startswith("Cross-check"):
            return "The sources agree [S1] [S2]."
        if system == VERIFICATION_INSTRUCTIONS:
            data = json.loads(prompt)
            sources = {s["id"]: s["text"] for s in data["sources"]}
            return json.dumps(
                {
                    "claims": [
                        {
                            "index": p["index"],
                            "supported": True,
                            "evidence": [
                                {"source_id": s, "quote": sources[s][:800]} for s in p["source_ids"]
                            ],
                        }
                        for p in data["passages"]
                    ]
                }
            )
        if system.startswith("Answer from the supplied evidence"):
            sources = json.loads(prompt.split("Sources: ", 1)[1])
            return "\n".join(f"{source['text']} [{source['id']}]" for source in sources)
        if prompt == COMPACT_REQUEST:
            if self.forget_history:
                return "We talked."
            return " ".join(m.content for m in messages[:-1])
        if "Roman Empire" in prompt:  # a long generation the user cancels
            token = current_token()
            for _ in range(500):
                token.raise_if_cancelled()
                time.sleep(0.01)
            return "An essay nobody waited for."
        if word := re.search(r"Reply with the word (\w+)", prompt):
            return word.group(1)
        if "12 plus 30" in prompt:
            return "42"
        if "add 8" in prompt:
            return "50" if any("42" in m.content for m in messages[:-1]) else "unknown"
        if prompt.endswith("?") or "Reply with" in prompt:
            if self.forget_history:
                return "I do not know."
            # Answer only from the carried conversation, never from recalled memories.
            return " ".join(m.content for m in messages[:-1] if m.role != "system") or "none"
        return "Noted."

    def _action(self, messages: list[Message]) -> str:
        last = messages[-1].content
        if last.startswith("You changed the workspace"):
            return _tool("read_file", path="hello.txt")
        task = next(
            m.content
            for m in reversed(messages)
            if m.role == "user"
            and not m.content.startswith("Tool observation:")
            and m.content != INVALID_OUTPUT_MESSAGE
        )
        if last.startswith("Tool observation:"):
            observation = json.loads(last.split("\n")[1])
            if observation["tool"] == "read_file" and not observation["success"]:
                return _tool("find_files", pattern="**/plan.md")
            if observation["tool"] == "find_files":
                return _tool("read_file", path=observation["output"].splitlines()[0])
            outcome = observation["output"] if observation["success"] else "It was not done."
            return json.dumps({"action": "finish", "message": f"Result: {outcome}"})
        if last == INVALID_OUTPUT_MESSAGE or last == task:
            for name in ("notes.md", "plan.md", "README.md"):
                if name in task:
                    return _tool("read_file", path=name)
            if "hello.txt" in task:
                return _tool("write_file", path="hello.txt", content="hi")
            if task.startswith("Research "):
                return _tool("research", question=task)
            if url := re.search(r"https://\S+", task):
                return _tool("fetch_url", url=url.group())
        if last == task:  # a conversational turn: answer from the history, as in chat
            history = [m for m in messages[1:] if m.role != "system"]
            return json.dumps({"action": "finish", "message": self._reply(history)})
        return json.dumps({"action": "finish", "message": "done"})


def _tool(name: str, **args) -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args})


@pytest.fixture(autouse=True)
def quick_cancel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(interactive_evaluation, "CANCEL_AFTER_SECONDS", 0.05)


def _evaluate(cases, model, tmp_path: Path):
    return run_interactive_evaluation(
        model, cases, runs=1, timeout_seconds=30, output_dir=tmp_path / "results"
    )


def test_every_shipped_scenario_passes_against_a_model_that_behaves(tmp_path: Path):
    cases = load_interactive_cases()

    report, path = _evaluate(cases, ScriptedModel(), tmp_path)

    failures = {
        r.case_id: (r.failed_step, r.failure_reason) for r in report.results if not r.success
    }
    assert failures == {}
    assert len(cases) == 12
    assert report.summary()["success"] is True
    saved = json.loads(path.read_text())
    assert saved["summary"]["passes"] == 12
    growth = next(r for r in report.results if r.case_id == "interactive-context-growth")
    tokens = [step.context_tokens for step in growth.steps]
    assert tokens == sorted(tokens) and len(set(tokens)) == len(tokens)


def test_a_model_that_loses_history_fails_the_context_scenarios(tmp_path: Path):
    cases = [c for c in load_interactive_cases() if c.id == "interactive-multi-turn-context"]

    report, _ = _evaluate(cases, ScriptedModel(forget_history=True), tmp_path)

    assert (report.results[0].failed_step, report.results[0].failure_reason) == (
        2,
        "answer_mismatch",
    )


def test_a_summary_that_drops_facts_fails_the_compact_scenario(tmp_path: Path):
    cases = [c for c in load_interactive_cases() if c.id == "interactive-compact-keeps-facts"]

    report, _ = _evaluate(cases, ScriptedModel(forget_history=True), tmp_path)

    assert (report.results[0].failed_step, report.results[0].failure_reason) == (
        3,
        "answer_mismatch",
    )


def test_a_model_that_skips_the_research_tool_fails_the_research_scenario(tmp_path: Path):
    class NoResearch(ScriptedModel):
        def _action(self, messages):
            if messages[-1].content.startswith("Research "):
                return json.dumps({"action": "finish", "message": "Water boils at 100 °C."})
            return super()._action(messages)

    cases = [c for c in load_interactive_cases() if c.id == "interactive-research-tool"]

    report, _ = _evaluate(cases, NoResearch(), tmp_path)

    assert report.results[0].failure_reason == "expected_tool_not_called"


def test_compaction_that_fails_is_reported_as_such(tmp_path: Path):
    case = InteractiveEvalCase(
        id="empty-compact", steps=[InteractiveStep(prompt="Hello?", compact=True)]
    )

    report, _ = _evaluate([case], ScriptedModel(), tmp_path)

    step = report.results[0].steps[0]
    assert step.failure_reason == "compact_failed"
    assert step.error == "Compaction failed: There is no conversation to compact."


def test_new_conversation_really_drops_the_old_history(tmp_path: Path):
    case = InteractiveEvalCase(
        id="leak",
        steps=[
            InteractiveStep(prompt="My dog is called Biscuit."),
            InteractiveStep(
                prompt="What is my dog called?", new_conversation=True, answer_contains=["Biscuit"]
            ),
        ],
    )

    report, _ = _evaluate([case], ScriptedModel(), tmp_path)

    assert report.results[0].failure_reason == "answer_mismatch"


def test_cancelled_prompt_never_reaches_the_model_again(tmp_path: Path):
    cases = [c for c in load_interactive_cases() if c.id == "interactive-cancel-then-continue"]
    model = ScriptedModel()

    report, _ = _evaluate(cases, model, tmp_path)

    assert report.results[0].success
    assert report.results[0].steps[1].cancelled
    assert not any("Roman" in m.content for m in model.requests[-1])


def test_a_turn_that_finishes_before_ctrl_c_fails_the_cancel_step(tmp_path: Path):
    case = InteractiveEvalCase(id="fast", steps=[InteractiveStep(prompt="Say hi.", cancel=True)])

    report, _ = _evaluate([case], ScriptedModel(), tmp_path)

    assert report.results[0].failure_reason == "finished_before_cancel"


def test_approval_answers_come_from_the_case_and_the_policy_decides(tmp_path: Path):
    write = {"prompt": "Create a file named hello.txt containing the text hi.", "tools": True}
    cases = [
        InteractiveEvalCase(
            id="wrongly-approved",
            steps=[InteractiveStep(**write, approve=True, files_absent=["hello.txt"])],
        ),
        InteractiveEvalCase(
            id="read-needs-no-approval",
            files={"README.md": "hello world\n"},
            steps=[
                InteractiveStep(
                    prompt="What does README.md contain?", tools=True, approval_requested=True
                )
            ],
        ),
    ]

    report, _ = _evaluate(cases, ScriptedModel(), tmp_path)

    reasons = {r.case_id: r.failure_reason for r in report.results}
    assert reasons == {
        "wrongly-approved": "denied_change_was_made",
        "read-needs-no-approval": "approval_expectation_mismatch",
    }


def test_context_that_stops_growing_is_reported(tmp_path: Path):
    class FlatContext(ScriptedModel):
        def context_usage(self, messages):
            return ContextUsage(10, 100_000)

    case = InteractiveEvalCase(
        id="flat",
        context_grows=True,
        steps=[InteractiveStep(prompt="One."), InteractiveStep(prompt="Two.")],
    )

    report, _ = _evaluate([case], FlatContext(), tmp_path)

    assert (report.results[0].failed_step, report.results[0].failure_reason) == (
        2,
        "context_did_not_grow",
    )


def test_steps_reject_checks_that_cannot_apply():
    with pytest.raises(ValueError, match="need tools"):
        InteractiveStep(prompt="hi", approve=True)
    with pytest.raises(ValueError, match="Cancelled steps"):
        InteractiveStep(prompt="hi", cancel=True, answer_contains=["x"])
    with pytest.raises(EvaluationError):
        load_interactive_cases(Path("missing.json"))


def test_rendering_events_does_not_change_what_the_model_sees(tmp_path: Path):
    def chat_requests(sink) -> list[list[Message]]:
        model = ScriptedModel()
        config = ProjectConfig(memory_path=str(tmp_path / f"{id(sink)}.sqlite3"))
        application = JamieApplication(config, backend_factory=lambda: model)

        async def run():
            async with application:
                conversation = application.start_conversation()
                history = []
                for prompt in ("My codename is Heron.", "What is my codename?"):
                    turn = await application.run_turn(
                        prompt, history=history, conversation_id=conversation, event_sink=sink
                    )
                    history += [Message("user", prompt), Message("assistant", turn.message)]

        asyncio.run(run())
        return model.requests

    def agent_requests(sink) -> list[list[Message]]:
        model = ScriptedModel()
        (tmp_path / "README.md").write_text("hello world\n")
        run_agent(model, create_read_only_registry(tmp_path), "Read README.md", event_sink=sink)
        return model.requests

    renderer = EventRenderer(Console(file=io.StringIO(), width=120))
    assert chat_requests(renderer.handle) == chat_requests(discard_event)
    assert agent_requests(renderer.handle) == agent_requests(discard_event)
    assert "Read README.md" in renderer._console.file.getvalue()


def test_eval_interactive_command_reports_results(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(eval_cli, "LlamaCppBackend", ScriptedModel)
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "one", "steps": [{"prompt": "Reply with ok.", "answer_contains": ["none"]}]}]
        )
    )

    result = CliRunner().invoke(
        cli.app,
        ["eval", "interactive", "--cases", str(cases), "--output-dir", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert "[1/1] one run 1: PASS" in result.stderr
    assert re.search(r"Passes\s+1/1", result.stdout)
    assert len(list((tmp_path / "out").glob("interactive-*.json"))) == 1


def test_a_turn_over_the_timeout_is_cancelled_not_abandoned(tmp_path: Path):
    model = ScriptedModel()
    case = InteractiveEvalCase(
        id="slow",
        steps=[InteractiveStep(prompt="Write about the Roman Empire.", answer_contains=["x"])],
    )

    started = time.monotonic()
    report, _ = run_interactive_evaluation(
        model, [case], runs=1, timeout_seconds=0.2, output_dir=tmp_path / "results"
    )

    step = report.results[0].steps[0]
    assert step.error == "The turn exceeded its 0.2-second timeout."
    assert not step.cancelled
    assert report.results[0].failure_reason == "turn_error"
    assert time.monotonic() - started < 3  # the model call stopped instead of running on
