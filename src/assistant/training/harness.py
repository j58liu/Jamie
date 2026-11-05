"""Run a scripted trajectory through Jamie's real tools and record training examples.

The system prompt, tool list, tool observations, and action JSON are produced by the
same code Jamie runs (``_system_instructions``, ``tool_observation``, the registry), so
the model is trained on exactly what it will see. Only the model's own turns are
scripted; each one becomes an example whose last message is the turn to learn.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from assistant.agent.loop import ToolResult, _system_instructions, tool_observation
from assistant.agent.protocol import ToolAction
from assistant.background import BackgroundStore
from assistant.connectors import Connectors
from assistant.connectors.fakes import FakeCalendar, FakeGmail, FakeReminders
from assistant.connectors.timeutil import describe_now
from assistant.jobs.profile import ApplicantProfile, ProfileStore
from assistant.llm.base import LLMResponse
from assistant.memory import MemoryStore
from assistant.runtime.application import IDENTITY
from assistant.security.approvals import CallbackApprovalProvider
from assistant.tools import mac_files
from assistant.tools.integrations import register_personal_tools
from assistant.tools.mac_files import MacPaths, register_mac_file_tools
from assistant.tools.macos import CommandResult
from assistant.tools.registry import ToolRegistry, create_macos_registry
from assistant.tools.scopes import select_scopes
from assistant.tools.tasks import register_task_tools
from assistant.training.worlds import World

type Step = tuple[str, dict[str, Any]] | Callable[[list[ToolResult]], tuple[str, dict]]
type Finish = str | Callable[[list[ToolResult]], str]


@dataclass
class Plan:
    scopes: tuple[str, ...]
    prompt: str
    steps: list[Step] = field(default_factory=list)
    finish: Finish = ""
    facts: list[str] = field(default_factory=list)  # personal facts in the system prompt
    approve: bool = True
    web: dict[str, str] = field(default_factory=dict)  # stub web tool outputs by tool
    pages: dict[str, str] = field(default_factory=dict)  # stub browser outputs by tool
    weight: int = 1
    expect_fail: frozenset[int] = frozenset()  # steps meant to fail (to teach recovery)


class TrajectoryError(RuntimeError):
    """A scripted step failed unexpectedly; the trajectory would teach a mistake."""


class _Runner:
    """macOS commands (open, mdfind) succeed without doing anything."""

    def run(self, arguments, *, input_text=None, timeout):
        if arguments and arguments[0] == "mdfind":
            from assistant.tools.base import ToolError

            raise ToolError("no Spotlight here")  # use the folder walk
        return CommandResult("", "")


class _Stub:
    """A tool with a real tool's name, schema, and risk, and a scripted output."""

    def __init__(self, real: type, output: Callable[[Any], str]) -> None:
        self.name = real.name
        self.description = real.description
        self.input_schema = real.Input if hasattr(real, "Input") else real.input_schema
        self.risk = real.risk
        self.ends_turn = getattr(real, "ends_turn", False)
        self._output = output
        context = getattr(real, "prompt_context", None)
        if isinstance(context, staticmethod) or callable(context):
            try:
                text = real.prompt_context()
            except TypeError:
                text = None
            if text:
                self.prompt_context = lambda: text

    def execute(self, arguments) -> str:
        return self._output(arguments)


class _Extractor:
    """Stands in for the model inside jobs_profile_from_resume."""

    def __init__(self, world: World) -> None:
        self._world = world

    def chat(self, messages, **options):
        w = self._world
        return LLMResponse(
            json.dumps(
                {
                    "first_name": w.user_first,
                    "last_name": w.user_last,
                    "email": f"{w.user_first.lower()}@example.com",
                    "phone": "+1 555 0142",
                    "city": w.city[0],
                    "summary": "Software engineer.",
                }
            )
        )


@contextlib.contextmanager
def home_of(world: World) -> Iterator[None]:
    original = mac_files._home
    mac_files._home = lambda: world.home.resolve()
    try:
        yield
    finally:
        mac_files._home = original


def _web_tools(registry: ToolRegistry, outputs: dict[str, str]) -> None:
    from assistant.research import ResearchTool
    from assistant.tools.web import ExtractPageTool, FetchUrlTool, WebSearchTool

    for real in (ResearchTool, WebSearchTool, FetchUrlTool, ExtractPageTool):
        text = outputs.get(real.name, "No results.")
        registry.register(_Stub(real, lambda _args, text=text: text))


def _browser_tools(registry: ToolRegistry, outputs: dict[str, str]) -> None:
    from assistant.browser import tools as browser

    for real in (
        browser.BrowserOpenTool,
        browser.BrowserSnapshotTool,
        browser.BrowserClickTool,
        browser.BrowserTypeTool,
        browser.BrowserSelectTool,
        browser.BrowserCheckTool,
        browser.BrowserScrollTool,
        browser.BrowserBackTool,
        browser.BrowserUploadTool,
        browser.BrowserSubmitTool,
        browser.BrowserScreenshotTool,
        browser.BrowserShowTool,
    ):
        text = outputs.get(real.name, "Done.")
        registry.register(_Stub(real, lambda _args, text=text: text))


def make_registry(world: World, plan: Plan, approvals) -> ToolRegistry:
    workspace = world.home / "Documents" / "Work"
    registry = create_macos_registry(workspace, approvals=approvals, runner=_Runner())
    register_mac_file_tools(registry, MacPaths([str(world.home)]), _Runner())
    _web_tools(registry, plan.web)
    _browser_tools(registry, plan.pages)
    connectors = Connectors(
        tz=world.tz,
        gmail=FakeGmail(list(world.messages)),
        calendar=FakeCalendar(list(world.events), world.tz),
        reminders=FakeReminders(list(world.reminders)),
    )
    register_personal_tools(registry, connectors, now=lambda: world.now)
    store = BackgroundStore(world.root / "background.sqlite3")
    memory = MemoryStore(world.root / "memory.sqlite3")
    from assistant.jobs.tools import register_job_tools

    profiles = ProfileStore(world.root / "applicant.json")
    if not profiles.path.exists():
        profiles.save(ApplicantProfile())
    register_job_tools(
        registry, store, MacPaths([str(world.home)]), _Extractor(world), profiles=profiles
    )
    register_task_tools(registry, store, lambda: memory)
    return registry


def system_prompt(tools, world: World, facts: list[str]) -> str:
    context = [IDENTITY]
    if facts:
        day = world.now.date().isoformat()
        lines = "".join(
            f"- {fact} [fact; you said; {day}; memory {i + 1}]\n" for i, fact in enumerate(facts)
        )
        context.append(
            "Known facts about the user from local memory. Provenance is in brackets; "
            "facts stated or corrected by the user outrank inferred ones. Use only facts "
            f"relevant to the request.\n{lines}"
        )
    return f"{_system_instructions(tools)}\n\n" + "\n\n".join(context)


def _action(tool: str, args: dict) -> str:
    return json.dumps(
        {"action": "tool", "tool": tool, "args": args}, ensure_ascii=False, separators=(",", ":")
    )


def _finish(message: str) -> str:
    return json.dumps({"action": "finish", "message": message}, ensure_ascii=False,
                      separators=(",", ":"))  # fmt: skip


def run_plan(world: World, plan: Plan) -> list[dict]:
    """Execute the plan; return one example per model turn."""
    approvals = CallbackApprovalProvider(lambda request: plan.approve)
    with home_of(world):
        registry = make_registry(world, plan, approvals)
        tools = select_scopes(registry, plan.scopes)
        # The clock line comes from the connector tools; keep it when they are scoped out.
        system = system_prompt(tools, world, plan.facts)
        if "Current local time" not in system:
            system = system.replace(IDENTITY, f"{describe_now(world.now)}\n\n{IDENTITY}", 1)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": plan.prompt},
        ]
        examples = []
        observations: list[ToolResult] = []
        for step in plan.steps:
            tool, args = step(observations) if callable(step) else step
            action = _action(tool, args)
            examples.append({"messages": [*messages, {"role": "assistant", "content": action}]})
            result = tools.execute(tool, dict(args))
            index = len(observations)
            if result.success == (index in plan.expect_fail):
                raise TrajectoryError(f"step {index} {tool}: {result.output[:200]}")
            observations.append(result)
            if result.denied or result.final:
                return examples * plan.weight  # the runtime ends the turn itself
            observation = tool_observation(ToolAction(action="tool", tool=tool, args=args), result)
            messages += [
                {"role": "assistant", "content": action},
                {"role": "user", "content": observation.content},
            ]
        final = plan.finish(observations) if callable(plan.finish) else plan.finish
        examples.append({"messages": [*messages, {"role": "assistant", "content": _finish(final)}]})
        registry.close()
    return examples * plan.weight
