import asyncio
import io
import json

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from assistant.config import ProjectConfig
from assistant.llm.base import LLMResponse, current_response_schema
from assistant.runtime import JamieApplication
from assistant.tools.registry import create_workspace_registry
from assistant.tools.scopes import ToolScope, select_tools
from assistant.ui.repl import run_repl


def test_modes_and_local_commands_reduce_model_work_without_bypassing_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("hello from the file\n")
    calls = []
    registries = []

    class Backend:
        def chat(self, messages, **options):
            calls.append((list(messages), current_response_schema()))
            if current_response_schema() is None:
                return LLMResponse("Chat answer")
            return LLMResponse(json.dumps({"action": "finish", "message": "Task answer"}))

    def tools():
        registry = create_workspace_registry(tmp_path)
        registries.append(registry)
        return registry

    console = Console(file=io.StringIO(), width=200, no_color=True)

    async def run():
        with create_pipe_input() as pipe:
            pipe.send_text(
                "task one\n/mode chat\nchat two\n/scope read\n/mode task\ntask three\n"
                "/read notes.md\n/find *.md\n/search hello\n/ls\n/read ../outside\n/quit\n"
            )
            async with JamieApplication(
                ProjectConfig(memory_path=str(tmp_path / "memory.sqlite3")),
                backend_factory=Backend,
            ) as app:
                await run_repl(
                    app,
                    console=console,
                    session=PromptSession(input=pipe, output=DummyOutput()),
                    agent_tools_factory=tools,
                )
                assert app.model_calls == 3

    asyncio.run(run())
    assert len(calls) == 3 and len(registries) == 1
    assert calls[1][1] is None
    assert "write_file" in calls[0][0][0].content
    assert "write_file" not in calls[2][0][0].content
    assert len(calls[2][0][0].content) < len(calls[0][0][0].content)
    output = console.file.getvalue()
    assert "Chat answer" in output and "Tool scope: read" in output
    assert "hello from the file" in output and "notes.md" in output
    assert "outside" in output
    audited = [event.tool for event in registries[0].audit.events]
    assert all(
        name in audited for name in ("read_file", "find_files", "search_files", "list_directory")
    )


def test_smaller_tool_sets_keep_policy_and_undo_state(tmp_path):
    registry = create_workspace_registry(tmp_path)
    read = select_tools(registry, ToolScope.READ)
    files = select_tools(registry, ToolScope.FILES)
    assert len(read.definitions()) == 4
    assert not read.execute("write_file", {"path": "x", "content": "bad"}).success
    assert files.execute("write_file", {"path": "x", "content": "bad"}).denied
    assert not (tmp_path / "x").exists()
    assert read.audit is files.audit is registry.audit
    assert read._changes is files._changes is registry._changes
    registry.close()
