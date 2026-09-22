import json
from io import StringIO

import pytest
from rich.console import Console

from assistant.agent.events import (
    AssistantDelta,
    AssistantFinished,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnStarted,
)
from assistant.agent.loop import ToolResult
from assistant.ui.renderer import EventRenderer


@pytest.fixture
def rendered():
    output = StringIO()
    renderer = EventRenderer(
        Console(file=output, width=200, no_color=True),
        answer_prefix="Jamie › ",
    )

    def render(*events) -> str:
        for event in events:
            renderer.handle(event)
        return output.getvalue()

    return render


def test_tool_progress_reads_as_prose_without_protocol_json(rendered):
    text = rendered(
        TurnStarted("read foo"),
        ToolStarted("read_file", {"path": "foo.py", "offset": 0}),
        ToolFinished("read_file", ToolResult(success=True, output="x\n" * 318), 0.1),
    )

    assert text == "  ▸ Reading foo.py\n  ✓ Read foo.py · 318 lines\n"
    assert "path" not in text
    assert "{" not in text


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("web_search", {"query": "free threading"}, '  ◦ Searching web for "free threading"\n'),
        ("fetch_url", {"url": "https://www.example.com/a/b"}, "  ▸ Reading example.com\n"),
        ("read_file", {"path": "src/jamie/agent.py"}, "  ▸ Reading src/jamie/agent.py\n"),
        ("write_file", {"path": "report.md", "content": "secret"}, "  ▸ Writing report.md\n"),
        ("open_application", {"name": "Safari"}, "  ▸ Opening Safari\n"),
        ("clipboard_read", {}, "  ▸ Reading the clipboard\n"),
        ("mystery_tool", {"path": "x"}, "  ▸ Running mystery_tool\n"),
    ],
)
def test_each_tool_start_names_the_action(rendered, tool, args, expected):
    assert rendered(ToolStarted(tool, args)) == expected


def test_search_results_are_counted_not_printed(rendered):
    output = json.dumps({"results": [{"url": "https://example.com", "title": "Example"}]})

    text = rendered(
        ToolStarted("web_search", {"query": "python"}),
        ToolFinished("web_search", ToolResult(success=True, output=output), 0.2),
    )

    assert text.endswith('  ✓ Searched web for "python" · 1 result\n')
    assert "example.com" not in text


def test_failed_tool_shows_the_action_and_the_reason(rendered):
    text = rendered(
        ToolStarted("fetch_url", {"url": "https://example.com/page"}),
        ToolFailed("fetch_url", "Browser navigation failed\nconnection timed out", 0.3),
    )

    assert text == (
        "  ▸ Reading example.com\n"
        "  ✗ Reading example.com failed\n"
        "    Browser navigation failed\n"
        "    connection timed out\n"
    )


def test_answer_is_printed_once_under_the_prefix(rendered):
    text = rendered(AssistantFinished("Hello [bold]Jamie[/bold]"))

    assert text == "Jamie › Hello [bold]Jamie[/bold]\n"


def test_streamed_answer_arrives_incrementally_and_is_not_repeated(rendered):
    text = rendered(
        TurnStarted("what is free-threading?"),
        AssistantDelta("Python free-threading "),
        AssistantDelta("became..."),
    )
    assert text == "Jamie › Python free-threading became..."

    text = rendered(AssistantFinished("Python free-threading became..."))
    assert text == "Jamie › Python free-threading became...\n"
