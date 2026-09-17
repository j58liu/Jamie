import pytest

from assistant.agent.protocol import (
    ACTION_PROTOCOL_INSTRUCTIONS,
    FinishAction,
    ProtocolError,
    ToolAction,
    parse_action,
)


def test_protocol_instructions_define_both_actions():
    assert '{"action":"tool"' in ACTION_PROTOCOL_INSTRUCTIONS
    assert '{"action":"finish"' in ACTION_PROTOCOL_INSTRUCTIONS
    assert "exactly one JSON object" in ACTION_PROTOCOL_INSTRUCTIONS


def test_parses_tool_action():
    result = parse_action('{"action":"tool","tool":"read_file","args":{"path":"README.md"}}')
    assert result == ToolAction(action="tool", tool="read_file", args={"path": "README.md"})


def test_parses_finish_action():
    result = parse_action('{"action":"finish","message":"Done."}')
    assert result == FinishAction(action="finish", message="Done.")


def test_parses_json_fence():
    result = parse_action(
        '```json\n{"action":"tool","tool":"read_file","args":{"path":"README.md"}}\n```'
    )
    assert isinstance(result, ToolAction)


def test_parses_response_after_think_block():
    result = parse_action(
        "<think>I should answer with the finish action.</think>\n"
        '{"action":"finish","message":"Done."}'
    )
    assert isinstance(result, FinishAction)


def test_parses_fenced_response_after_think_block():
    result = parse_action(
        "<think>Choose the read_file tool.</think>\n"
        "```json\n"
        '{"action":"tool","tool":"read_file","args":{"path":"README.md"}}\n'
        "```"
    )
    assert isinstance(result, ToolAction)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not JSON", "not valid JSON"),
        ('Here is the action: {"action":"finish","message":"Done."}', "not valid JSON"),
        ('{"action":"wait","message":"Later."}', "does not match"),
        ('{"action":"tool","tool":"read_file"}', "does not match"),
        ('{"action":"tool","tool":"read_file","args":[]}', "does not match"),
        ('{"action":"finish"}', "does not match"),
        ('{"action":"finish","message":"Done.","extra":true}', "does not match"),
        ('{"action":"finish","message":"first","message":"second"}', "not valid JSON"),
        ('{"action":"tool","tool":"read_file","args":{"line":NaN}}', "not valid JSON"),
        ("<think>unfinished", "unclosed think block"),
        ('```json\n{"action":"finish","message":"Done."}', "invalid JSON fence"),
        ('```python\n{"action":"finish","message":"Done."}\n```', "invalid JSON fence"),
    ],
)
def test_rejects_invalid_protocol_responses(text: str, message: str):
    with pytest.raises(ProtocolError, match=message):
        parse_action(text)
