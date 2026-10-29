import json
from datetime import UTC, datetime
from pathlib import Path

from assistant.config import ProjectConfig
from assistant.connectors import (
    Capability,
    ConnectionStatus,
    ConnectorInfo,
    Connectors,
    build_connectors,
)
from assistant.security import PermissionPolicy, PolicyAction
from assistant.tools import ToolRegistry, create_agent_registry
from assistant.tools.integrations import PERSONAL_TOOLS, register_personal_tools
from assistant.tools.scopes import SCOPE_MAX_STEPS, SCOPE_TOOLS, ToolScope, select_tools


class Stub:
    def __init__(self, name):
        self.name = name

    def info(self):
        return ConnectorInfo(self.name, ConnectionStatus.CONNECTED, frozenset({Capability.READ}))


def connectors():
    return Connectors(
        tz=UTC,
        gmail=Stub("gmail"),
        calendar=Stub("google_calendar"),
        reminders=Stub("apple_reminders"),
    )


def test_personal_scope_contains_exactly_the_connector_tools(tmp_path: Path):
    assert SCOPE_TOOLS[ToolScope.PERSONAL] == PERSONAL_TOOLS
    registry = create_agent_registry(tmp_path, connectors=connectors())
    names = {d["name"] for d in select_tools(registry, ToolScope.PERSONAL).definitions()}
    assert names == PERSONAL_TOOLS
    assert SCOPE_MAX_STEPS[ToolScope.PERSONAL] > 12


def test_no_connectors_registers_no_personal_tools(tmp_path: Path):
    registry = create_agent_registry(tmp_path)
    names = {d["name"] for d in registry.definitions()}
    assert not names & PERSONAL_TOOLS
    assert build_connectors(ProjectConfig()).all() == []


def test_only_configured_connectors_are_registered():
    registry = register_personal_tools(ToolRegistry(), Connectors(tz=UTC, gmail=Stub("gmail")))
    names = {d["name"] for d in registry.definitions()}
    assert "gmail_search_messages" in names
    assert not any(name.startswith(("calendar_", "reminders_")) for name in names)


def test_reads_run_freely_and_external_actions_ask():
    policy = PermissionPolicy()
    for name in PERSONAL_TOOLS:
        expected = (
            PolicyAction.ALLOW
            if name.endswith(
                (
                    "_list",
                    "_status",
                    "_search_messages",
                    "_get_message",
                    "_get_thread",
                    "_get_draft",
                    "_list_events",
                    "_check_availability",
                    "_get_event",
                )
            )
            else PolicyAction.CONFIRM
        )
        assert policy.decide(name).action is expected, name


def test_status_tool_reports_each_connector():
    registry = register_personal_tools(ToolRegistry(), connectors())
    result = registry.execute("connectors_status", {})
    assert [item["name"] for item in json.loads(result.output)] == [
        "gmail",
        "google_calendar",
        "apple_reminders",
    ]


def test_personal_prompt_is_small_and_has_one_time_line():
    registry = register_personal_tools(
        ToolRegistry(), connectors(), now=lambda: datetime(2026, 10, 25, 14, tzinfo=UTC)
    )
    instructions = registry.instructions()
    assert instructions.count("Current local time") == 1
    assert '"tool":"gmail_search_messages"' in instructions


def test_configured_google_client_creates_connectors(tmp_path: Path):
    secrets = tmp_path / "client.json"
    secrets.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "s"}}))
    config = ProjectConfig.model_validate(
        {
            "timezone": "America/Toronto",
            "google": {"client_secrets_path": str(secrets)},
            "reminders": {"enabled": True, "helper_path": str(tmp_path / "helper")},
        }
    )
    built = build_connectors(config)
    assert [c.name for c in built.all()] == ["gmail", "google_calendar", "apple_reminders"]
    assert built.tz.key == "America/Toronto"
