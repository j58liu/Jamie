import time
from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
from rich.console import Console

from assistant import schedule
from assistant.memory import MemoryStore
from assistant.tools.scopes import UNATTENDED_TOOLS
from assistant.ui.commands import ReplState, create_command_registry


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("daily 08:00 Morning briefing", ("daily", "08:00", "Morning briefing")),
        ("Daily 7:05 Weather\nand news", ("daily", "07:05", "Weather\nand news")),
        ("at 15:30 Remind me to call Alex", ("once", "15:30", "Remind me to call Alex")),
    ],
)
def test_parse_reads_daily_and_one_off_tasks(argument, expected):
    assert schedule.parse(argument) == expected


@pytest.mark.parametrize("argument", ["daily 08:00", "weekly 08:00 x", "at 8 x", "at 24:00 x"])
def test_parse_rejects_anything_else(argument):
    with pytest.raises(ValueError):
        schedule.parse(argument)


@pytest.fixture
def toronto(monkeypatch):
    monkeypatch.setenv("TZ", "America/Toronto")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def local(text):
    return datetime.fromisoformat(text).astimezone()


@pytest.mark.usefixtures("toronto")
def test_next_occurrence_is_today_or_tomorrow_in_local_wall_time():
    assert schedule.next_occurrence("08:00", local("2026-10-20T07:00-04:00")) == local(
        "2026-10-20T08:00-04:00"
    )
    assert schedule.next_occurrence("08:00", local("2026-10-20T08:00-04:00")) == local(
        "2026-10-21T08:00-04:00"
    )
    # Clocks go back on 1 November 2026: 08:00 stays 08:00 local, now UTC-5.
    after = schedule.next_occurrence("08:00", local("2026-10-31T09:00-04:00"))
    assert after.isoformat() == "2026-11-01T08:00:00-05:00"


def test_daily_tasks_skip_missed_days_and_one_off_tasks_end():
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    assert schedule.after_run("once", "08:00", now) is None
    assert schedule.after_run("daily", "08:00", now) > now


def test_store_lists_due_tasks_reschedules_and_removes(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    later = store.add_schedule("later", "daily", "09:00", now + timedelta(hours=1))
    due = store.add_schedule("due", "once", "08:00", now - timedelta(minutes=5))

    assert [task.prompt for task in store.schedules()] == ["due", "later"]
    assert [task.id for task in store.schedules(due_by=now)] == [due]
    store.reschedule(due, None)
    store.reschedule(later, now - timedelta(minutes=1))
    assert [task.id for task in store.schedules(due_by=now)] == [later]
    assert store.remove_schedule(later) and not store.remove_schedule(later)
    assert store.schedules() == []


def test_forgotten_memories_are_no_longer_recalled(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    session = store.start_session("chat")
    bad = store.remember_exchange(session, "hello", "I copied hello to the clipboard.")
    good = store.remember_exchange(session, "hello there", "Hi! How can I help?")

    assert {r.id for r in store.search_memories("hello")} == {bad, good}
    assert store.forget_memory(bad) and not store.forget_memory(bad)
    assert [r.id for r in store.search_memories("hello")] == [good]


def run(state, line):
    create_command_registry().dispatch(state, line)
    output = state.console.file.getvalue()
    state.console.file.seek(0)
    state.console.file.truncate()
    return output


class Application:
    def __init__(self, memory):
        self.memory = memory


@pytest.fixture
def state(tmp_path):
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    return ReplState(Console(file=StringIO(), width=200), application=Application(memory))


def test_schedule_commands_add_list_and_remove(state):
    added = run(state, "/schedule daily 08:00 Give me a morning briefing")
    assert added.startswith("Scheduled task 1: every day at 08:00, next ")
    assert "Scheduled task 2: once at 15:30" in run(state, "/schedule at 15:30 Call Alex")

    listing = run(state, "/schedule")
    assert "daily 08:00" in listing and "Give me a morning briefing" in listing
    assert "once 15:30" in listing and "Call Alex" in listing
    assert "Usage: /schedule daily" in run(state, "/schedule weekly 08:00 x")

    assert run(state, "/unschedule 2") == "Removed scheduled task 2.\n"
    assert "No scheduled task 2" in run(state, "/unschedule 2")
    assert "Usage: /unschedule" in run(state, "/unschedule two")


def test_memory_commands_show_and_forget(state):
    memory = state.application.memory
    session = memory.start_session("chat")
    memory_id = memory.remember_exchange(session, "hello", "I copied hello to the clipboard.")

    listing = run(state, "/memories hello")
    assert f"{memory_id}" in listing and "I copied hello" in listing
    assert "Forgot memory" in run(state, f"/forget {memory_id}")
    assert "No memories match hello" in run(state, "/memories hello")
    assert f"No memory {memory_id}" in run(state, f"/forget {memory_id}")


def test_unattended_tools_can_read_and_search_but_never_act():
    assert {"read_file", "web_search", "research"} <= UNATTENDED_TOOLS
    assert UNATTENDED_TOOLS.isdisjoint(
        {"write_file", "edit_file", "trash_file", "clipboard_write", "run_applescript"}
    )
