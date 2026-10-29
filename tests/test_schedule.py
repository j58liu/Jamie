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
        ("weekly Mon 09:00 Plan the week", ("weekly", "mon 09:00", "Plan the week")),
        ("weekly sunday 18:30 Review", ("weekly", "sun 18:30", "Review")),
        ("every 2h Check the news", ("every", "2h", "Check the news")),
        ("Every 30M Stretch", ("every", "30m", "Stretch")),
        ("on 2026-11-03 15:30 Vote", ("once", "2026-11-03 15:30", "Vote")),
    ],
)
def test_parse_reads_every_kind_of_schedule(argument, expected):
    assert schedule.parse(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        "daily 08:00",
        "weekly 08:00 x",
        "weekly funday 08:00 x",
        "at 8 x",
        "at 24:00 x",
        "every 2 x",
        "every 4m x",
        "on 2026-02-30 08:00 x",
        "hourly 08:00 x",
    ],
)
def test_parse_rejects_anything_else(argument):
    with pytest.raises(ValueError):
        schedule.parse(argument)


def test_descriptions_read_as_plain_english():
    assert schedule.describe("daily", "08:00") == "every day at 08:00"
    assert schedule.describe("weekly", "mon 09:00") == "every Mon at 09:00"
    assert schedule.describe("every", "2h") == "every 2h"
    assert schedule.describe("once", "15:30") == "once at 15:30"
    assert schedule.describe("once", "2026-11-03 15:30") == "once on 2026-11-03 15:30"


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
def test_next_run_is_today_or_tomorrow_in_local_wall_time():
    assert schedule.next_run("daily", "08:00", local("2026-10-20T07:00-04:00")) == local(
        "2026-10-20T08:00-04:00"
    )
    assert schedule.next_run("daily", "08:00", local("2026-10-20T08:00-04:00")) == local(
        "2026-10-21T08:00-04:00"
    )
    # Clocks go back on 1 November 2026: 08:00 stays 08:00 local, now UTC-5.
    after = schedule.next_run("daily", "08:00", local("2026-10-31T09:00-04:00"))
    assert after.isoformat() == "2026-11-01T08:00:00-05:00"


@pytest.mark.usefixtures("toronto")
def test_weekly_interval_and_dated_runs():
    tuesday = local("2026-10-20T10:00-04:00")
    assert schedule.next_run("weekly", "mon 09:00", tuesday) == local("2026-10-26T09:00-04:00")
    # From Saturday, the next Monday 09:00 is after the clocks go back, so UTC-5.
    weekly = schedule.next_run("weekly", "mon 09:00", local("2026-10-31T10:00-04:00"))
    assert weekly.isoformat() == "2026-11-02T09:00:00-05:00"
    assert schedule.next_run("weekly", "tue 11:00", tuesday) == local("2026-10-20T11:00-04:00")
    assert schedule.next_run("every", "90m", tuesday) == tuesday + timedelta(minutes=90)
    assert schedule.next_run("once", "2026-11-03 15:30", tuesday) == local("2026-11-03T15:30-05:00")


def test_repeating_tasks_skip_missed_runs_and_one_off_tasks_end():
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    assert schedule.after_run("once", "08:00", now) is None
    assert schedule.after_run("once", "2026-11-03 15:30", now) is None
    assert schedule.after_run("daily", "08:00", now) > now
    assert schedule.after_run("weekly", "mon 09:00", now) > now
    assert schedule.after_run("every", "2h", now) == now + timedelta(hours=2)


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


def test_store_edits_tasks_and_held_tasks_are_never_due(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    task_id = store.add_schedule("old", "weekly", "mon 09:00", now - timedelta(hours=1))

    assert store.update_schedule(task_id, prompt="new", missed="ask")
    assert store.update_schedule(task_id, held=True)
    assert not store.update_schedule(999, prompt="nothing")
    (task,) = store.schedules()
    assert (task.prompt, task.missed, task.held) == ("new", "ask", True)
    assert store.schedules(due_by=now) == []


def test_a_version_1_database_keeps_its_tasks(tmp_path):
    import sqlite3

    path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE schedules (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL,"
            " repeat TEXT NOT NULL CHECK (repeat IN ('daily', 'once')), time TEXT NOT NULL,"
            " next_run TEXT NOT NULL, created_at TEXT NOT NULL);"
            "INSERT INTO schedules VALUES"
            " (4, 'briefing', 'daily', '08:00', '2026-10-21T12:00:00+00:00', 'x');"
            "PRAGMA user_version = 1;"
        )

    store = MemoryStore(path)
    (task,) = store.schedules()
    assert (task.id, task.prompt, task.missed, task.held) == (4, "briefing", "run", False)
    later = datetime(2026, 10, 21, 12, tzinfo=UTC)
    assert store.add_schedule("week", "weekly", "mon 09:00", later) == 5


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

    assert "every Fri at 17:00" in run(state, "/schedule weekly fri 17:00 Weekly review")
    assert "every 2h" in run(state, "/schedule every 2h Check the news")

    listing = run(state, "/schedule")
    assert "every day at 08:00" in listing and "Give me a morning briefing" in listing
    assert "once at 15:30" in listing and "Call Alex" in listing
    assert "Weekly review" in listing and "Check the news" in listing
    assert "Usage: /schedule daily" in run(state, "/schedule weekly 08:00 x")
    assert "already passed" in run(state, "/schedule on 2020-01-01 08:00 Too late")

    assert run(state, "/unschedule 2") == "Removed scheduled task 2.\n"
    assert "No scheduled task 2" in run(state, "/unschedule 2")
    assert "Usage: /unschedule" in run(state, "/unschedule two")


def test_schedule_commands_edit_and_set_the_missed_policy(state):
    run(state, "/schedule daily 08:00 Morning briefing")

    assert run(state, "/schedule edit 1 Briefing with weather") == "Updated scheduled task 1.\n"
    assert run(state, "/schedule missed 1 skip") == "Updated scheduled task 1.\n"
    listing = run(state, "/schedule")
    assert "Briefing with weather" in listing and "(if missed: skip)" in listing

    assert "Usage: /schedule missed" in run(state, "/schedule missed 1 later")
    assert "Usage: /schedule edit" in run(state, "/schedule edit 1")
    assert "No scheduled task 9" in run(state, "/schedule edit 9 x")
    # The terminal has no task runner; Jamie.app handles /schedule run.
    assert "Open Jamie.app" in run(state, "/schedule run 1")
    assert "No scheduled task 9" in run(state, "/schedule run 9")


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
