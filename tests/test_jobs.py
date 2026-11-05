"""Job applications end to end, in a real Chromium against local application pages."""

import asyncio
import glob
import json
import os
import threading
from pathlib import Path

import pytest

from assistant.background import BackgroundStore, BackgroundWorker
from assistant.background.store import DONE, NEEDS_INPUT
from assistant.browser import BrowserSession
from assistant.config import ProjectConfig
from assistant.jobs import JOB_KIND, queue_applications
from assistant.jobs.filler import best_option, classify
from assistant.jobs.profile import ApplicantProfile, ProfileStore, normalize_question
from assistant.jobs.runner import grants_for_task, job_application_runner
from assistant.jobs.tools import register_job_tools
from assistant.llm.base import LLMResponse
from assistant.tools.mac_files import MacPaths
from assistant.tools.registry import ToolRegistry

PAGES = Path(__file__).parent / "fixtures" / "pages"


def _chromium():
    if os.environ.get("JAMIE_BROWSER_PATH"):
        return os.environ["JAMIE_BROWSER_PATH"]
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return found[-1] if found else None


class Model:
    """Drafts answers and picks options like the local model, from scripted replies."""

    def __init__(self, answers=None, choices=None):
        self.answers = answers or {}
        self.choices = choices or {}
        self.questions = []

    def chat(self, messages, **options):
        question = messages[-1].content
        self.questions.append(question)
        table = self.choices if "Options:" in question else self.answers
        answer = next((a for q, a in table.items() if q.lower() in question.lower()), "UNKNOWN")
        return LLMResponse(json.dumps({"answer": answer}))


class App:
    def __init__(self, backend):
        self.backend = backend


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    pytest.importorskip("playwright")
    executable = _chromium()
    if executable is None:
        pytest.skip("No Chromium for Playwright on this machine.")
    os.environ["JAMIE_BROWSER_PATH"] = executable
    session = BrowserSession(tmp_path_factory.mktemp("profile"), channel=None, allow_file_urls=True)
    yield session
    session.close()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMIE_MEMORY_PATH", str(tmp_path / "memory.sqlite3"))
    resume = tmp_path / "resume.txt"
    resume.write_text("Sam Rivera. Python engineer at Initrode, 2019-2026. SQL, pandas.")
    profiles = ProfileStore(tmp_path / "applicant.json")
    profiles.save(
        ApplicantProfile(
            first_name="Sam",
            last_name="Rivera",
            email="sam@example.com",
            phone="+1 415 555 0100",
            linkedin="https://linkedin.com/in/samrivera",
            resume_path=str(resume),
            authorized_to_work=True,
            needs_sponsorship=False,
        )
    )
    store = BackgroundStore(tmp_path / "background.sqlite3")
    return store, profiles


def run_worker(store, runner, *, answer=None):
    """Drain the queue; ``answer`` approves (True) or denies (False) each submit request."""
    seen = []
    stop = threading.Event()

    def respond():
        while not stop.is_set():
            for approval in store.pending_approvals():
                seen.append(approval)
                store.answer(approval.id, answer)
            stop.wait(0.05)

    responder = threading.Thread(target=respond, daemon=True)
    if answer is not None:
        responder.start()
    worker = BackgroundWorker(
        store, {JOB_KIND: runner}, poll_seconds=0.05, grants_for=grants_for_task
    )
    try:
        asyncio.run(worker.drain())
    finally:
        stop.set()
    return seen


def runner_for(browser, profiles, model):
    return job_application_runner(App(model), ProjectConfig(), browser, profiles=profiles, pause=0)


def test_single_page_application_waits_for_approval_then_submits(browser, setup):
    store, profiles = setup
    model = Model(answers={"Why do you want to work at Acme": "I love building launch software."})
    [task_id] = queue_applications(store, [(PAGES / "greenhouse.html").as_uri()])

    approvals = run_worker(store, runner_for(browser, profiles, model), answer=True)

    task = store.task(task_id)
    assert task.status == DONE, task.error
    assert task.result.startswith("Submitted: Software Engineer at Acme")
    [approval] = approvals
    assert (
        approval.tool == "job_submit" and approval.attachment == task.payload["review_screenshot"]
    )
    assert "SUBMIT the job application" in approval.summary
    assert "First Name *: Sam" in approval.summary
    assert "I love building launch software. (drafted by Jamie)" in approval.summary
    filled = "\n".join(task.payload["filled"])
    assert "Resume/CV *: résumé: resume.txt" in filled
    assert "Are you legally authorized to work in the United States? *: Yes" in filled
    assert "Will you now or in the future require visa sponsorship? *: No" in filled
    assert "Gender: Decline To Self Identify" in filled
    assert "Veteran Status: I don't wish to answer" in filled
    assert "I agree to the privacy policy: checked" in filled
    assert Path(task.payload["confirmation_screenshot"]).exists() and task.payload["confirmed"]


def test_declining_leaves_the_application_unsent(browser, setup):
    store, profiles = setup
    model = Model(answers={"Why": "Rockets."})
    [task_id] = queue_applications(store, [(PAGES / "greenhouse.html").as_uri()])
    run_worker(store, runner_for(browser, profiles, model), answer=False)
    task = store.task(task_id)
    assert task.status == DONE and task.result.startswith("Not submitted")
    assert "submitted" not in task.payload


def test_auto_submit_needs_no_approval(browser, setup):
    store, profiles = setup
    model = Model(answers={"Why": "Rockets."})
    [task_id] = queue_applications(store, [(PAGES / "greenhouse.html").as_uri()], auto_submit=True)
    approvals = run_worker(store, runner_for(browser, profiles, model), answer=False)
    assert approvals == []  # a denial would have stopped it; nothing was asked
    assert store.task(task_id).result.startswith("Submitted")


def test_multi_step_application_from_a_job_posting(browser, setup):
    store, profiles = setup
    profiles.remember_answers({"Notice period": "Two weeks"})
    model = Model()
    [task_id] = queue_applications(store, [(PAGES / "jobpost.html").as_uri()])

    approvals = run_worker(store, runner_for(browser, profiles, model), answer=True)

    task = store.task(task_id)
    assert task.status == DONE, task.error
    assert task.payload["confirmed"]
    filled = "\n".join(task.payload["filled"])
    assert "Full name: Sam Rivera" in filled
    assert "Notice period: Two weeks" in filled  # from saved answers
    assert "How did you hear about us?: Company website" in filled
    assert approvals[0].summary.count("Submit application") == 1


def test_unknown_required_questions_come_back_to_the_user(browser, setup):
    store, profiles = setup
    [task_id] = queue_applications(store, [(PAGES / "jobpost.html").as_uri()])
    run_worker(store, runner_for(browser, profiles, Model()), answer=True)
    task = store.task(task_id)
    assert task.status == NEEDS_INPUT
    assert task.payload["questions"] == ["Notice period"]

    registry = register_job_tools(
        ToolRegistry(), store, MacPaths([str(PAGES.parent)]), profiles=profiles
    )
    answered = registry.execute(
        "jobs_answer_questions", {"task_id": task_id, "answers": {"Notice period": "1 month"}}
    )
    assert answered.success and store.task(task_id).status == "queued"
    run_worker(store, runner_for(browser, profiles, Model()), answer=True)
    assert store.task(task_id).status == DONE


def test_government_ids_are_never_filled(browser, setup):
    store, profiles = setup
    model = Model(answers={"Social Security": "123-45-6789"})
    [task_id] = queue_applications(store, [(PAGES / "ssn.html").as_uri()])
    run_worker(store, runner_for(browser, profiles, model), answer=True)
    task = store.task(task_id)
    assert task.status == NEEDS_INPUT
    assert task.payload["questions"] == ["Social Security Number"]
    assert not any("Social Security" in q for q in model.questions)


def test_login_walls_ask_the_user_to_sign_in(browser, setup):
    store, profiles = setup
    [task_id] = queue_applications(store, [(PAGES / "login.html").as_uri()])
    run_worker(store, runner_for(browser, profiles, Model()), answer=True)
    task = store.task(task_id)
    assert task.status == NEEDS_INPUT and "sign in" in task.error


def test_an_incomplete_profile_stops_before_opening_anything(setup):
    store, profiles = setup
    profiles.save(ApplicantProfile(first_name="Sam"))
    [task_id] = queue_applications(store, ["https://jobs.lever.co/acme/1"])
    run_worker(store, runner_for(None, profiles, Model()), answer=True)
    task = store.task(task_id)
    assert task.status == NEEDS_INPUT
    assert task.payload["questions"] == ["last name", "email", "phone", "resume path"]


def test_profile_tools_update_and_read_back(setup, tmp_path):
    store, profiles = setup
    registry = register_job_tools(
        ToolRegistry(), store, MacPaths([str(tmp_path)]), profiles=profiles
    )
    updated = registry.execute(
        "jobs_update_profile", {"fields": {"city": "Oakland", "willing_to_relocate": False}}
    )
    assert updated.success and "ready" in updated.output
    shown = json.loads(registry.execute("jobs_show_profile", {}).output)
    assert shown["city"] == "Oakland" and shown["willing_to_relocate"] is False
    bad = registry.execute("jobs_update_profile", {"fields": {"ssn": "1"}})
    assert not bad.success and "Unknown fields" in bad.output
    outside = registry.execute("jobs_update_profile", {"fields": {"resume_path": "/etc/hosts"}})
    assert not outside.success
    queued = registry.execute("jobs_apply", {"urls": ["https://jobs.lever.co/a/1", "ftp://x"]})
    assert queued.success and store.tasks()[0].payload["submit"] == "review"


def test_profile_from_resume_fills_only_empty_fields(setup, tmp_path):
    store, _ = setup
    profiles = ProfileStore(tmp_path / "fresh.json")
    profiles.update(email="mine@example.com")
    resume = tmp_path / "cv.txt"
    resume.write_text("Ada Lovelace, ada@example.com, +44 20 0000 0000, London")

    class Extractor:
        def chat(self, messages, **options):
            return LLMResponse(json.dumps({
                "first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com",
                "phone": "+44 20 0000 0000", "city": "London", "summary": "",
            }))  # fmt: skip

    registry = register_job_tools(
        ToolRegistry(), store, MacPaths([str(tmp_path)]), backend=Extractor(), profiles=profiles
    )
    result = registry.execute("jobs_profile_from_resume", {"path": str(resume)})
    assert result.success
    profile = profiles.load()
    assert (profile.first_name, profile.city, profile.email) == (
        "Ada",
        "London",
        "mine@example.com",
    )
    assert profile.resume_path == str(resume.resolve())


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("First Name *", "first_name"),
        ("Preferred first name", "preferred_name"),
        ("LinkedIn URL", "linkedin"),
        ("Will you require sponsorship?", "needs_sponsorship"),
        ("Are you authorized to work in Canada?", "authorized_to_work"),
        ("Résumé", "resume"),
        ("Desired salary", "salary_expectation"),
        ("Tell us about a project", None),
    ],
)
def test_labels_map_to_profile_fields(label, key):
    assert classify(label) == key


def test_options_match_meaning():
    assert best_option("Yes", ["Select", "Yes, I am", "No"]) == "Yes, I am"
    assert best_option("Decline to self-identify", ["Male", "I prefer not to say"]) == (
        "I prefer not to say"
    )
    assert best_option("Canada", ["United States", "Canada"]) == "Canada"
    assert best_option("Mars", ["Earth"]) is None


def test_saved_answers_match_close_wordings():
    profile = ApplicantProfile(
        answers={normalize_question("What is your notice period?"): "2w", "name": "Sam"}
    )
    assert profile.answer_for("Notice period (required)") == "2w"
    assert profile.answer_for("What is your notice period") == "2w"
    assert profile.answer_for("Company name") is None  # too short to match by wording
