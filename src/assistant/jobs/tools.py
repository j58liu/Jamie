"""Tools for job applications: the applicant profile, answers, and queueing postings."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from assistant.background.store import NEEDS_INPUT, BackgroundStore
from assistant.jobs import JOB_KIND, queue_applications
from assistant.jobs.profile import ApplicantProfile, ProfileStore
from assistant.llm.base import Message, constrained_to
from assistant.security.policy import PermissionLevel
from assistant.tools.base import ToolError

JOB_TOOLS = (
    "jobs_show_profile",
    "jobs_update_profile",
    "jobs_profile_from_resume",
    "jobs_answer_questions",
    "jobs_apply",
)
EDITABLE = tuple(name for name in ApplicantProfile.model_fields if name not in {"answers"})


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShowProfileTool:
    name = "jobs_show_profile"
    description = "Show the applicant profile Jamie uses for job applications, and what is missing."
    risk = PermissionLevel.READ

    class Input(_Input):
        pass

    input_schema = Input

    def __init__(self, profiles: ProfileStore) -> None:
        self._profiles = profiles

    def execute(self, arguments: Input) -> str:
        profile = self._profiles.load()
        data = {k: v for k, v in profile.model_dump().items() if v not in ("", None, {})}
        data["saved_answers"] = len(profile.answers)
        data.pop("answers", None)
        data["missing"] = profile.missing()
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class UpdateProfileTool:
    name = "jobs_update_profile"
    description = (
        'Save details the user gives for job applications, e.g. {"phone":"…", '
        '"needs_sponsorship":false, "resume_path":"~/Documents/resume.pdf"}. Fields: '
        + ", ".join(EDITABLE)
        + "."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        fields: dict[str, str | bool | None] = Field(min_length=1, max_length=40)

    input_schema = Input

    def __init__(self, profiles: ProfileStore, paths) -> None:
        self._profiles = profiles
        self._paths = paths

    def execute(self, arguments: Input) -> str:
        unknown = sorted(set(arguments.fields) - set(EDITABLE))
        if unknown:
            raise ToolError(f"Unknown fields {unknown}. Use: {', '.join(EDITABLE)}.")
        changes = dict(arguments.fields)
        for key in ("resume_path", "cover_letter_path"):
            if changes.get(key):
                changes[key] = str(self._paths.resolve(str(changes[key])))
        try:
            profile = self._profiles.update(**changes)
        except ValueError as exc:
            raise ToolError(f"Those details do not fit the profile: {exc}") from exc
        missing = profile.missing()
        tail = f" Still missing: {', '.join(missing)}." if missing else " The profile is ready."
        return f"Saved {', '.join(sorted(changes))}.{tail}"


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string", "maxLength": 300}
        for key in (
            "first_name", "last_name", "email", "phone", "city", "state", "country",
            "linkedin", "github", "website", "current_company", "current_title",
            "years_experience", "school", "degree", "graduation_year",
        )
    }
    | {"summary": {"type": "string", "maxLength": 800}},
    "required": ["first_name", "last_name", "email", "phone"],
    "additionalProperties": False,
}  # fmt: skip


class ProfileFromResumeTool:
    name = "jobs_profile_from_resume"
    description = (
        "Set up the job-application profile from the user's résumé file (path from "
        "mac_search_files). Fills empty fields only and remembers the résumé to upload."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        path: str = Field(min_length=1, max_length=1000)

    input_schema = Input

    def __init__(self, profiles: ProfileStore, paths, backend) -> None:
        self._profiles = profiles
        self._paths = paths
        self._backend = backend

    def execute(self, arguments: Input) -> str:
        from assistant.agent.protocol import _remove_thinking
        from assistant.tools.mac_files import file_text

        path: Path = self._paths.resolve(arguments.path)
        text = file_text(path)[:8000]
        if not text.strip():
            raise ToolError("The résumé has no readable text.")
        extracted: dict[str, str] = {}
        if self._backend is not None:
            system = (
                "Copy the applicant's details from this résumé into the form. Copy exactly; "
                "leave a field empty when the résumé does not state it."
            )
            with constrained_to(PROFILE_SCHEMA):
                response = self._backend.chat(
                    [Message("system", system), Message("user", text)],
                    temperature=0.0,
                    max_tokens=900,
                )
            try:
                extracted = json.loads(_remove_thinking(response.content))
            except ValueError as exc:
                raise ToolError("Could not read details from the résumé.") from exc
        profile = self._profiles.load()
        changes: dict[str, object] = {"resume_path": str(path)}
        for key, value in extracted.items():
            if isinstance(value, str) and value.strip() and not getattr(profile, key, ""):
                changes[key] = value.strip()
        profile = self._profiles.update(**changes)
        missing = profile.missing()
        filled = sorted(k for k in changes if k != "resume_path")
        note = f" Still missing: {', '.join(missing)}." if missing else ""
        return (
            f"Saved the résumé ({path.name}) and {len(filled)} details: {', '.join(filled)}."
            f"{note} Ask the user about work authorization and sponsorship if not set."
        )


class AnswerQuestionsTool:
    name = "jobs_answer_questions"
    description = (
        "Save the user's answers to questions a job application asked (task status "
        "needs_input), then retry that application. answers maps each question to the answer."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        task_id: int = Field(ge=1)
        answers: dict[str, str] = Field(min_length=1, max_length=40)

    input_schema = Input

    def __init__(self, profiles: ProfileStore, store: BackgroundStore) -> None:
        self._profiles = profiles
        self._store = store

    def execute(self, arguments: Input) -> str:
        task = self._store.task(arguments.task_id)
        if task is None or task.kind != JOB_KIND:
            raise ToolError(f"Task {arguments.task_id} is not a job application.")
        self._profiles.remember_answers(arguments.answers)
        if task.status == NEEDS_INPUT:
            self._store.retry(task.id)
            return f"Saved {len(arguments.answers)} answers and restarted task {task.id}."
        return f"Saved {len(arguments.answers)} answers for future applications."


class ApplyTool:
    name = "jobs_apply"
    description = (
        "Apply to job postings in the background, one task per link. auto_submit=true only "
        "if the user said to submit without asking; otherwise they approve each submission."
    )
    risk = PermissionLevel.SAFE_WRITE

    class Input(_Input):
        urls: list[str] = Field(min_length=1, max_length=25)
        auto_submit: bool = False

    input_schema = Input

    def __init__(self, store: BackgroundStore) -> None:
        self._store = store

    def execute(self, arguments: Input) -> str:
        urls = [u.strip() for u in arguments.urls if u.strip().startswith(("http://", "https://"))]
        if not urls:
            raise ToolError("Give the links to the job postings (https://…).")
        ids = queue_applications(
            self._store, urls, auto_submit=arguments.auto_submit, source="agent"
        )
        return f"Queued {len(ids)} applications as background tasks {ids}."


def register_job_tools(registry, store: BackgroundStore | None, paths, backend=None,
                       profiles: ProfileStore | None = None):  # fmt: skip
    if store is None:
        return registry
    profiles = profiles or ProfileStore()
    registry.register(ShowProfileTool(profiles))
    registry.register(UpdateProfileTool(profiles, paths))
    registry.register(ProfileFromResumeTool(profiles, paths, backend))
    registry.register(AnswerQuestionsTool(profiles, store))
    registry.register(ApplyTool(store))
    return registry
