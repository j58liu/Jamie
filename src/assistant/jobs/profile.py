"""The applicant profile Jamie fills job applications from, and answers it has learned.

It lives in Jamie's private data folder (which the file tools cannot read), and changes
only when the user says so: in conversation, from their résumé, or by answering a
question an application asked.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DECLINE = "Decline to self-identify"


class ApplicantProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str = ""
    last_name: str = ""
    preferred_name: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    address: str = ""
    postal_code: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""
    resume_path: str = ""
    cover_letter_path: str = ""
    current_company: str = ""
    current_title: str = ""
    years_experience: str = ""
    school: str = ""
    degree: str = ""
    graduation_year: str = ""
    authorized_to_work: bool | None = None
    needs_sponsorship: bool | None = None
    willing_to_relocate: bool | None = None
    salary_expectation: str = ""
    earliest_start: str = ""
    notice_period: str = ""
    how_did_you_hear: str = "Company website"
    pronouns: str = ""
    gender: str = DECLINE
    race: str = DECLINE
    veteran: str = DECLINE
    disability: str = DECLINE
    write_cover_letters: bool = True
    summary: str = Field(default="", max_length=4000)  # a few lines about the applicant
    answers: dict[str, str] = Field(default_factory=dict)  # question -> the user's answer

    @property
    def full_name(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name) if part)

    @property
    def location(self) -> str:
        return ", ".join(part for part in (self.city, self.state, self.country) if part)

    def missing(self) -> list[str]:
        """What every application needs and the profile still lacks."""
        needed = {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "resume_path": self.resume_path,
        }
        return [name for name, value in needed.items() if not value]

    def for_model(self) -> dict[str, object]:
        """What the model may see when it drafts an answer (no file paths)."""
        data = self.model_dump(exclude={"resume_path", "cover_letter_path", "answers"})
        data["full_name"] = self.full_name
        return {key: value for key, value in data.items() if value not in ("", None)}

    def answer_for(self, question: str) -> str | None:
        key = normalize_question(question)
        if not key:
            return None
        if key in self.answers:
            return self.answers[key]
        for known, answer in self.answers.items():
            if known and (known in key or key in known) and min(len(known), len(key)) >= 12:
                return answer
        return None


def normalize_question(question: str) -> str:
    text = re.sub(r"[*:?()\[\]]", " ", question.casefold())
    text = re.sub(r"\b(required|optional)\b", " ", text)
    return " ".join(text.split())[:200]


def default_profile_path() -> Path:
    from assistant.memory import default_memory_path

    return default_memory_path().with_name("applicant.json")


class ProfileStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_profile_path()

    def load(self) -> ApplicantProfile:
        try:
            return ApplicantProfile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ApplicantProfile()
        except (OSError, ValidationError, ValueError):
            return ApplicantProfile()

    def save(self, profile: ApplicantProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2)
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".applicant-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(data)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def update(self, **changes: object) -> ApplicantProfile:
        profile = self.load()
        merged = {**profile.model_dump(), **changes}
        updated = ApplicantProfile.model_validate(merged)
        self.save(updated)
        return updated

    def remember_answers(self, answers: dict[str, str]) -> ApplicantProfile:
        profile = self.load()
        for question, answer in answers.items():
            key = normalize_question(question)
            if key and answer.strip():
                profile.answers[key] = answer.strip()
        self.save(profile)
        return profile
