"""Fill a job application page by page, and stop at the final button.

Known fields (name, email, phone, links, work authorization, EEO questions...) are
matched to the applicant profile by their labels, without the model. The model only
drafts answers to open questions, from the profile and résumé, and picks among options
when no rule applies; it says UNKNOWN rather than invent. Questions Jamie must never
answer for the user (government IDs, bank details) and unanswerable required questions
are returned to the user instead of being guessed.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from assistant.browser.session import BrowserTab, Element, Snapshot
from assistant.jobs.profile import DECLINE, ApplicantProfile
from assistant.llm.base import Message, constrained_to

MAX_PAGES = 12
TEXT_TYPES = {"", "text", "email", "tel", "url", "number", "search", "date", "month"}

ENTRY = re.compile(
    r"^(easy apply|apply now|apply for this (job|position|role)|apply to this job|"
    r"apply manually|apply|i'?m interested|start (your )?application|apply on company site|"
    r"continue to application)\b",
    re.IGNORECASE,
)
NEXT = re.compile(
    r"^(next|continue|save (and|&) continue|review|review (your )?application|proceed|"
    r"next step|go to next step)\b",
    re.IGNORECASE,
)
FINAL = re.compile(
    r"^(submit|submit (your |my )?application|send (my |your )?application|apply|"
    r"complete (your )?application|finish|submit and apply)\b",
    re.IGNORECASE,
)
SUCCESS = re.compile(
    r"(thank you for (applying|your application|your interest)|application (was |has been )?"
    r"(submitted|received|sent)|we('ve| have) received your application|you applied|"
    r"your application is in|successfully (submitted|applied))",
    re.IGNORECASE,
)
LOGIN = re.compile(r"\b(sign in|log in|login|create (an )?account|password)\b", re.IGNORECASE)
NEVER = re.compile(
    r"social security|\bssn\b|\bsin number|national (id|insurance)|passport|driver'?s licen|"
    r"bank (account|routing)|credit card|card number|\bpassword\b",
    re.IGNORECASE,
)
CONSENT = re.compile(
    r"\b(i (agree|acknowledge|certify|consent|confirm|understand|have read)|privacy|"
    r"terms (of|and)|data (processing|retention))\b",
    re.IGNORECASE,
)

# label pattern -> profile attribute (first match wins; order matters)
RULES: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), key)
    for pattern, key in (
        (r"preferred (first )?name|nickname", "preferred_name"),
        (r"first name|given name|^first$", "first_name"),
        (r"last name|surname|family name|^last$", "last_name"),
        (r"^(full |legal |your )?name\b", "full_name"),
        (r"e-?mail", "email"),
        (r"phone|mobile|cell", "phone"),
        (r"linkedin", "linkedin"),
        (r"github", "github"),
        (r"website|portfolio|personal (site|url|page)|\bblog\b|other (url|link)", "website"),
        (r"zip|postal", "postal_code"),
        (r"street|address line|^address", "address"),
        (r"^country", "country"),
        (r"^(state|province|region)\b", "state"),
        (r"^(current )?location|^city|where are you (located|based)|current city", "location"),
        (r"current (company|employer)|^(company|employer)( name)?$", "current_company"),
        (r"current (job )?(title|role|position)|^(job )?title$", "current_title"),
        (r"years of (professional |relevant |work )?experience|how many years", "years_experience"),
        (r"school|university|college|institution", "school"),
        (r"\bdegree\b|field of study|major", "degree"),
        (r"graduat", "graduation_year"),
        (r"salary|compensation|pay (expectation|range)|desired pay", "salary_expectation"),
        (r"start date|available to start|earliest (start|available)|when can you start",
         "earliest_start"),
        (r"notice period", "notice_period"),
        (r"sponsor", "needs_sponsorship"),
        (r"authori[sz]ed to work|legally (authorized|allowed|eligible)|work authori[sz]ation|"
         r"right to work|eligible to work", "authorized_to_work"),
        (r"relocat", "willing_to_relocate"),
        (r"how did you (hear|find|learn)|where did you (hear|find|see)|^source$",
         "how_did_you_hear"),
        (r"pronoun", "pronouns"),
        (r"\bgender\b|\bsex\b", "gender"),
        (r"\brace\b|ethnic|hispanic|latin[ox]", "race"),
        (r"veteran", "veteran"),
        (r"disabilit", "disability"),
        (r"resume|résumé|\bcv\b|curriculum vitae", "resume"),
        (r"cover letter", "cover_letter"),
    )
)  # fmt: skip
DECLINE_OPTION = re.compile(
    r"decline|prefer not|don'?t wish|do not wish|not to (say|answer|disclose)|choose not",
    re.IGNORECASE,
)


class NeedsUser(Exception):
    """The application needs something only the user can give: a login, or answers."""

    def __init__(self, message: str, questions: list[str] | None = None) -> None:
        super().__init__(message)
        self.questions = questions or []


@dataclass
class Filled:
    label: str
    value: str
    how: str  # profile | saved answer | drafted | chosen | consent | file


@dataclass
class ApplicationState:
    url: str
    profile: ApplicantProfile
    job_text: str = ""
    filled: list[Filled] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)
    done: set[str] = field(default_factory=set)  # fields already handled, by key

    def summary(self, limit: int = 40) -> str:
        lines = [f"- {f.label}: {f.value}" + (" (drafted by Jamie)" if f.how == "drafted" else "")
                 for f in self.filled[:limit]]  # fmt: skip
        if len(self.filled) > limit:
            lines.append(f"- … and {len(self.filled) - limit} more")
        return "\n".join(lines)


def _yes_no(value: bool | None) -> str | None:
    return None if value is None else ("Yes" if value else "No")


def profile_value(profile: ApplicantProfile, key: str) -> str | None:
    if key in {"needs_sponsorship", "authorized_to_work", "willing_to_relocate"}:
        return _yes_no(getattr(profile, key))
    if key == "preferred_name":
        return profile.preferred_name or profile.first_name or None
    if key in {"full_name", "location"}:
        return getattr(profile, key) or None
    value = getattr(profile, key, "")
    return str(value) if value not in ("", None) else None


def classify(label: str) -> str | None:
    text = label.strip()
    for pattern, key in RULES:
        if pattern.search(text):
            return key
    return None


def best_option(wanted: str, options: list[str]) -> str | None:
    """The option that means ``wanted``: exact, then yes/no and decline, then contains."""
    clean = [o for o in options if o.strip() and not re.match(r"^(select|choose|--)", o, re.I)]
    target = wanted.strip().casefold()
    for option in clean:
        if option.strip().casefold() == target:
            return option
    if target in {"yes", "no"}:
        for option in clean:
            if re.match(rf"^{target}\b", option.strip(), re.IGNORECASE):
                return option
    if wanted == DECLINE or DECLINE_OPTION.search(wanted):
        for option in clean:
            if DECLINE_OPTION.search(option):
                return option
    for option in clean:
        lowered = option.casefold()
        if target and (target in lowered or lowered in target):
            return option
    return None


def _field_text(element: Element) -> str:
    return element.question or element.label


def _key(element: Element) -> str:
    if element.type in {"radio"} or element.role == "radio":
        return f"group:{element.name or element.question}"
    return f"{element.tag}:{element.name}:{element.label}:{element.question}"


class Drafter:
    """Ask the local model, briefly and only from the applicant's own information."""

    def __init__(self, backend, profile: ApplicantProfile, resume_text: str, job_text: str):
        self._backend = backend
        self._profile = profile
        self._resume = resume_text[:6000]
        self._job = job_text[:3000]

    def _context(self) -> str:
        return (
            f"Job posting (excerpt): {self._job}\n\n"
            f"Applicant profile: {json.dumps(self._profile.for_model(), ensure_ascii=False)}\n\n"
            f"Résumé (excerpt): {self._resume}"
        )

    def write(self, question: str, *, long: bool) -> str | None:
        if self._backend is None:
            return None
        words = "120 words" if long else "15 words"
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string", "maxLength": 1500 if long else 200}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        system = (
            "You fill in a job application for the applicant, in their voice. Answer the "
            "question truthfully using only the profile and résumé below. Never invent "
            "employers, dates, numbers, credentials, or personal details. If they do not "
            f"answer the question, reply exactly UNKNOWN. At most {words}.\n\n{self._context()}"
        )
        with constrained_to(schema):
            response = self._backend.chat(
                [Message("system", system), Message("user", f"Question: {question}")],
                temperature=0.3,
                max_tokens=600 if long else 120,
            )
        return _answer(response.content)

    def choose(self, question: str, options: list[str]) -> str | None:
        if self._backend is None or not options:
            return None
        schema = {
            "type": "object",
            "properties": {"answer": {"enum": [*options, "UNKNOWN"]}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        system = (
            "You fill in a job application for the applicant. Pick the option that is true "
            "for them according to the profile and résumé. If they do not say, answer "
            f"UNKNOWN.\n\n{self._context()}"
        )
        with constrained_to(schema):
            response = self._backend.chat(
                [
                    Message("system", system),
                    Message("user", f"Question: {question}\nOptions: {json.dumps(options)}"),
                ],
                temperature=0.0,
                max_tokens=80,
            )
        answer = _answer(response.content)
        return answer if answer in options else None


def _answer(content: str) -> str | None:
    from assistant.agent.protocol import _remove_thinking

    try:
        data = json.loads(_remove_thinking(content))
        answer = str(data.get("answer", "")).strip()
    except (ValueError, AttributeError, TypeError):
        return None
    if not answer or answer.upper().startswith("UNKNOWN"):
        return None
    return answer


class ApplicationFiller:
    """Drive one application in one browser tab."""

    def __init__(
        self,
        tab: BrowserTab,
        profile: ApplicantProfile,
        *,
        drafter: Drafter | None = None,
        progress: Callable[[str], object] = lambda text: None,
        cancelled: Callable[[], bool] = lambda: False,
        pause: float = 0.15,
    ) -> None:
        self.tab = tab
        self.profile = profile
        self.drafter = drafter
        self._progress = progress
        self._cancelled = cancelled
        self._pause = pause

    # -- navigation -------------------------------------------------------------------

    def open(self, url: str) -> tuple[ApplicationState, Snapshot]:
        snapshot = self.tab.open(url)
        state = ApplicationState(url=url, profile=self.profile, job_text=snapshot.text)
        # A job page shows the posting and an Apply button; some flows ask twice
        # (Workday: Apply, then Apply Manually). A page with a real form is the form.
        for _ in range(3):
            self._check_login(snapshot)
            if len(self._form_fields(snapshot)) >= 3:
                break
            entry = self._button(snapshot, ENTRY, allow_final=False)
            if entry is None:
                break
            self._progress(f"Opening the application ({entry.label})")
            snapshot = self.tab.click(entry.ref)
        self._check_login(snapshot)
        return state, snapshot

    def fill_until_final(self, state: ApplicationState, snapshot: Snapshot) -> Snapshot:
        """Fill every page; return the snapshot whose next step is the final button."""
        for page in range(1, MAX_PAGES + 1):
            if self._cancelled():
                raise InterruptedError("Cancelled.")
            self._progress(f"Filling page {page}")
            snapshot = self.fill_page(state, snapshot)
            if SUCCESS.search(snapshot.text):
                return snapshot  # some flows submit without a separate final step
            final = self._button(snapshot, FINAL, allow_final=True)
            following = self._button(snapshot, NEXT, allow_final=False)
            order = {e.ref: i for i, e in enumerate(snapshot.elements)}
            if following is not None and (final is None or order[final.ref] > order[following.ref]):
                missing = self._missing_required(snapshot, state)
                if missing:
                    raise NeedsUser("The application needs answers only you can give.", missing)
                before = snapshot.url, snapshot.text
                snapshot = self.tab.click(following.ref)
                if (snapshot.url, snapshot.text) == before:
                    errors = [e.label for e in snapshot.elements if e.invalid]
                    raise NeedsUser(
                        "The form would not move to the next page.", errors or state.unanswered
                    )
                continue
            if final is not None:
                missing = self._missing_required(snapshot, state)
                if missing:
                    raise NeedsUser("The application needs answers only you can give.", missing)
                return snapshot
            raise NeedsUser(
                "Jamie could not find the application's next or submit button. Open it in "
                "the browser to finish by hand."
            )
        raise NeedsUser("The application has more pages than Jamie expected.")

    def final_button(self, snapshot: Snapshot) -> Element | None:
        return self._button(snapshot, FINAL, allow_final=True)

    @staticmethod
    def _button(snapshot: Snapshot, pattern: re.Pattern, *, allow_final: bool) -> Element | None:
        matches = [
            e
            for e in snapshot.elements
            if e.clickable and pattern.search(e.label.strip()) and (allow_final or not e.final)
        ]
        if not matches:
            return None
        # Prefer real buttons over links, then the last one (forms put actions at the end).
        buttons = [e for e in matches if e.tag == "button" or e.type == "submit"]
        return (buttons or matches)[-1]

    def _check_login(self, snapshot: Snapshot) -> None:
        password = any(e.type == "password" for e in snapshot.elements)
        if password or (not self._form_fields(snapshot) and LOGIN.search(snapshot.title)):
            raise NeedsUser(
                "This site needs you to sign in first. Say “open the browser so I can log "
                f"in” and sign in to {_host(snapshot.url)}, then retry the application."
            )

    @staticmethod
    def _form_fields(snapshot: Snapshot) -> list[Element]:
        return [
            e
            for e in snapshot.elements
            if e.tag in {"input", "textarea", "select"}
            and e.type not in {"submit", "button", "hidden", "search", "image"}
            and not (e.tag == "input" and e.type in {"", "text"} and "search" in e.label.lower())
        ]

    def _missing_required(self, snapshot: Snapshot, state: ApplicationState) -> list[str]:
        missing = []
        groups: dict[str, list[Element]] = {}
        for element in self._form_fields(snapshot):
            if not element.required:
                continue
            if element.type == "radio":
                groups.setdefault(element.name or element.question, []).append(element)
                continue
            if element.type == "checkbox":
                if not element.checked:
                    missing.append(_field_text(element))
                continue
            if element.type == "file":
                if not element.value:
                    missing.append(_field_text(element))
                continue
            if not element.value or (element.tag == "select" and _placeholder(element.value)):
                missing.append(_field_text(element))
        for name, radios in groups.items():
            if not any(r.checked for r in radios):
                missing.append(radios[0].question or name)
        return list(dict.fromkeys(m for m in missing if m))

    # -- filling ------------------------------------------------------------------------

    def fill_page(self, state: ApplicationState, snapshot: Snapshot) -> Snapshot:
        radios: dict[str, list[Element]] = {}
        for element in self._form_fields(snapshot):
            if element.type == "radio" or element.role == "radio":
                radios.setdefault(element.name or element.question, []).append(element)
        handled_groups: set[str] = set()
        for element in list(self._form_fields(snapshot)):
            if self._cancelled():
                raise InterruptedError("Cancelled.")
            key = _key(element)
            if key in state.done:
                continue
            if element.type == "radio" or element.role == "radio":
                group = element.name or element.question
                if group in handled_groups:
                    continue
                handled_groups.add(group)
                snapshot = self._radio_group(state, radios[group]) or snapshot
            else:
                snapshot = self._one(state, element) or snapshot
            state.done.add(key)
            if self._pause:
                time.sleep(self._pause)
        return self.tab.snapshot()

    def _one(self, state: ApplicationState, element: Element) -> Snapshot | None:
        text = _field_text(element)
        if NEVER.search(text):
            if element.required:
                state.unanswered.append(text)
            return None
        if element.type == "file":
            return self._file(state, element)
        if element.type == "checkbox":
            return self._checkbox(state, element)
        if element.tag == "select":
            if element.value and not _placeholder(element.value):
                return None
            return self._select(state, element)
        if element.value:
            return None  # already filled (by the site, or a previous attempt)
        return self._text(state, element)

    def _value(self, state: ApplicationState, element: Element) -> tuple[str | None, str]:
        text = _field_text(element)
        saved = self.profile.answer_for(text)
        if saved:
            return saved, "saved answer"
        key = classify(text)
        if key in {"resume", "cover_letter"}:
            return None, ""
        if key is not None:
            value = profile_value(self.profile, key)
            if value:
                return value, "profile"
        return None, ""

    def _text(self, state: ApplicationState, element: Element) -> Snapshot | None:
        value, how = self._value(state, element)
        text = _field_text(element)
        if value is None and classify(text) == "cover_letter" and element.tag == "textarea":
            if self.profile.write_cover_letters and self.drafter is not None:
                value = self.drafter.write(
                    "Write a short cover letter for this job (3 short paragraphs).", long=True
                )
                how = "drafted"
        if value is None and self.drafter is not None and (element.required or _looks_open(text)):
            value = self.drafter.write(text, long=element.tag == "textarea")
            how = "drafted"
        if value is None:
            if element.required:
                state.unanswered.append(text)
            return None
        if element.role == "combobox" or element.type == "search":
            snapshot = self._combobox(element, value)
        else:
            snapshot = self.tab.fill(element.ref, value)
        state.filled.append(Filled(text, value if len(value) < 160 else value[:157] + "…", how))
        return snapshot

    def _select(self, state: ApplicationState, element: Element) -> Snapshot | None:
        text = _field_text(element)
        options = list(element.options)
        value, how = self._value(state, element)
        choice = best_option(value, options) if value else None
        if choice is None and self.drafter is not None:
            choice = self.drafter.choose(text, [o for o in options if not _placeholder(o)])
            how = "chosen"
        if choice is None:
            if element.required:
                state.unanswered.append(text)
            return None
        state.filled.append(Filled(text, choice, how))
        return self.tab.select(element.ref, choice)

    def _radio_group(self, state: ApplicationState, radios: list[Element]) -> Snapshot | None:
        if any(r.checked for r in radios):
            return None
        question = radios[0].question or radios[0].name
        if NEVER.search(question):
            state.unanswered.append(question)
            return None
        labels = [r.label for r in radios]
        value, how = self._value(state, radios[0])
        choice = best_option(value, labels) if value else None
        if choice is None and self.drafter is not None:
            choice = self.drafter.choose(question, labels)
            how = "chosen"
        if choice is None:
            if any(r.required for r in radios):
                state.unanswered.append(question)
            return None
        target = next(r for r in radios if r.label == choice)
        state.filled.append(Filled(question, choice, how))
        return self.tab.set_checked(target.ref, True)

    def _checkbox(self, state: ApplicationState, element: Element) -> Snapshot | None:
        text = element.label or element.question
        if element.checked:
            return None
        if CONSENT.search(text):
            state.filled.append(Filled(text, "checked", "consent"))
            return self.tab.set_checked(element.ref, True)
        value, how = self._value(state, element)
        if value is None and element.question:
            value = self.profile.answer_for(element.question)
            how = "saved answer"
        if value and value.strip().casefold() in {"yes", "true", element.label.casefold()}:
            state.filled.append(Filled(text, "checked", how))
            return self.tab.set_checked(element.ref, True)
        if element.required:
            state.unanswered.append(text)
        return None

    def _file(self, state: ApplicationState, element: Element) -> Snapshot | None:
        if element.value:
            return None
        text = _field_text(element)
        key = classify(text)
        uploaded = any(f.how == "file" and f.value.startswith("résumé") for f in state.filled)
        if key == "cover_letter":
            path = self.profile.cover_letter_path
        elif key == "resume" or not uploaded:
            path = self.profile.resume_path  # an unlabeled file field is the résumé
        else:
            path = None
        if not path:
            if element.required:
                state.unanswered.append(text or "File upload")
            return None
        file = Path(path).expanduser()
        if not file.is_file():
            raise NeedsUser(f"The file in your profile is missing: {path}")
        snapshot = self.tab.upload(element.ref, file)
        kind = "cover letter" if key == "cover_letter" else "résumé"
        state.filled.append(Filled(text or "File", f"{kind}: {file.name}", "file"))
        return snapshot

    def _combobox(self, element: Element, value: str) -> Snapshot:
        """Type into a searchable dropdown and pick the matching suggestion."""

        def act(page):
            locator = self.tab.locator(element.ref)
            locator.click()
            locator.fill(value)
            page.wait_for_timeout(600)
            option = page.get_by_role("option", name=re.compile(re.escape(value[:40]), re.I))
            if option.count():
                option.first.click()
            else:
                first = page.get_by_role("option")
                if first.count():
                    first.first.click()
            return self.tab._snapshot(page)

        return self.tab.run(act)


def _placeholder(value: str) -> bool:
    return not value.strip() or bool(re.match(r"^(select|choose|--|please select)", value, re.I))


def _looks_open(text: str) -> bool:
    return text.rstrip().endswith("?") or bool(
        re.match(r"^(why|what|how|tell us|describe|please (describe|explain|share))", text, re.I)
    )


def _host(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc or url
