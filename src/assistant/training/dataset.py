"""Build Jamie's fine-tuning set: python -m assistant.training.dataset --out training/data

Writes train.jsonl and valid.jsonl in the chat format mlx-lm reads ({"messages": [...]});
train with --mask-prompt so only each example's last (assistant) message is learned.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

from assistant.jobs.filler import Drafter
from assistant.jobs.profile import ApplicantProfile
from assistant.routing import (
    SCHEDULE_INSTRUCTIONS,
    SUBJECTS,
    TIMINGS,
    _question,
)
from assistant.training.harness import run_plan
from assistant.training.scenarios import SCENARIOS
from assistant.training.worlds import COMPANIES, FIRST, build_world

# -- routing: one answer letter, read from the first token's probabilities ---------------

SUBJECT_EXAMPLES = {
    "A": ["hi", "thanks!", "what is 17 times 23", "tell me a joke", "explain recursion simply",
          "how do you say thank you in Japanese", "what's the capital of Australia",
          "who wrote Pride and Prejudice", "good morning jamie", "what can you do?"],
    "B": ["find my tax documents", "what's in my Downloads folder",
          "summarize the PDF I downloaded yesterday", "where did I save my resume",
          "move the screenshots on my desktop into a folder", "read my lease and tell me the rent",
          "what was I working on yesterday", "open the budget spreadsheet",
          "search my documents for the word invoice", "copy my CV to the desktop"],
    "C": ["what's on my calendar tomorrow", "did Alex reply to my email",
          "remind me to call mom at 6", "draft a reply to Priya saying yes",
          "am I free Thursday afternoon", "move my 3pm meeting to 4",
          "any important emails today?", "add dentist appointment Friday 10am",
          "what reminders do I have", "send the draft to Sam"],
    "D": ["what's the weather in Paris this weekend", "latest news about electric cars",
          "research the best budget mirrorless cameras", "who won the game last night",
          "what is the current price of bitcoin", "look up flight prices to Tokyo",
          "find reviews of the Framework laptop", "what changed in iOS 26"],
    "E": ["book a table for two at an Italian place on OpenTable", "check my Amazon order status",
          "log in to my bank and download last month's statement",
          "fill out the form at https://forms.example.com/signup",
          "cancel my gym membership on their website", "open the browser so I can log in",
          "renew my library books on the library website",
          "buy the cheapest USB-C cable on that site"],
    "F": ["open Spotify", "copy this to my clipboard: hello", "run my Morning shortcut",
          "open Safari", "show the Downloads folder in Finder", "what's on my clipboard"],
    "G": ["apply to this job https://jobs.lever.co/acme/123",
          "update my job profile salary to 120k",
          "set up job applications from my resume", "apply to these 3 postings for me",
          "my notice period is two weeks for job applications", "show my applicant profile"],
    "H": ["what are you working on?", "stop the background task", "cancel the morning summary",
          "how are my job applications going", "what's scheduled", "is anything waiting for me"],
}  # fmt: skip
TIMING_EXAMPLES = {
    "A": ["what's on my calendar tomorrow", "find my resume", "read Alex's email",
          "open Spotify", "summarize this PDF", "what's the weather tomorrow",
          "draft a reply to Sam", "move my 3pm to 4pm"],
    "B": ["research laptops under $1000 while I work", "compare 10 apartments for me, no rush",
          "go through my downloads and organize everything, let me know when done",
          "look into flights to Lisbon in March and get back to me",
          "take your time and summarize every PDF in my documents",
          "apply to these jobs while I'm in meetings"],
    "C": ["every morning at 8, summarize my email", "every Friday at 5pm list my week",
          "tomorrow at 9am check the news about chips", "every 2 hours check if the site is up",
          "on Monday at 10 remind me what's due this week", "daily at noon check my calendar"],
}  # fmt: skip


def routing_examples(rng: random.Random) -> list[dict]:
    examples = []
    for letter, prompts in SUBJECT_EXAMPLES.items():
        for prompt in prompts:
            question = _question(prompt, "What does this message ask Jamie to work with?", SUBJECTS)
            examples.append(_pair(question[0].content, letter))
    for letter, prompts in TIMING_EXAMPLES.items():
        for prompt in prompts:
            question = _question(prompt, "When should Jamie do this?", TIMINGS)
            examples.append(_pair(question[0].content, letter))
    rng.shuffle(examples)
    return examples


def _pair(user: str, answer: str, system: str | None = None) -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    return {"messages": [*messages, {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}  # fmt: skip


# -- scheduling from plain words ------------------------------------------------------------

SCHEDULES = [
    ("every morning at {h}, {task}", "daily", "", "{hh}:00", "", ""),
    ("{task} every day at {h}", "daily", "", "{hh}:00", "", ""),
    ("every {dayname} at {h}, {task}", "weekly", "{day}", "{hh}:00", "", ""),
    ("every {n} hours, {task}", "every", "", "09:00", "", "{n}h"),
    ("on {date} at {h}, {task}", "once", "", "{hh}:00", "{date}", ""),
]
TASKS = [
    ("summarize my unread email", "Summarize my unread email."),
    ("check the news about AI chips", "Check the news about AI chips."),
    ("list what's on my calendar", "List what's on my calendar."),
    ("check if my package shipped", "Check if my package shipped."),
]
DAYS = [("monday", "mon"), ("tuesday", "tue"), ("friday", "fri"), ("sunday", "sun")]


def schedule_examples(rng: random.Random, count: int = 60) -> list[dict]:
    examples = []
    for _ in range(count):
        template, repeat, day, time, date, interval = rng.choice(SCHEDULES)
        task, instruction = rng.choice(TASKS)
        hour = rng.choice([7, 8, 9, 12, 17, 18])
        dayname, short = rng.choice(DAYS)
        n = rng.choice([2, 3, 6])
        when = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        values = {"h": f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}", "hh": f"{hour:02d}",
                  "task": task, "dayname": dayname, "day": short, "n": n, "date": when}  # fmt: skip
        answer = {
            "repeat": repeat,
            "day": day.format(**values),
            "time": time.format(**values),
            "date": date.format(**values),
            "interval": interval.format(**values),
            "task": instruction,
        }
        system = SCHEDULE_INSTRUCTIONS.format(now="Wednesday 2026-06-10 09:30")
        examples.append(_pair(template.format(**values), json.dumps(answer), system))
    return examples


# -- job answers: from the applicant's own information, or UNKNOWN ----------------------------


def drafter_examples(rng: random.Random, count: int = 40) -> list[dict]:
    examples = []
    for _ in range(count):
        first = rng.choice(FIRST)
        company = rng.choice(COMPANIES)
        profile = ApplicantProfile(
            first_name=first, last_name="Doe", email=f"{first.lower()}@example.com",
            phone="+1 555 0100", city="Austin", current_title="Data Engineer",
            years_experience=str(rng.randint(2, 9)), needs_sponsorship=False,
        )  # fmt: skip
        resume = f"{first} Doe. Data Engineer at {company}. Built Spark pipelines; SQL; Airflow."
        drafter = Drafter(None, profile, resume, f"{rng.choice(COMPANIES)} hiring Data Engineer.")
        system = (
            "You fill in a job application for the applicant, in their voice. Answer the "
            "question truthfully using only the profile and résumé below. Never invent "
            "employers, dates, numbers, credentials, or personal details. If they do not "
            "answer the question, reply exactly UNKNOWN. At most 15 words.\n\n"
            f"{drafter._context()}"
        )
        question, answer = rng.choice(
            [
                ("Current job title", "Data Engineer"),
                ("Years of experience with data engineering", profile.years_experience),
                ("Current employer", company),
                ("What is your GPA?", "UNKNOWN"),
                ("Date of birth", "UNKNOWN"),
                ("Which tools have you used for orchestration?", "Airflow"),
                ("List any security clearances", "UNKNOWN"),
            ]
        )
        examples.append(_pair(f"Question: {question}", json.dumps({"answer": answer}), system))
    return examples


# -- assembly -------------------------------------------------------------------------------


def trajectory_examples(rng: random.Random, worlds: int) -> tuple[list[dict], Counter]:
    examples: list[dict] = []
    problems: Counter = Counter()
    for _ in range(worlds):
        with tempfile.TemporaryDirectory(prefix="jamie-train-") as folder:
            world = build_world(random.Random(rng.random()), Path(folder))
            for scenario, weight in SCENARIOS:
                for _ in range(weight):
                    try:
                        plan = scenario(world)
                        examples += run_plan(world, plan)
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        problems[f"{scenario.__name__}: {type(exc).__name__}: {exc}"[:160]] += 1
    return examples, problems


def build(out: Path, *, seed: int = 7, worlds: int = 40, valid_fraction: float = 0.05) -> dict:
    rng = random.Random(seed)
    trajectories, problems = trajectory_examples(rng, worlds)
    examples = trajectories + routing_examples(rng) + schedule_examples(rng)
    examples += drafter_examples(rng)
    unique = list({json.dumps(e, sort_keys=True): e for e in examples}.values())
    rng.shuffle(unique)
    cut = max(1, int(len(unique) * valid_fraction))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", unique[:cut]), ("train", unique[cut:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"train": len(unique) - cut, "valid": cut, "problems": dict(problems)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("training/data"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--worlds", type=int, default=40)
    arguments = parser.parse_args(argv)
    report = build(arguments.out, seed=arguments.seed, worlds=arguments.worlds)
    print(json.dumps(report, indent=2))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
