"""Randomized but realistic little worlds for generating training trajectories.

Each world has a home folder with documents, an inbox, a calendar, reminders, Jamie's
own task store, and an applicant profile. Tools run for real against it (with fake
providers), so every observation in the training data is exactly what Jamie's tools
return at runtime.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

FIRST = [
    "Alex",
    "Priya",
    "Sam",
    "Jordan",
    "Mei",
    "Diego",
    "Fatima",
    "Noah",
    "Lena",
    "Omar",
    "Grace",
    "Hiro",
    "Isabel",
    "Kofi",
    "Maya",
    "Ravi",
    "Sofia",
    "Tom",
    "Yuki",
    "Zara",
]
LAST = [
    "Kim",
    "Shah",
    "Lee",
    "Garcia",
    "Chen",
    "Okafor",
    "Novak",
    "Haddad",
    "Brown",
    "Rossi",
    "Silva",
    "Tanaka",
    "Dubois",
    "Mensah",
    "Patel",
    "Weber",
    "Costa",
    "Murphy",
]
COMPANIES = [
    "Northwind",
    "Bluefin Labs",
    "Cedar Health",
    "Orbit Analytics",
    "Lumen Foods",
    "Harbor Bank",
    "Pinecrest",
    "Quill Studio",
    "Vertex Robotics",
    "Marigold",
]
TOPICS = [
    "budget review",
    "launch plan",
    "hiring sync",
    "quarterly goals",
    "design review",
    "vendor contract",
    "onboarding",
    "roadmap",
    "customer feedback",
    "pricing update",
]
ZONES = ["America/Los_Angeles", "America/New_York", "Europe/London", "America/Chicago"]
CITIES = [("Oakland", "CA", "United States"), ("Austin", "TX", "United States"),
          ("Toronto", "ON", "Canada"), ("Seattle", "WA", "United States"),
          ("London", "", "United Kingdom"), ("Chicago", "IL", "United States")]  # fmt: skip

DOCUMENTS = [
    (
        "lease",
        "Documents/Home/Lease Renewal {year}.txt",
        "Lease renewal for {street}. Monthly rent increases from ${old} to ${new}. "
        "Sign and return by {due}. The new term runs 12 months.",
        "rent goes from ${old} to ${new} a month, and you need to sign and return it by {due}",
    ),
    (
        "insurance",
        "Documents/Insurance/Car insurance {year}.txt",
        "Policy {code}. Annual premium ${new}. Renewal date {due}. Deductible ${old}.",
        "the annual premium is ${new}, the deductible is ${old}, and it renews on {due}",
    ),
    (
        "trip",
        "Documents/Travel/{city} trip.txt",
        "Flight to {city} departs {due} at 07:45 from gate {code}. Hotel confirmation "
        "number {code2}. Checkout is at 11:00.",
        "your flight to {city} leaves {due} at 07:45 (gate {code}), and the hotel "
        "confirmation number is {code2}",
    ),
    (
        "recipe",
        "Documents/Recipes/{dish}.txt",
        "{dish}. Serves {old_small}. Bake at 180C for {new_small} minutes. Needs flour, "
        "eggs, butter, and sugar.",
        "it serves {old_small} and bakes at 180°C for {new_small} minutes",
    ),
    (
        "notes",
        "Documents/Work/{topic} notes.md",
        "# {topic_title}\nDecision: ship on {due}. Owner: {person}. Open question: "
        "the {topic} budget is ${new}.",
        "the decision is to ship on {due}, {person} owns it, and the open question is the "
        "${new} budget",
    ),
]
DISHES = ["Banana bread", "Lemon cake", "Focaccia", "Apple pie", "Shortbread"]
STREETS = ["14 Elm Street", "221 Pine Avenue", "9 Harbor Road", "77 Maple Court"]


@dataclass
class Doc:
    kind: str
    path: Path  # absolute
    rel: str  # relative to home
    text: str
    gist: str


@dataclass
class World:
    rng: random.Random
    root: Path
    tz: ZoneInfo
    now: datetime
    user_first: str
    user_last: str
    city: tuple[str, str, str]
    docs: list[Doc] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    reminders: list[dict] = field(default_factory=list)
    people: list[tuple[str, str]] = field(default_factory=list)

    @property
    def home(self) -> Path:
        return self.root / "home"

    def short(self, path: Path) -> str:
        return "~/" + path.relative_to(self.home).as_posix()

    def day(self, offset: int, hour: int, minute: int = 0) -> datetime:
        base = (self.now + timedelta(days=offset)).date()
        return datetime(base.year, base.month, base.day, hour, minute, tzinfo=self.tz)


def _person(rng: random.Random) -> tuple[str, str]:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    return f"{first} {last}", f"{first.lower()}.{last.lower()}@example.com"


def build_world(rng: random.Random, root: Path) -> World:
    tz = ZoneInfo(rng.choice(ZONES))
    now = datetime(2026, rng.randint(1, 12), rng.randint(1, 26), rng.randint(8, 18),
                   rng.choice([0, 15, 30, 45]), tzinfo=tz)  # fmt: skip
    world = World(
        rng=rng,
        root=root,
        tz=tz,
        now=now,
        user_first=rng.choice(FIRST),
        user_last=rng.choice(LAST),
        city=rng.choice(CITIES),
    )
    home = world.home
    for folder in ("Documents", "Downloads", "Desktop", "Documents/Work"):
        (home / folder).mkdir(parents=True, exist_ok=True)
    world.people = [_person(rng) for _ in range(4)]
    for kind, path_template, text_template, gist_template in rng.sample(DOCUMENTS, 3):
        values = {
            "year": now.year,
            "street": rng.choice(STREETS),
            "old": rng.choice([1200, 1450, 1800, 500, 250]),
            "new": rng.choice([1350, 1600, 1950, 2400, 900]),
            "due": (now + timedelta(days=rng.randint(5, 40))).strftime("%B %-d"),
            "code": f"{rng.choice('ABCDEFG')}{rng.randint(10, 60)}",
            "code2": f"{rng.randint(100000, 999999)}",
            "city": rng.choice(["Lisbon", "Denver", "Kyoto", "Montreal", "Mexico City"]),
            "dish": rng.choice(DISHES),
            "old_small": rng.randint(4, 12),
            "new_small": rng.randint(25, 60),
            "topic": rng.choice(TOPICS),
            "person": world.people[0][0],
        }
        values["topic_title"] = values["topic"].capitalize()
        rel = path_template.format(**values)
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = text_template.format(**values)
        path.write_text(text + "\n")
        world.docs.append(Doc(kind, path, rel, text, gist_template.format(**values)))
    resume = home / "Documents" / f"{world.user_first} {world.user_last} Resume.txt"
    resume.write_text(
        f"{world.user_first} {world.user_last}\n{world.user_first.lower()}@example.com | "
        f"+1 555 01{rng.randint(10, 99)} | {world.city[0]}\nSoftware engineer at "
        f"{rng.choice(COMPANIES)} since 2021. Python, SQL, data pipelines.\n"
    )
    world.docs.append(Doc("resume", resume, resume.relative_to(home).as_posix(), "", "résumé"))
    (home / "Downloads" / "IMG_2041.png").write_bytes(b"png")

    for index, (name, email) in enumerate(world.people):
        topic = rng.choice(TOPICS)
        when = world.day(rng.randint(1, 6), rng.choice([10, 11, 14, 15, 16]))
        world.messages.append(
            {
                "id": f"m{index}{rng.randint(100, 999)}",
                "thread_id": f"t{index}{rng.randint(100, 999)}",
                "sender": f"{name} <{email}>",
                "subject": topic.capitalize(),
                "snippet": f"Could we meet {when.strftime('%A')} at "
                f"{when.strftime('%-I%p').lower()} about the {topic}?",
                "received_at": (now - timedelta(hours=rng.randint(2, 70))).isoformat(),
                "to": ["me@example.com"],
                "body": f"Hi! Could we meet {when.strftime('%A')} at "
                f"{when.strftime('%-I%p').lower()} for 30 minutes about the {topic}? "
                f"{name.split()[0]}",
            }
        )
    for index in range(3):
        start = world.day(rng.randint(0, 3), rng.choice([9, 10, 13, 15]))
        world.events.append(
            {
                "id": f"e{index}{rng.randint(100, 999)}",
                "title": rng.choice(TOPICS).capitalize(),
                "start": start.isoformat(),
                "end": (start + timedelta(minutes=rng.choice([30, 60]))).isoformat(),
            }
        )
    world.reminders = [
        {"id": f"R-{rng.randint(10, 99)}", "title": "Renew passport", "list": "Reminders"},
        {"id": f"R-{rng.randint(100, 199)}", "title": "Call the dentist", "list": "Reminders"},
    ]
    return world
