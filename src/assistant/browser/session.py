"""A real browser Jamie controls, with its own saved logins.

The browser keeps a persistent profile in Jamie's private data folder, so after the user
logs in once (Jamie opens a visible window for that), later tasks reuse the session.
The file tools can never read that folder, so cookies stay out of the model's reach.

Playwright's synchronous API must stay on the thread that started it, while agent tools
run on worker threads; every browser call is therefore handed to one browser thread.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

from assistant.tools.base import ToolError

DEFAULT_TIMEOUT_MS = 20_000
CALL_TIMEOUT_SECONDS = 180.0
MAX_ELEMENTS = 160
MAX_TEXT = 1_800

# Clicking one of these sends something on the user's behalf or cannot be undone.
FINAL_WORDS = re.compile(
    r"\b(submit|send|place (?:your )?order|pay|purchase|buy|checkout|check out|confirm|"
    r"delete|remove account|transfer|donate|subscribe|unsubscribe|post|publish)\b",
    re.IGNORECASE,
)
FORM_FINAL_WORDS = re.compile(
    r"\b(apply|book|reserve|register|sign ?up|create account|finish|done|complete)\b",
    re.IGNORECASE,
)


class BrowserUnavailable(ToolError):
    """Playwright or a browser is not installed."""


def default_profile_dir() -> Path:
    from assistant.memory import default_memory_path

    return default_memory_path().with_name("browser-profile")


def screenshots_dir() -> Path:
    from assistant.memory import default_memory_path

    return default_memory_path().with_name("screenshots")


@dataclass(frozen=True)
class Element:
    ref: int
    frame: int
    tag: str
    type: str
    label: str
    value: str = ""
    checked: bool | None = None
    required: bool = False
    options: tuple[str, ...] = ()
    in_form: bool = False
    role: str = ""
    name: str = ""
    question: str = ""  # for radio buttons and checkboxes: the question they answer
    invalid: bool = False

    @property
    def clickable(self) -> bool:
        return (
            self.tag in {"a", "button"}
            or self.type in {"submit", "button", "image"}
            or self.role in {"button", "link", "menuitem"}
        )

    @property
    def final(self) -> bool:
        """Clicking it would submit, send, buy, or delete something."""
        if not self.clickable:
            return False
        if FINAL_WORDS.search(self.label):
            return True
        submit_like = self.type == "submit" or (self.tag == "button" and self.in_form)
        return submit_like and bool(FORM_FINAL_WORDS.search(self.label))

    def describe(self) -> str:
        kind = self.tag if self.tag != "input" else f"input {self.type or 'text'}"
        line = f"[{self.ref}] {kind} {self.label!r}" if self.label else f"[{self.ref}] {kind}"
        if self.question:
            line += f" for {self.question!r}"
        if self.required:
            line += " (required)"
        if self.options:
            shown = ", ".join(self.options[:12])
            line += f" options: {shown}{', …' if len(self.options) > 12 else ''}"
        if self.checked is not None:
            line += " [checked]" if self.checked else " [unchecked]"
        elif self.tag in {"input", "textarea", "select"} and self.type not in {
            "submit",
            "button",
            "file",
        }:
            line += f" = {self.value!r}"
        if self.final:
            line += " (final: use browser_submit)"
        return line


@dataclass(frozen=True)
class Snapshot:
    url: str
    title: str
    text: str
    elements: tuple[Element, ...]

    def describe(self) -> str:
        lines = [f"Page: {self.title} | {self.url}"]
        if self.text:
            lines.append(f"Text: {self.text}")
        lines.append("Elements:")
        lines += [element.describe() for element in self.elements] or ["(none)"]
        return "\n".join(lines)


# Runs in each frame: number the visible controls and describe them.
COLLECT = r"""
(start) => {
  const clip = (t, n) => (t || "").replace(/\s+/g, " ").trim().slice(0, n);
  // Numbers stay with their element for as long as it exists, so a field that appears
  // while Jamie fills a form never shifts the numbers of the others.
  const visible = (el) => {
    if (el.type === "file") return true;  // often styled away behind a button
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
  };
  const nearbyText = (el) => {
    let node = el;
    for (let depth = 0; depth < 3 && node; depth++) {
      let sibling = node.previousElementSibling;
      while (sibling) {
        const text = clip(sibling.innerText, 120);
        if (text) return text;
        sibling = sibling.previousElementSibling;
      }
      node = node.parentElement;
      if (node && node.matches("fieldset") && node.querySelector("legend"))
        return clip(node.querySelector("legend").innerText, 120);
    }
    return "";
  };
  // A label that wraps its control would otherwise include the control's own text,
  // such as every option of a select.
  const labelText = (label) => {
    const copy = label.cloneNode(true);
    for (const n of copy.querySelectorAll("input,select,textarea,button")) n.remove();
    return clip(copy.textContent, 120);
  };
  const labelOf = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria) return clip(aria, 120);
    const by = el.getAttribute("aria-labelledby");
    if (by) {
      const text = by.split(/\s+/)
        .map((id) => document.getElementById(id)?.innerText || "").join(" ");
      if (clip(text, 120)) return clip(text, 120);
    }
    if (el.labels && el.labels.length && labelText(el.labels[0])) return labelText(el.labels[0]);
    const wrap = el.closest("label");
    if (wrap && labelText(wrap)) return labelText(wrap);
    const tag = el.tagName.toLowerCase();
    if (tag === "a" || tag === "button" || el.getAttribute("role"))
      return clip(el.innerText || el.value || el.title, 120);
    if (el.type === "submit" || el.type === "button") return clip(el.value, 120);
    return clip(el.placeholder, 120) || nearbyText(el) || clip(el.name || el.title, 120);
  };
  // The question a radio button or checkbox answers ("Yes" alone means little).
  const questionOf = (el) => {
    const set = el.closest("fieldset");
    if (set && set.querySelector("legend")) return clip(set.querySelector("legend").innerText, 200);
    const group = el.closest("[role=radiogroup],[role=group]");
    if (group) {
      const aria = group.getAttribute("aria-label");
      if (aria) return clip(aria, 200);
      const by = group.getAttribute("aria-labelledby");
      const text = by && document.getElementById(by)?.innerText;
      if (text) return clip(text, 200);
    }
    const own = labelOf(el);
    let node = el.parentElement;
    for (let depth = 0; depth < 5 && node; depth++) {
      let sibling = node.previousElementSibling;
      while (sibling) {
        const text = clip(sibling.innerText, 200);
        if (text && text !== own) return text;
        sibling = sibling.previousElementSibling;
      }
      node = node.parentElement;
    }
    return "";
  };
  const selector = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea",
    "[role=button]", "[role=link]", "[role=checkbox]", "[role=radio]", "[role=combobox]",
    "[role=option]", "[role=tab]", "[role=menuitem]", "[role=switch]", "[contenteditable=true]",
  ].join(",");
  const items = [];
  let n = start;
  for (const el of document.querySelectorAll(selector)) {
    if (!visible(el) || el.disabled) continue;
    let ref = Number(el.getAttribute("data-jamie-ref"));
    if (!ref) {
      n += 1;
      ref = n;
      el.setAttribute("data-jamie-ref", String(n));
    }
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || (tag === "select" ? "select" : "")).toLowerCase();
    const checkable = type === "checkbox" || type === "radio";
    let value = "";
    if (type === "password") value = el.value ? "••••" : "";
    else if (tag === "select") value = el.selectedOptions[0]?.text?.trim() || "";
    else if ("value" in el && type !== "file") value = clip(el.value, 200);
    else if (type === "file") value = el.files && el.files.length ? el.files[0].name : "";
    items.push({
      ref, tag, type, role: el.getAttribute("role") || "",
      name: el.getAttribute("name") || "",
      question: checkable || el.getAttribute("role") === "radio" ? questionOf(el) : "",
      invalid: el.getAttribute("aria-invalid") === "true",
      label: labelOf(el),
      value,
      checked: checkable ? el.checked
        : (el.getAttribute("aria-checked") === null ? null
          : el.getAttribute("aria-checked") === "true"),
      required: el.required || el.getAttribute("aria-required") === "true",
      options: tag === "select"
        ? [...el.options].map((o) => clip(o.text, 60)).filter(Boolean) : [],
      in_form: !!el.closest("form"),
    });
  }
  const headings = [...document.querySelectorAll("h1,h2,h3")]
    .map((h) => clip(h.innerText, 100)).filter(Boolean);
  const body = clip(document.body ? document.body.innerText : "", 4000);
  return { items, next: n, title: document.title, headings, body };
}
"""


class _BrowserThread:
    """Run every Playwright call on one thread."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._loop, name="jamie-browser", daemon=True)
        self._thread.start()

    def call(self, function: Callable[..., Any], *args: Any, timeout: float = CALL_TIMEOUT_SECONDS):
        if threading.current_thread() is self._thread:
            return function(*args)
        future: Future = Future()
        self._queue.put((function, args, future))
        return future.result(timeout)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            function, args, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10)


def _executable() -> str | None:
    explicit = os.environ.get("JAMIE_BROWSER_PATH")
    if explicit:
        return explicit
    return None


class BrowserSession:
    """One browser profile shared by every task; each task works in its own tab."""

    def __init__(
        self,
        profile_dir: Path | None = None,
        *,
        headless: bool = True,
        channel: str | None = None,
        allow_file_urls: bool = False,
    ) -> None:
        self.profile_dir = profile_dir if profile_dir is not None else default_profile_dir()
        self._headless = headless
        self._channel = channel
        self.allow_file_urls = allow_file_urls
        self._thread = _BrowserThread()
        self._playwright = None
        self._context = None
        self._generation = 0  # changes when the browser restarts; old pages are gone
        self._default: BrowserTab | None = None

    # -- lifecycle (browser thread) -----------------------------------------------------

    def _launch(self, headless: bool) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserUnavailable(
                "The browser needs Playwright: pip install playwright && playwright install "
                "chromium"
            ) from exc
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        self.profile_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        options: dict[str, Any] = {
            "headless": headless,
            "viewport": {"width": 1280, "height": 900},
            "accept_downloads": False,
        }
        executable = _executable()
        if executable:
            options["executable_path"] = executable
        elif self._channel:
            options["channel"] = self._channel
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                str(self.profile_dir), **options
            )
        except Exception as exc:  # noqa: BLE001 - Playwright raises its own Error type
            if self._channel and not executable:
                self._channel = None  # Chrome is not installed; use Playwright's Chromium
                return self._launch(headless)
            raise BrowserUnavailable(
                f"Cannot start the browser ({str(exc).splitlines()[0]}). Run: playwright "
                "install chromium"
            ) from exc
        self._context.set_default_timeout(DEFAULT_TIMEOUT_MS)
        self._headless = headless
        self._generation += 1

    def _context_for(self, headless: bool | None = None):
        wanted = self._headless if headless is None else headless
        if self._context is None or wanted != self._headless:
            self._launch(wanted)
        return self._context

    def call(self, function: Callable[..., Any], *args: Any) -> Any:
        return self._thread.call(function, *args)

    def tab(self) -> BrowserTab:
        """A new tab for one task; nothing starts until it is used."""
        return BrowserTab(self)

    @property
    def default_tab(self) -> BrowserTab:
        if self._default is None:
            self._default = self.tab()
        return self._default

    def show(self, tab: BrowserTab | None = None) -> str:
        """Open a visible window (to log in, or to watch); later tasks reuse the logins."""
        tab = tab or self.default_tab

        def visible():
            url = tab._page.url if tab._page is not None and not tab._page.is_closed() else ""
            self._context_for(headless=False)
            page = tab._ensure()
            if url.startswith(("http://", "https://")):
                page.goto(url, wait_until="domcontentloaded")
            page.bring_to_front()
            return page.url

        return self.call(visible)

    def hide(self) -> None:
        self.call(lambda: self._context_for(headless=True))

    def close(self) -> None:
        def stop() -> None:
            if self._context is not None:
                self._context.close()
            if self._playwright is not None:
                self._playwright.stop()
            self._context = self._playwright = None

        try:
            self.call(stop)
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
        self._thread.stop()

    def check_url(self, url: str) -> str:
        url = url.strip()
        if url.startswith("file://") and self.allow_file_urls:
            return url
        if not re.match(r"^https?://[^\s/]+", url):
            raise ToolError("Open an http or https address.")
        return url


class BrowserTab:
    """One page and its latest snapshot; the element numbers belong to that snapshot."""

    def __init__(self, session: BrowserSession) -> None:
        self.session = session
        self._page = None
        self._generation = -1
        self._frames: dict[int, Any] = {}
        self._next_ref = 0  # only grows, so numbers are unique across frames and pages
        self.last_snapshot: Snapshot | None = None

    def _ensure(self):
        context = self.session._context_for()
        if (
            self._page is None
            or self._page.is_closed()
            or self._generation != self.session._generation
        ):
            self._page = context.new_page()
            self._generation = self.session._generation
            self._frames = {}
        return self._page

    def run(self, function: Callable[..., Any], *args: Any) -> Any:
        """Run ``function(page, *args)`` on the browser thread."""
        return self.session.call(lambda: function(self._ensure(), *args))

    def close(self) -> None:
        def shut():
            if self._page is not None and not self._page.is_closed():
                self._page.close()
            self._page = None

        try:
            self.session.call(shut)
        except Exception:  # noqa: BLE001 - best effort
            pass

    def show(self) -> str:
        return self.session.show(self)

    def open(self, url: str) -> Snapshot:
        url = self.session.check_url(url)

        def go(page):
            page.goto(url, wait_until="domcontentloaded")
            _settle(page)
            return self._snapshot(page)

        return self.run(go)

    def snapshot(self) -> Snapshot:
        return self.run(self._snapshot)

    def _snapshot(self, page) -> Snapshot:
        elements: list[Element] = []
        self._frames = {}
        texts: list[str] = []
        title = ""
        start = self._next_ref
        for index, frame in enumerate(page.frames):
            try:
                data = frame.evaluate(COLLECT, start)
            except Exception:  # noqa: BLE001 - cross-origin or detached frames
                continue
            if index == 0:
                title = data.get("title") or ""
            start = data["next"]
            for item in data["items"]:
                self._frames[item["ref"]] = frame
                elements.append(
                    Element(
                        ref=item["ref"],
                        frame=index,
                        tag=item["tag"],
                        type=item["type"],
                        label=item["label"],
                        value=item["value"],
                        checked=item["checked"],
                        required=bool(item["required"]),
                        options=tuple(item["options"]),
                        in_form=bool(item["in_form"]),
                        role=item.get("role", ""),
                        name=item.get("name", ""),
                        question=item.get("question", ""),
                        invalid=bool(item.get("invalid")),
                    )
                )
            heading = " / ".join(data.get("headings", [])[:6])
            body = data.get("body", "")
            if heading or body:
                texts.append(f"{heading} :: {body}" if heading else body)
        self._next_ref = start
        text = " | ".join(texts)[:MAX_TEXT]
        snapshot = Snapshot(page.url, title, text, tuple(elements[:MAX_ELEMENTS]))
        self.last_snapshot = snapshot
        return snapshot

    def element(self, ref: int) -> Element:
        snapshot = self.last_snapshot
        found = next((e for e in snapshot.elements if e.ref == ref), None) if snapshot else None
        if found is None:
            raise ToolError(f"No element [{ref}] on the page. Take a new browser_snapshot.")
        return found

    def locator(self, ref: int):
        """Playwright locator for an element of the latest snapshot (browser thread only)."""
        frame = self._frames.get(ref)
        if frame is None:
            raise ToolError(f"No element [{ref}] on the page. Take a new browser_snapshot.")
        return frame.locator(f'[data-jamie-ref="{ref}"]').first

    def click(self, ref: int) -> Snapshot:
        def act(page):
            before = len(page.context.pages)
            self.locator(ref).click()
            _settle(page)
            pages = page.context.pages
            if len(pages) > before and not pages[-1].is_closed():
                # The link opened a new tab (an external application form, say): follow it.
                page = self._page = pages[-1]
                _settle(page)
            return self._snapshot(page)

        return self.run(act)

    def fill(self, ref: int, text: str, *, press_enter: bool = False) -> Snapshot:
        def act(page):
            locator = self.locator(ref)
            if locator.get_attribute("contenteditable") == "true":
                locator.click()
                page.keyboard.insert_text(text)
            else:
                locator.fill(text)
            if press_enter:
                locator.press("Enter")
                _settle(page)
            return self._snapshot(page)

        return self.run(act)

    def select(self, ref: int, option: str) -> Snapshot:
        def act(page):
            locator = self.locator(ref)
            labels = locator.evaluate("s => [...s.options].map(o => o.text.trim())")
            match = _closest(option, labels)
            if match is None:
                raise ToolError(f"No option like {option!r}. Options: {labels[:15]}")
            locator.select_option(label=match)
            return self._snapshot(page)

        return self.run(act)

    def set_checked(self, ref: int, checked: bool) -> Snapshot:
        def act(page):
            self.locator(ref).set_checked(checked, force=True)
            return self._snapshot(page)

        return self.run(act)

    def upload(self, ref: int, path: Path) -> Snapshot:
        def act(page):
            self.locator(ref).set_input_files(str(path))
            _settle(page)
            return self._snapshot(page)

        return self.run(act)

    def scroll(self, down: bool = True) -> Snapshot:
        def act(page):
            page.mouse.wheel(0, 800 if down else -800)
            page.wait_for_timeout(300)
            return self._snapshot(page)

        return self.run(act)

    def back(self) -> Snapshot:
        def act(page):
            page.go_back(wait_until="domcontentloaded")
            _settle(page)
            return self._snapshot(page)

        return self.run(act)

    def screenshot(self, name: str = "page", *, full_page: bool = True) -> Path:
        folder = screenshots_dir()
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:40] or "page"
        path = folder / f"{stamp}-{safe}.jpg"

        def shoot(page):
            page.screenshot(path=str(path), full_page=full_page, type="jpeg", quality=70)
            return path

        return self.run(shoot)

    def url(self) -> str:
        return self.run(lambda page: page.url)


def _settle(page) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=10_000)
        page.wait_for_load_state("networkidle", timeout=4_000)
    except Exception:  # noqa: BLE001 - busy pages never go idle; carry on
        pass


def _closest(wanted: str, options: list[str]) -> str | None:
    target = wanted.casefold().strip()
    for option in options:
        if option.casefold().strip() == target:
            return option
    for option in options:
        if target and (target in option.casefold() or option.casefold().strip() in target):
            if option.strip():
                return option
    return None


def browser_installed() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True
