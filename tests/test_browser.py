"""Drive a real Chromium against local pages; skipped where no browser is installed."""

import glob
import os
from pathlib import Path

import pytest

from assistant.browser import BrowserSession
from assistant.browser.session import Element
from assistant.security.approvals import CallbackApprovalProvider
from assistant.tools.mac_files import MacPaths
from assistant.tools.registry import ToolRegistry

PAGES = Path(__file__).parent / "fixtures" / "pages"


def _chromium() -> str | None:
    if os.environ.get("JAMIE_BROWSER_PATH"):
        return os.environ["JAMIE_BROWSER_PATH"]
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return found[-1] if found else None


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    pytest.importorskip("playwright")
    data = tmp_path_factory.mktemp("data")
    previous = os.environ.get("JAMIE_MEMORY_PATH")
    os.environ["JAMIE_MEMORY_PATH"] = str(data / "memory.sqlite3")  # screenshots go here
    executable = _chromium()
    if executable is None:
        pytest.skip("No Chromium for Playwright on this machine.")
    os.environ["JAMIE_BROWSER_PATH"] = executable
    session = BrowserSession(tmp_path_factory.mktemp("profile"), channel=None, allow_file_urls=True)
    yield session
    session.close()
    if previous is None:
        os.environ.pop("JAMIE_MEMORY_PATH", None)
    else:
        os.environ["JAMIE_MEMORY_PATH"] = previous


def page(name: str) -> str:
    return (PAGES / name).as_uri()


def test_snapshot_numbers_controls_with_their_labels(browser):
    tab = browser.tab()
    snapshot = tab.open(page("shop.html"))
    labels = {e.label: e for e in snapshot.elements}
    assert snapshot.title == "Corner Bakery"
    assert labels["Your name"].required and labels["Your name"].tag == "input"
    assert labels["Size"].options == ("Small", "Large (serves 12)")
    assert labels["Add candles"].checked is False
    assert labels["Message on the cake"].tag == "textarea"  # label from nearby text
    assert labels["Photo to print"].type == "file"  # hidden file inputs still count
    assert labels["Place order"].final and not labels["Show options"].final
    assert labels["Coupon"].frame == 1  # controls inside frames are numbered too
    assert "Order a cake" in snapshot.text
    tab.close()


def test_typing_selecting_and_checking_update_the_page(browser):
    tab = browser.tab()
    snapshot = tab.open(page("shop.html"))
    ref = {e.label: e.ref for e in snapshot.elements}
    tab.fill(ref["Your name"], "Sam")
    tab.select(ref["Size"], "large")
    tab.set_checked(ref["Add candles"], True)
    snapshot = tab.fill(ref["Coupon"], "CAKE10")
    values = {e.label: e for e in snapshot.elements}
    assert values["Your name"].value == "Sam"
    assert values["Size"].value == "Large (serves 12)"
    assert values["Add candles"].checked is True
    assert values["Coupon"].value == "CAKE10"
    tab.close()


def registry_with(browser, approve, tmp_path):
    from assistant.browser.tools import register_browser_tools

    asked = []

    def answer(request):
        asked.append(request)
        return approve

    registry = ToolRegistry(approvals=CallbackApprovalProvider(answer))
    register_browser_tools(registry, browser, MacPaths([str(tmp_path)]))
    return registry, asked


def test_tools_refuse_final_clicks_and_passwords(browser, tmp_path):
    registry, asked = registry_with(browser, True, tmp_path)
    opened = registry.execute("browser_open", {"url": page("shop.html")})
    assert opened.success and "(final: use browser_submit)" in opened.output
    lines = {line.split("] ", 1)[1].split(" '", 1)[1].split("'")[0]: line
             for line in opened.output.splitlines() if line.startswith("[")}  # fmt: skip
    ref = lambda label: int(lines[label][1:].split("]")[0])  # noqa: E731

    click = registry.execute("browser_click", {"ref": ref("Place order")})
    assert not click.success and "browser_submit" in click.output
    typed = registry.execute("browser_type", {"ref": ref("Account password"), "text": "x"})
    assert not typed.success and "never types passwords" in typed.output
    shown = registry.execute("browser_click", {"ref": ref("Show options")})
    assert shown.success and "More options" in shown.output
    assert asked == []  # ordinary browsing never asks


def test_submit_asks_and_is_bound_to_the_page_and_button(browser, tmp_path):
    registry, asked = registry_with(browser, True, tmp_path)
    opened = registry.execute("browser_open", {"url": page("shop.html")}).output
    line = next(line for line in opened.splitlines() if "'Place order'" in line)
    ref = int(line[1:].split("]")[0])
    name = next(line for line in opened.splitlines() if "'Your name'" in line)
    registry.execute("browser_type", {"ref": int(name[1:].split("]")[0]), "text": "Sam"})

    wrong = registry.execute(
        "browser_submit",
        {"ref": ref, "button_text": "Delete account", "page_url": page("shop.html")},
    )
    assert not wrong.success and "nothing was clicked" in wrong.output
    done = registry.execute(
        "browser_submit",
        {"ref": ref, "button_text": "Place order", "page_url": page("shop.html")},
    )
    assert done.success and "Thanks, your order is in!" in done.output
    assert len(asked) == 2 and asked[0].tool == "browser_submit"

    denied, _ = registry_with(browser, False, tmp_path)
    denied.execute("browser_open", {"url": page("shop.html")})
    result = denied.execute(
        "browser_submit", {"ref": ref, "button_text": "Place order", "page_url": page("shop.html")}
    )
    assert result.denied


def test_uploads_only_send_allowed_files(browser, tmp_path):
    registry, asked = registry_with(browser, True, tmp_path)
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4")
    opened = registry.execute("browser_open", {"url": page("shop.html")}).output
    line = next(line for line in opened.splitlines() if "Photo to print" in line)
    ref = int(line[1:].split("]")[0])
    uploaded = registry.execute("browser_upload", {"ref": ref, "path": str(resume)})
    assert uploaded.success and asked[-1].tool == "browser_upload"
    outside = registry.execute("browser_upload", {"ref": ref, "path": "/etc/hosts"})
    assert not outside.success and "not allowed" in outside.output


def test_open_refuses_non_web_addresses(tmp_path):
    session = BrowserSession(tmp_path, channel=None)
    try:
        with pytest.raises(Exception, match="http or https"):
            session.check_url("file:///etc/passwd")
        with pytest.raises(Exception, match="http or https"):
            session.check_url("javascript:alert(1)")
    finally:
        session.close()


@pytest.mark.parametrize(
    ("element", "final"),
    [
        (Element(1, 0, "button", "submit", "Submit application", in_form=True), True),
        (Element(1, 0, "a", "", "Apply for this job"), False),
        (Element(1, 0, "button", "", "Apply", in_form=True), True),
        (Element(1, 0, "button", "button", "Next"), False),
        (Element(1, 0, "button", "", "Pay $40.00"), True),
        (Element(1, 0, "input", "text", "Send to"), False),
    ],
)
def test_final_buttons_are_recognised(element, final):
    assert element.final is final
