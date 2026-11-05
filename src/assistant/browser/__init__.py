"""Jamie's own browser, with saved logins, for websites and job applications."""

from assistant.browser.session import (
    BrowserSession,
    BrowserTab,
    BrowserUnavailable,
    Element,
    Snapshot,
    browser_installed,
    default_profile_dir,
    screenshots_dir,
)
from assistant.browser.tools import BROWSER_TOOLS, register_browser_tools

__all__ = [
    "BROWSER_TOOLS",
    "BrowserSession",
    "BrowserTab",
    "BrowserUnavailable",
    "Element",
    "Snapshot",
    "browser_installed",
    "default_profile_dir",
    "register_browser_tools",
    "screenshots_dir",
]
