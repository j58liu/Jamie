"""Keep Jamie running with its window closed, like other Mac apps that stay in the Dock."""

from __future__ import annotations

import threading
from typing import Any


def hide_instead_of_closing(window: Any, quitting: threading.Event) -> None:
    """The close button hides the window; quitting (⌘Q, /quit, the Dock menu) closes it."""

    def closing() -> bool:
        if quitting.is_set():
            return True
        window.hide()
        return False  # pywebview cancels the close

    window.events.closing += closing


def handle_reopen_and_quit(window: Any, quitting: threading.Event) -> None:
    """Show the window on a Dock click, and mark a real quit before windows close.

    pywebview's app delegate sends both the close button and ⌘Q through the same
    closing event, so the quit has to be flagged first. Call before webview.start.
    """
    import objc
    from webview.platforms.cocoa import BrowserView

    base = BrowserView.AppDelegate

    class JamieAppDelegate(base):
        def applicationShouldTerminate_(self, app):
            quitting.set()
            return objc.super(JamieAppDelegate, self).applicationShouldTerminate_(app)

        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, visible):
            window.show()
            return True

    BrowserView.AppDelegate = JamieAppDelegate


def toggle_window(window: Any) -> None:
    """Hide Jamie when it is in front; otherwise bring its window forward. Main thread."""
    from AppKit import NSApplication

    app = NSApplication.sharedApplication()
    if app.isActive() and app.keyWindow() is not None:
        app.hide_(None)
    else:
        app.unhide_(None)
        window.show()  # also activates Jamie
