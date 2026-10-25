"""macOS notifications sent as Jamie.app; clicking one brings the window forward.

Notifications need an app bundle to send them as, so they work inside Jamie.app and
quietly do nothing for `jamie gui` started from a terminal.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import objc
import UserNotifications as UN
from Foundation import NSBundle, NSObject

from assistant.ui.login_item import BUNDLE_ID

UNNotificationCenterDelegate = objc.protocolNamed("UNUserNotificationCenterDelegate")


class _Delegate(NSObject, protocols=[UNNotificationCenterDelegate]):
    def initWithClick_(self, click):
        self = objc.super(_Delegate, self).init()
        if self is not None:
            self.click = click
        return self

    def userNotificationCenter_willPresentNotification_withCompletionHandler_(
        self, center, notification, handler
    ):
        # Show the banner even while Jamie is the frontmost app.
        handler(
            UN.UNNotificationPresentationOptionBanner
            | UN.UNNotificationPresentationOptionList
            | UN.UNNotificationPresentationOptionSound
        )

    def userNotificationCenter_didReceiveNotificationResponse_withCompletionHandler_(
        self, center, response, handler
    ):
        self.click()
        handler()


class Notifier:
    """Create on the main thread before the app finishes launching."""

    def __init__(self, on_click: Callable[[], object]) -> None:
        self._center = None
        if NSBundle.mainBundle().bundleIdentifier() != BUNDLE_ID:
            return
        self._center = UN.UNUserNotificationCenter.currentNotificationCenter()
        self._delegate = _Delegate.alloc().initWithClick_(on_click)
        self._center.setDelegate_(self._delegate)

    @property
    def available(self) -> bool:
        return self._center is not None

    def ask_permission(self) -> None:
        """macOS asks once; afterwards the answer lives in System Settings › Notifications."""
        if self._center is not None:
            self._center.requestAuthorizationWithOptions_completionHandler_(
                UN.UNAuthorizationOptionAlert | UN.UNAuthorizationOptionSound,
                lambda granted, error: None,
            )

    def send(self, title: str, body: str) -> None:
        if self._center is None:
            return
        content = UN.UNMutableNotificationContent.alloc().init()
        content.setTitle_("Jamie")
        content.setSubtitle_(title)
        content.setBody_(body)
        content.setSound_(UN.UNNotificationSound.defaultSound())
        request = UN.UNNotificationRequest.requestWithIdentifier_content_trigger_(
            uuid4().hex, content, None
        )
        self._center.addNotificationRequest_withCompletionHandler_(request, None)
