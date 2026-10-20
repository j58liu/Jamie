"""A global keyboard shortcut through Carbon's RegisterEventHotKey.

Unlike key monitoring, a registered hot key needs no Accessibility permission: macOS
delivers only that key combination to Jamie. macOS 15 rejects shortcuts whose only
modifier is Option, so the default uses Control and Option.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass


def _four_char(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("event_class", ctypes.c_uint32), ("event_kind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
KEYBOARD_EVENTS = _four_char("keyb")
HOT_KEY_PRESSED = 5
CONTROL, OPTION, COMMAND, SHIFT = 0x1000, 0x0800, 0x0100, 0x0200
KEY_J = 38  # kVK_ANSI_J, the J key's position on any layout
SHORTCUT = (KEY_J, CONTROL | OPTION)
SHORTCUT_LABEL = "⌃⌥J"


@dataclass
class HotKey:
    """Keeps the C callback alive for as long as the shortcut is registered."""

    handler: object
    reference: ctypes.c_void_p


def register(callback: Callable[[], object], shortcut: tuple[int, int] = SHORTCUT) -> HotKey:
    """Call ``callback`` on the main thread when the shortcut is pressed anywhere.

    Register from the main thread after AppKit has started the application.
    Raises OSError when macOS refuses, for example because another app owns the keys.
    """
    carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
    carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
    carbon.InstallEventHandler.argtypes = [
        ctypes.c_void_p,
        _HANDLER,
        ctypes.c_ulong,
        ctypes.POINTER(_EventTypeSpec),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    carbon.InstallEventHandler.restype = ctypes.c_int32
    carbon.RegisterEventHotKey.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        _EventHotKeyID,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    carbon.RegisterEventHotKey.restype = ctypes.c_int32

    def handle(call: object, event: object, data: object) -> int:
        try:
            callback()
        except Exception:  # noqa: BLE001 - an exception must not cross into Carbon
            pass
        return 0

    handler = _HANDLER(handle)
    target = carbon.GetApplicationEventTarget()
    kinds = _EventTypeSpec(KEYBOARD_EVENTS, HOT_KEY_PRESSED)
    status = carbon.InstallEventHandler(target, handler, 1, ctypes.byref(kinds), None, None)
    if status:
        raise OSError(f"macOS could not listen for the shortcut (error {status}).")
    reference = ctypes.c_void_p()
    key, modifiers = shortcut
    identity = _EventHotKeyID(_four_char("jmie"), 1)
    status = carbon.RegisterEventHotKey(
        key, modifiers, identity, target, 0, ctypes.byref(reference)
    )
    if status:
        raise OSError(f"macOS refused the shortcut {SHORTCUT_LABEL} (error {status}).")
    return HotKey(handler, reference)
