"""Linux Qt/X11 compatibility defaults for frozen RHO Planner builds.

Runs before main.py imports PySide6.  The observatory Linux Mint 21.3 host
crashed while Qt's XCB plugin initialized xkbcommon.  Give Qt/xkbcommon an
explicit distro XKB data path and prefer XCB only for real X11 sessions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


if sys.platform.startswith("linux"):
    xkb_root = Path("/usr/share/X11/xkb")
    if xkb_root.is_dir():
        os.environ.setdefault("QT_XKB_CONFIG_ROOT", str(xkb_root))
        os.environ.setdefault("XKB_CONFIG_ROOT", str(xkb_root))

    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if session_type == "x11" or (
        os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
