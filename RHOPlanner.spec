# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()
sys.path.insert(0, str(ROOT / "build_support"))

from pyinstaller_config import build_collection, icon_for_platform

APP_NAME = "RHOPlanner"
APP_DISPLAY_NAME = "RHO Planner"
APP_VERSION = os.environ.get("RHO_PLANNER_VERSION", "0.0.0-dev")


def _macos_bundle_version(value: str) -> str:
    """Return an Apple-compatible numeric bundle version.

    Git tags such as ``1.2.3`` are retained. Development identifiers such as
    ``dev-abcdef0`` become ``0.0.0`` so Finder/Gatekeeper do not reject the
    bundle metadata. The original version remains available in the custom
    RHOPlannerVersion Info.plist key.
    """
    match = re.search(r"(?<!\d)(\d+(?:\.\d+){0,3})(?!\d)", str(value))
    return match.group(1) if match else "0.0.0"


MACOS_BUNDLE_VERSION = _macos_bundle_version(APP_VERSION)
ICON = icon_for_platform(ROOT, sys.platform)
DATAS, BINARIES, HIDDENIMPORTS = build_collection()

analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDENIMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "build_support" / "rthook_network.py")],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier="edu.ufl.rho.planner",
        version=MACOS_BUNDLE_VERSION,
        info_plist={
            "CFBundleDisplayName": APP_DISPLAY_NAME,
            "CFBundleName": APP_DISPLAY_NAME,
            "CFBundleShortVersionString": MACOS_BUNDLE_VERSION,
            "CFBundleVersion": MACOS_BUNDLE_VERSION,
            "RHOPlannerVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
