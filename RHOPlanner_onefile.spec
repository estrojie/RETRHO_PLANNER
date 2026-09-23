# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()
sys.path.insert(0, str(ROOT / "build_support"))

from pyinstaller_config import build_collection, icon_for_platform

DATAS, BINARIES, HIDDENIMPORTS = build_collection()\n\nRUNTIME_HOOKS = [str(ROOT / "build_support" / "rthook_network.py")]\nif sys.platform.startswith("linux"):\n    RUNTIME_HOOKS.append(str(ROOT / "build_support" / "rthook_linux_qt_compat.py"))
ICON = icon_for_platform(ROOT, sys.platform)\nSTRIP_BINARIES = sys.platform.startswith("linux")

analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDENIMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=RUNTIME_HOOKS,
    excludes=[
        "PyQt5", "PyQt6", "PySide2", "tkinter", "IPython",
        "jupyter", "notebook", "pytest", "sphinx",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="RHOPlanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=STRIP_BINARIES,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
