from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


def _extend_unique(target: list, values: list) -> None:
    seen = set(target)
    for value in values:
        key = tuple(value) if isinstance(value, (list, tuple)) else value
        if key not in seen:
            target.append(value)
            seen.add(key)


def build_collection() -> Tuple[List[tuple], List[tuple], List[str]]:
    """Return a targeted PyInstaller collection for RHO Planner.

    PyInstaller's standard hooks already handle the large compiled stacks
    (NumPy, pandas, Matplotlib, Astropy, Pillow, and PySide6).  Avoid collecting
    whole package trees again: that increases archive size and import scanning.
    Only package data and dynamic imports that RHO Planner actually relies on
    are added here.
    """
    datas: List[tuple] = []
    binaries: List[tuple] = []
    hiddenimports: List[str] = [
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
        "openpyxl",
        "certifi",
        "tzdata",
        "astropy_iers_data",
        "astroquery.skyview",
        "astroquery.simbad",
        "astroquery.vizier",
        "astroquery.utils",
        "astroquery.query",
        "astroplan",
        "pyvo",
    ]

    # These packages read runtime data files that are not always discoverable
    # from static imports.  Keep the set intentionally small.
    for package in (
        "astropy_iers_data",
        "astroquery",
        "pyvo",
        "astroplan",
        "certifi",
        "tzdata",
    ):
        try:
            _extend_unique(datas, collect_data_files(package, include_py_files=False))
        except Exception:
            pass
        try:
            _extend_unique(datas, copy_metadata(package))
        except Exception:
            pass

    # Metadata is useful for engines selected dynamically at runtime, without
    # pulling every module in those packages into the executable.
    for package in ("openpyxl", "matplotlib", "astropy"):
        try:
            _extend_unique(datas, copy_metadata(package))
        except Exception:
            pass

    return datas, binaries, hiddenimports


def icon_for_platform(root: Path, platform_name: str) -> str | None:
    assets = root / "assets"
    if platform_name == "win32":
        path = assets / "rho_planner.ico"
    elif platform_name == "darwin":
        icns = assets / "rho_planner.icns"
        path = icns if icns.exists() else assets / "rho_planner.png"
    else:
        return None
    return str(path) if path.exists() else None
