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
    """Return conservative data, binary, and hidden-import collections.

    PyInstaller already has strong hooks for NumPy, SciPy, pandas, Matplotlib,
    Astropy, Pillow, and PySide6. The additions here cover package metadata,
    astronomy data files, timezone/certificate data, Excel support, and the
    astroquery services used dynamically by RHO Planner.
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
    ]

    for package in (
        "astropy",
        "astropy_iers_data",
        "astroquery",
        "astroplan",
        "matplotlib",
        "certifi",
        "tzdata",
        "openpyxl",
    ):
        try:
            _extend_unique(datas, collect_data_files(package, include_py_files=False))
        except Exception:
            pass
        try:
            _extend_unique(datas, copy_metadata(package))
        except Exception:
            pass

    for package in ("astroquery", "astroplan", "openpyxl"):
        try:
            _extend_unique(
                hiddenimports,
                collect_submodules(package, on_error="warn once"),
            )
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
