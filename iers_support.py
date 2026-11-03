"""Nonblocking, persistent IERS Earth-orientation data maintenance for RHO Planner.

Astropy's bundled table is always available for offline planning.  A separate
background task periodically refreshes a validated table in the user's cache.
No coordinate calculation needs to wait for an IERS network request.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import requests
from astropy.config import get_cache_dir
from astropy.utils import iers

_LOG = logging.getLogger(__name__)
_REFRESH_INTERVAL_S = 7 * 24 * 60 * 60
_MAX_TABLE_BYTES = 15 * 1024 * 1024
_start_lock = threading.Lock()
_state_lock = threading.Lock()
_started = False
_status = "Bundled IERS data (checking for updates)"


def configure_iers() -> None:
    """Keep astronomical calculations offline and functional during outages.

    Disable Astropy's *synchronous* network access; we handle updates in our
    own worker.  Only the age restriction is relaxed: IERS will warn about
    extrapolation, which may be inaccurate when tables are very old.
    """
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None


def _cache_path() -> Path:
    """A writable, persistent per-user path (Windows/macOS/Linux)."""
    return Path(get_cache_dir("rho_planner")) / "finals2000A.all"


def status_message() -> str:
    with _state_lock:
        return _status


def _set_status(message: str) -> None:
    global _status
    with _state_lock:
        _status = message


def _last_mjd(table) -> float:
    """The end of a table's coverage, for rejecting older replacements."""
    last = table["MJD"][-1]
    return float(getattr(last, "value", last))


def refresh_iers_now() -> str:
    """Load a cached table, then refresh it if older than seven days.

    Called only on the background thread in the desktop app.  Returns
    'updated', 'cached', or 'offline', and never lets network errors escape.
    """
    configure_iers()
    using_cache = False
    try:
        cache_path = _cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _LOG.warning("Cannot open IERS cache: %s", exc)
        _set_status("Bundled IERS data (cache unavailable; accuracy may decrease)")
        return "offline"

    if cache_path.is_file():
        try:
            cached = iers.IERS_A.open(str(cache_path))
            if _last_mjd(cached) >= _last_mjd(iers.earth_orientation_table.get()):
                iers.earth_orientation_table.set(cached)
                using_cache = True
                _set_status("Cached IERS data (checking freshness)")
        except Exception as exc:
            _LOG.warning("Unable to read cached IERS data: %s", exc)

    if using_cache:
        try:
            if time.time() - cache_path.stat().st_mtime < _REFRESH_INTERVAL_S:
                _set_status("Cached IERS data (current)")
                return "cached"
        except OSError:
            pass

    errors = []
    for url in (iers.conf.iers_auto_url, iers.conf.iers_auto_url_mirror):
        tmp_path = None
        try:
            response = requests.get(url, timeout=(3, 10))
            response.raise_for_status()
            payload = response.content
            if not (1000 <= len(payload) <= _MAX_TABLE_BYTES):
                raise ValueError("Unexpected IERS download size")

            # Validate before committing, and never overwrite a good cached
            # table with malformed or older information.
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".iers-", suffix=".all",
                dir=cache_path.parent, delete=False,
            ) as temp:
                tmp_path = Path(temp.name)
                temp.write(payload)
            candidate = iers.IERS_A.open(str(tmp_path))
            if len(candidate) < 1000:
                raise ValueError("IERS download contained too few rows")
            if _last_mjd(candidate) < _last_mjd(iers.earth_orientation_table.get()):
                raise ValueError("Downloaded IERS table is older than the active table")

            os.replace(tmp_path, cache_path)
            iers.earth_orientation_table.set(candidate)
            _set_status("IERS data updated (online)")
            return "updated"
        except Exception as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    _LOG.warning("IERS update unavailable: %s", "; ".join(errors))
    source = "Cached" if using_cache else "Bundled"
    _set_status(f"{source} IERS data (offline/stale; accuracy may decrease)")
    return "offline"


def start_background_refresh() -> bool:
    """Run maintenance at most once per application launch, without GUI delay."""
    global _started
    with _start_lock:
        if _started:
            return False
        _started = True
    try:
        threading.Thread(
            target=refresh_iers_now, name="RHO-IERS-refresh", daemon=True
        ).start()
        return True
    except Exception as exc:
        with _start_lock:
            _started = False
        _LOG.warning("Could not start IERS maintenance: %s", exc)
        _set_status("Bundled IERS data (automatic refresh unavailable)")
        return False


configure_iers()
