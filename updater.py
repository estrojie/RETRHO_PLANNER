"""GitHub release updater for RHO Planner.

The updater is intentionally split into network/download logic and a small
platform-specific replacement helper. The running application never overwrites
itself directly. Instead it downloads and verifies the selected release asset,
starts a helper outside the installation directory, then exits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Callable, Optional

import requests


GITHUB_REPOSITORY = "estrojie/RETRHO_PLANNER"
LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
)
RELEASES_PAGE = f"https://github.com/{GITHUB_REPOSITORY}/releases"
REQUEST_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "RHOPlanner-Updater",
}
DEFAULT_TIMEOUT_S = 12


@dataclass(frozen=True)
class ReleaseInfo:
    current_version: str
    latest_version: str
    tag_name: str
    html_url: str
    body: str
    asset_name: str
    asset_url: str
    checksum_url: str
    update_available: bool
    comparison_supported: bool = True


def _numeric_version(value: str) -> Optional[tuple[int, ...]]:
    """Parse release-like versions such as v1.4 or 1.4.2.

    Development strings intentionally return None so a PR/dev build does not
    try to replace itself with the latest public release.
    """
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    if not re.fullmatch(r"\d+(?:\.\d+)*", text):
        return None
    return tuple(int(part) for part in text.split("."))


def version_is_newer(latest: str, current: str) -> bool:
    latest_v = _numeric_version(latest)
    current_v = _numeric_version(current)
    if latest_v is None or current_v is None:
        return False
    width = max(len(latest_v), len(current_v))
    latest_v += (0,) * (width - len(latest_v))
    current_v += (0,) * (width - len(current_v))
    return latest_v > current_v


def _normalized_machine(machine: Optional[str] = None) -> str:
    value = (machine or platform.machine() or "").strip().lower()
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }
    return aliases.get(value, value)


def installation_kind() -> str:
    """Return the update packaging style for this running copy."""
    if not getattr(sys, "frozen", False):
        return "development"
    if sys.platform == "win32":
        return "windows-exe"
    if sys.platform == "darwin":
        return "macos-app"
    if sys.platform.startswith("linux"):
        if os.environ.get("APPIMAGE"):
            return "linux-appimage"
        return "linux-tar"
    return "unsupported"


def release_asset_name(
    *,
    platform_name: Optional[str] = None,
    machine: Optional[str] = None,
    kind: Optional[str] = None,
) -> str:
    platform_name = platform_name or sys.platform
    kind = kind or installation_kind()
    machine = _normalized_machine(machine)

    if platform_name == "win32" or kind == "windows-exe":
        if machine and machine not in {"x86_64"}:
            raise RuntimeError(f"Unsupported Windows architecture: {machine}")
        return "RHOPlanner-Windows-x64.exe"

    if platform_name == "darwin" or kind == "macos-app":
        if machine == "arm64":
            return "RHOPlanner-macOS-arm64.zip"
        if machine in {"x86_64", ""}:
            return "RHOPlanner-macOS-x86_64.zip"
        raise RuntimeError(f"Unsupported macOS architecture: {machine}")

    if platform_name.startswith("linux") or kind.startswith("linux-"):
        if machine and machine not in {"x86_64"}:
            raise RuntimeError(f"Unsupported Linux architecture: {machine}")
        if kind == "linux-appimage":
            return "RHOPlanner-Linux-x86_64.AppImage"
        return "RHOPlanner-Linux-x86_64.tar.xz"

    raise RuntimeError(f"Automatic updates are not supported on {platform_name!r}.")


def _asset_map(payload: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for asset in payload.get("assets", []) or []:
        name = str(asset.get("name") or "").strip()
        url = str(asset.get("browser_download_url") or "").strip()
        if name and url:
            out[name] = url
    return out


def check_latest_release(
    current_version: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> ReleaseInfo:
    """Query GitHub's latest stable release and choose this platform's asset."""
    response = requests.get(
        LATEST_RELEASE_API,
        headers=REQUEST_HEADERS,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()

    tag_name = str(payload.get("tag_name") or "").strip()
    latest_version = tag_name[1:] if tag_name.lower().startswith("v") else tag_name
    if not latest_version:
        raise RuntimeError("GitHub returned a release without a version tag.")

    kind = installation_kind()
    # Development runs still show release information for the manual check.
    asset_kind = kind
    if kind == "development":
        if sys.platform == "win32":
            asset_kind = "windows-exe"
        elif sys.platform == "darwin":
            asset_kind = "macos-app"
        elif sys.platform.startswith("linux"):
            asset_kind = "linux-tar"

    wanted = release_asset_name(kind=asset_kind)
    assets = _asset_map(payload)
    asset_url = assets.get(wanted, "")
    checksum_url = assets.get("SHA256SUMS.txt", "")
    if not asset_url:
        raise RuntimeError(
            f"The latest release does not contain the expected asset {wanted!r}."
        )
    if not checksum_url:
        raise RuntimeError("The latest release does not contain SHA256SUMS.txt.")

    comparison_supported = _numeric_version(current_version) is not None
    return ReleaseInfo(
        current_version=str(current_version),
        latest_version=latest_version,
        tag_name=tag_name,
        html_url=str(payload.get("html_url") or RELEASES_PAGE),
        body=str(payload.get("body") or ""),
        asset_name=wanted,
        asset_url=asset_url,
        checksum_url=checksum_url,
        update_available=(
            comparison_supported
            and version_is_newer(latest_version, current_version)
        ),
        comparison_supported=comparison_supported,
    )


def parse_checksum_file(text: str) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, filename = parts
        filename = filename.strip().lstrip("*")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) and filename:
            checksums[filename] = digest.lower()
    return checksums


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_release_asset(
    info: ReleaseInfo,
    *,
    progress: Optional[Callable[[int], None]] = None,
    timeout_s: int = 60,
) -> Path:
    """Download the selected asset and verify it against SHA256SUMS.txt."""
    checksum_response = requests.get(
        info.checksum_url,
        headers=REQUEST_HEADERS,
        timeout=DEFAULT_TIMEOUT_S,
    )
    checksum_response.raise_for_status()
    checksums = parse_checksum_file(checksum_response.text)
    expected = checksums.get(info.asset_name)
    if not expected:
        raise RuntimeError(
            f"No SHA-256 checksum was published for {info.asset_name}."
        )

    update_dir = Path(tempfile.mkdtemp(prefix="rho-planner-update-"))
    final_path = update_dir / info.asset_name
    partial_path = final_path.with_suffix(final_path.suffix + ".part")

    with requests.get(
        info.asset_url,
        headers=REQUEST_HEADERS,
        stream=True,
        timeout=timeout_s,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        with partial_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                received += len(chunk)
                if progress is not None and total > 0:
                    progress(min(100, int(received * 100 / total)))

    actual = sha256_file(partial_path)
    if actual.lower() != expected.lower():
        try:
            partial_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "The downloaded update failed SHA-256 verification. "
            "The existing installation was not changed."
        )

    partial_path.replace(final_path)
    if progress is not None:
        progress(100)
    return final_path


def _mac_app_bundle() -> Optional[Path]:
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.name.endswith(".app"):
            return parent
    return None


def install_target() -> Optional[Path]:
    kind = installation_kind()
    if kind == "windows-exe":
        return Path(sys.executable).resolve()
    if kind == "macos-app":
        return _mac_app_bundle()
    if kind == "linux-appimage":
        value = os.environ.get("APPIMAGE")
        return Path(value).resolve() if value else None
    if kind == "linux-tar":
        return Path(sys.executable).resolve().parent
    return None


def can_self_install() -> tuple[bool, str]:
    kind = installation_kind()
    if kind == "development":
        return False, "Development checkouts are not replaced automatically."
    if kind == "unsupported":
        return False, "This operating system is not supported by the updater."

    target = install_target()
    if target is None:
        return False, "The updater could not determine the installation path."

    parent = target.parent
    if not parent.exists():
        return False, "The installation directory no longer exists."
    if not os.access(parent, os.W_OK):
        return False, (
            f"RHO Planner cannot write to {parent}. "
            "Install the update manually from the GitHub release page."
        )
    return True, ""


def _write_helper(contents: str, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="rho-planner-install-", suffix=suffix)
    os.close(fd)
    path = Path(name)
    path.write_text(contents, encoding="utf-8")
    if suffix == ".sh":
        path.chmod(0o700)
    return path


def _launch_windows_helper(downloaded: Path, target: Path) -> None:
    script = _write_helper(
        r'''param(
    [Parameter(Mandatory=$true)][int]$OldPid,
    [Parameter(Mandatory=$true)][string]$Downloaded,
    [Parameter(Mandatory=$true)][string]$Target
)
$ErrorActionPreference = "Stop"
try { Wait-Process -Id $OldPid -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Milliseconds 500

$Backup = "$Target.rho-old"
if (Test-Path $Backup) { Remove-Item -Force $Backup }

try {
    if (Test-Path $Target) { Move-Item -Force $Target $Backup }
    Move-Item -Force $Downloaded $Target
    Start-Process -FilePath $Target
    Start-Sleep -Seconds 2
    if (Test-Path $Backup) { Remove-Item -Force $Backup }
} catch {
    if ((Test-Path $Backup) -and -not (Test-Path $Target)) {
        Move-Item -Force $Backup $Target
    }
    throw
}
''',
        ".ps1",
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-OldPid",
            str(os.getpid()),
            "-Downloaded",
            str(downloaded),
            "-Target",
            str(target),
        ],
        close_fds=True,
        creationflags=creationflags,
    )


def _launch_unix_helper(downloaded: Path, target: Path, kind: str) -> None:
    q_downloaded = shlex.quote(str(downloaded))
    q_target = shlex.quote(str(target))
    q_pid = str(os.getpid())

    if kind == "linux-appimage":
        body = f'''#!/bin/sh
set -eu
OLD_PID={q_pid}
DOWNLOADED={q_downloaded}
TARGET={q_target}
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 0.25; done
BACKUP="$TARGET.rho-old"
rm -f "$BACKUP"
if [ -e "$TARGET" ]; then mv "$TARGET" "$BACKUP"; fi
if mv "$DOWNLOADED" "$TARGET"; then
    chmod +x "$TARGET"
    "$TARGET" >/dev/null 2>&1 &
    NEW_PID=$!
    sleep 2
    if kill -0 "$NEW_PID" 2>/dev/null; then
        rm -f "$BACKUP"
        exit 0
    fi
fi
if [ -e "$BACKUP" ]; then
    rm -f "$TARGET"
    mv "$BACKUP" "$TARGET"
fi
exit 1
'''
    elif kind == "linux-tar":
        body = f'''#!/bin/sh
set -eu
OLD_PID={q_pid}
ARCHIVE={q_downloaded}
TARGET_DIR={q_target}
PARENT="$(dirname "$TARGET_DIR")"
NAME="$(basename "$TARGET_DIR")"
STAGE="$(mktemp -d "$PARENT/.rho-update-stage.XXXXXX")"
BACKUP="$PARENT/$NAME.rho-old"
cleanup() {{ rm -rf "$STAGE"; }}
trap cleanup EXIT

tar -xJf "$ARCHIVE" -C "$STAGE"
NEW_DIR="$STAGE/RHOPlanner"
[ -x "$NEW_DIR/RHOPlanner" ]

while kill -0 "$OLD_PID" 2>/dev/null; do sleep 0.25; done
rm -rf "$BACKUP"
mv "$TARGET_DIR" "$BACKUP"
if mv "$NEW_DIR" "$TARGET_DIR"; then
    "$TARGET_DIR/RHOPlanner" >/dev/null 2>&1 &
    NEW_PID=$!
    sleep 2
    if kill -0 "$NEW_PID" 2>/dev/null; then
        rm -rf "$BACKUP"
        exit 0
    fi
fi
rm -rf "$TARGET_DIR"
mv "$BACKUP" "$TARGET_DIR"
exit 1
'''
    elif kind == "macos-app":
        body = f'''#!/bin/sh
set -eu
OLD_PID={q_pid}
ARCHIVE={q_downloaded}
TARGET_APP={q_target}
PARENT="$(dirname "$TARGET_APP")"
STAGE="$(mktemp -d "$PARENT/.rho-update-stage.XXXXXX")"
BACKUP="$TARGET_APP.rho-old"
cleanup() {{ rm -rf "$STAGE"; }}
trap cleanup EXIT

ditto -x -k "$ARCHIVE" "$STAGE"
NEW_APP="$(find "$STAGE" -maxdepth 2 -type d -name 'RHOPlanner.app' -print -quit)"
[ -n "$NEW_APP" ]

while kill -0 "$OLD_PID" 2>/dev/null; do sleep 0.25; done
rm -rf "$BACKUP"
mv "$TARGET_APP" "$BACKUP"
if mv "$NEW_APP" "$TARGET_APP"; then
    open "$TARGET_APP"
    sleep 2
    rm -rf "$BACKUP"
    exit 0
fi
rm -rf "$TARGET_APP"
mv "$BACKUP" "$TARGET_APP"
exit 1
'''
    else:
        raise RuntimeError(f"Unsupported installer kind: {kind}")

    script = _write_helper(body, ".sh")
    subprocess.Popen(
        ["/bin/sh", str(script)],
        start_new_session=True,
        close_fds=True,
    )


def launch_installer(downloaded: Path) -> None:
    """Start the platform helper that replaces the app after this process exits."""
    ok, reason = can_self_install()
    if not ok:
        raise RuntimeError(reason)

    kind = installation_kind()
    target = install_target()
    if target is None:
        raise RuntimeError("Could not determine the current installation path.")

    downloaded = Path(downloaded).resolve()
    if not downloaded.exists():
        raise RuntimeError("The downloaded update file no longer exists.")

    if kind == "windows-exe":
        _launch_windows_helper(downloaded, target)
    else:
        _launch_unix_helper(downloaded, target, kind)
