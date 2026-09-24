#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: smoke_test_linux_bundle.sh <container-image>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$ROOT/dist/RHOPlanner"

if [[ ! -x "$BUNDLE/RHOPlanner" ]]; then
  echo "Missing Linux bundle at $BUNDLE" >&2
  exit 1
fi

echo "Testing RHO Planner in $IMAGE"
docker run --rm \
  -v "$BUNDLE:/app:ro" \
  "$IMAGE" bash -lc '
    set -e
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq \
        ca-certificates coreutils libglib2.0-0 libgl1 libegl1 \
        libxkbcommon0 libxkbcommon-x11-0 libxcb-xkb1 xkb-data \
        libdbus-1-3 libfontconfig1 libx11-6 libxcb1 \
        libxext6 libxrender1 libxi6 libxrandr2 libxfixes3 \
        xvfb xauth \
        >/dev/null
      apt-get install -y -qq \
        libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
        libxcb-keysyms1 libxcb-image0 libxcb-render-util0 \
        libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 \
        >/dev/null 2>&1 || true
    elif command -v pacman >/dev/null 2>&1; then
      pacman -Sy --noconfirm --needed \
        ca-certificates coreutils glib2 mesa libxkbcommon xkeyboard-config \
        dbus fontconfig libx11 libxcb libxext libxrender libxi libxrandr libxfixes \
        xcb-util-cursor xcb-util-keysyms xcb-util-image xcb-util-renderutil \
        xorg-server-xvfb xorg-xauth \
        >/dev/null
    else
      echo "Unsupported test container package manager" >&2
      exit 1
    fi

    ldd /app/RHOPlanner | tee /tmp/rho-ldd.txt
    if grep -q "not found" /tmp/rho-ldd.txt; then
      echo "Missing shared libraries:" >&2
      grep "not found" /tmp/rho-ldd.txt >&2
      exit 1
    fi

    set +e
    QT_QPA_PLATFORM=offscreen timeout 15s /app/RHOPlanner >/tmp/rho-offscreen.stdout 2>/tmp/rho-offscreen.stderr
    offscreen_code=$?
    set -e
    cat /tmp/rho-offscreen.stdout || true
    cat /tmp/rho-offscreen.stderr >&2 || true
    if [[ $offscreen_code -ne 124 ]]; then
      echo "RHO Planner offscreen startup exited unexpectedly with code $offscreen_code" >&2
      exit $offscreen_code
    fi

    # Exercise the actual X11/XCB path.  The Mint 21.3 failure occurred during
    # xkb_x11_keymap_new_from_device(), which an offscreen test cannot catch.
    set +e
    xvfb-run -a -s "-screen 0 1024x768x24" \
      timeout 15s /app/RHOPlanner >/tmp/rho-x11.stdout 2>/tmp/rho-x11.stderr
    x11_code=$?
    set -e
    cat /tmp/rho-x11.stdout || true
    cat /tmp/rho-x11.stderr >&2 || true
    if [[ $x11_code -ne 124 ]]; then
      echo "RHO Planner X11/XCB startup exited unexpectedly with code $x11_code" >&2
      exit $x11_code
    fi
  '
