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
        libxkbcommon0 libdbus-1-3 libfontconfig1 libx11-6 libxcb1 \
        libxext6 libxrender1 libxi6 libxrandr2 libxfixes3 \
        >/dev/null
      apt-get install -y -qq \
        libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
        libxcb-keysyms1 libxcb-image0 libxcb-render-util0 \
        libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 \
        >/dev/null 2>&1 || true
    elif command -v pacman >/dev/null 2>&1; then
      pacman -Sy --noconfirm --needed \
        ca-certificates coreutils glib2 mesa libxkbcommon dbus fontconfig \
        libx11 libxcb libxext libxrender libxi libxrandr libxfixes \
        xcb-util-cursor xcb-util-keysyms xcb-util-image xcb-util-renderutil \
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
    QT_QPA_PLATFORM=offscreen timeout 35s /app/RHOPlanner >/tmp/rho.stdout 2>/tmp/rho.stderr
    code=$?
    set -e
    cat /tmp/rho.stdout || true
    cat /tmp/rho.stderr >&2 || true

    # 124 means the GUI remained alive until timeout, which is the expected
    # outcome for this headless launch test.
    if [[ $code -ne 124 ]]; then
      echo "RHO Planner exited unexpectedly with code $code" >&2
      exit $code
    fi
  '
