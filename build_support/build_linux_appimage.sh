#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist/RHOPlanner"
RELEASE="$ROOT/release"
APPDIR="$ROOT/build/RHOPlanner.AppDir"

if [[ ! -x "$DIST/RHOPlanner" ]]; then
  echo "Missing $DIST/RHOPlanner. Build RHOPlanner.spec first." >&2
  exit 1
fi

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$RELEASE"
cp -a "$DIST"/. "$APPDIR/usr/bin/"
cp "$ROOT/packaging/linux/AppRun" "$APPDIR/AppRun"
cp "$ROOT/packaging/linux/RHOPlanner.desktop" "$APPDIR/RHOPlanner.desktop"
cp "$ROOT/assets/rho-planner.svg" "$APPDIR/rho-planner.svg"
chmod +x "$APPDIR/AppRun" "$APPDIR/usr/bin/RHOPlanner"

TOOL="$ROOT/build/appimagetool-x86_64.AppImage"
if [[ ! -f "$TOOL" ]]; then
  curl -L --fail --retry 3 \
    -o "$TOOL" \
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$TOOL"
fi

ARCH=x86_64 "$TOOL" --appimage-extract-and-run \
  "$APPDIR" "$RELEASE/RHOPlanner-Linux-x86_64.AppImage"
chmod +x "$RELEASE/RHOPlanner-Linux-x86_64.AppImage"

tar -C "$ROOT/dist" -czf "$RELEASE/RHOPlanner-Linux-x86_64.tar.gz" RHOPlanner

echo "Created Linux release files in $RELEASE"
