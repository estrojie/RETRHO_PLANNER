#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PNG="$ROOT/assets/rho_planner.png"
ICONSET="$ROOT/assets/rho_planner.iconset"
ICNS="$ROOT/assets/rho_planner.icns"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "make_icns.sh must be run on macOS." >&2
  exit 1
fi

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$ICNS"
rm -rf "$ICONSET"
echo "Created $ICNS"
