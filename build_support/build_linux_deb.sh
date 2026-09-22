#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RHOPlanner"
PACKAGE_NAME="rhoplanner"
ARCH="amd64"
VERSION_RAW="${RHO_PLANNER_VERSION:-0.0.0-dev}"
DIST_DIR="dist/${APP_NAME}"
RELEASE_DIR="release"
ICON="assets/rho_planner.png"

if [[ ! -x "${DIST_DIR}/${APP_NAME}" ]]; then
  echo "Expected Linux onedir build at ${DIST_DIR}/${APP_NAME}" >&2
  exit 1
fi

if [[ ! -f "${ICON}" ]]; then
  echo "Missing icon: ${ICON}" >&2
  exit 1
fi

if [[ "${VERSION_RAW}" =~ ^[0-9] ]]; then
  DEB_VERSION="${VERSION_RAW//_/.}"
else
  DEB_VERSION="0.0.0+${VERSION_RAW//_/.}"
fi
DEB_VERSION="${DEB_VERSION//\//.}"
DEB_VERSION="${DEB_VERSION// /}"

PKG_ROOT="$(mktemp -d)"
trap 'rm -rf "$PKG_ROOT"' EXIT

mkdir -p   "${PKG_ROOT}/DEBIAN"   "${PKG_ROOT}/usr/lib/rhoplanner"   "${PKG_ROOT}/usr/bin"   "${PKG_ROOT}/usr/share/applications"   "${PKG_ROOT}/usr/share/icons/hicolor/256x256/apps"

cp -a "${DIST_DIR}/." "${PKG_ROOT}/usr/lib/rhoplanner/"
cp "${ICON}" "${PKG_ROOT}/usr/share/icons/hicolor/256x256/apps/rhoplanner.png"

cat > "${PKG_ROOT}/usr/bin/rhoplanner" <<'LAUNCHER'
#!/usr/bin/env bash
exec /usr/lib/rhoplanner/RHOPlanner "$@"
LAUNCHER
chmod 0755 "${PKG_ROOT}/usr/bin/rhoplanner"

cat > "${PKG_ROOT}/usr/share/applications/rhoplanner.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=RETRHO Planner
Comment=Rosemary Hill Observatory observation planning tool
Exec=rhoplanner
Icon=rhoplanner
Terminal=false
Categories=Education;Science;Astronomy;
StartupNotify=true
DESKTOP

cat > "${PKG_ROOT}/DEBIAN/control" <<EOF_CONTROL
Package: ${PACKAGE_NAME}
Version: ${DEB_VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: RETRHO Planner Project
Depends: libc6 (>= 2.31), libgl1, libegl1, libxkbcommon0, libdbus-1-3, libfontconfig1, libx11-6, libxcb1, libxext6, libxrender1, libxi6, libxrandr2, libxfixes3
Description: RETRHO Planner desktop observation-planning application
 Standalone desktop planning software for Rosemary Hill Observatory.
EOF_CONTROL

find "${PKG_ROOT}" -type d -exec chmod 0755 {} +
mkdir -p "${RELEASE_DIR}"
OUT="${RELEASE_DIR}/RHOPlanner-Linux-amd64.deb"
dpkg-deb --root-owner-group --build "${PKG_ROOT}" "${OUT}"
echo "Created ${OUT}"
