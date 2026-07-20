# RHO Planner desktop releases

This repository is configured to build distributable desktop versions of RHO Planner for:

- Windows x86-64
- macOS Apple Silicon (`arm64`)
- macOS Intel (`x86_64`)
- Linux x86-64 as both an AppImage and a portable tarball

The builds are produced on their native operating systems with PyInstaller. The Linux build is created in a `manylinux_2_28` container and is smoke-tested on Ubuntu 20.04, Linux Mint 22, and current Arch Linux.

## Files added for packaging

- `RHOPlanner.spec`: reliable folder-based build and macOS `.app` bundle
- `RHOPlanner_onefile.spec`: portable Windows single-file executable
- `requirements.txt`: runtime dependencies
- `requirements-build.txt`: frozen build-tool versions
- `.github/workflows/build-desktop.yml`: automated multi-platform builds and releases
- `build_support/`: PyInstaller helpers, SSL runtime setup, icon generation, Linux packaging, and Linux smoke tests
- `packaging/linux/`: AppImage desktop metadata and launcher
- `assets/`: application icons

## Build through GitHub Actions

1. Copy these files into the root of the existing repository, alongside `main.py` and `planner_core.py`.
2. Commit and push them.
3. Open the repository's **Actions** tab.
4. Select **Build desktop executables**.
5. Choose **Run workflow**.

A manual run stores the finished downloads as workflow artifacts.

## Publish a GitHub release

Create and push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow builds every platform with current GitHub-maintained actions and creates a GitHub release containing:

- `RHOPlanner-Windows-x64.exe`
- `RHOPlanner-Windows-x64.zip`
- `RHOPlanner-macOS-arm64.dmg`
- `RHOPlanner-macOS-arm64.zip`
- `RHOPlanner-macOS-x86_64.dmg`
- `RHOPlanner-macOS-x86_64.zip`
- `RHOPlanner-Linux-x86_64.AppImage`
- `RHOPlanner-Linux-x86_64.tar.gz`
- dependency manifests for each build
- `SHA256SUMS.txt` for download verification

## Why Windows has two downloads

The single `.exe` is convenient, but it extracts its internal files to a temporary directory at launch. This can result in slower startup and occasional antivirus false positives.

The `.zip` contains the folder-based build. It starts faster and is the recommended Windows download when reliability matters. Users should extract the whole folder before launching `RHOPlanner.exe`.

## Linux compatibility

The Linux x86-64 release intentionally uses `PySide6==6.9.3`. That version provides a `manylinux_2_28` x86-64 wheel, allowing the bundle to target glibc 2.28 or newer. The workflow builds in the official PyPA `manylinux_2_28_x86_64` container rather than on the newer Ubuntu runner itself.

The resulting target includes common current versions of:

- Linux Mint
- Ubuntu
- Debian
- Arch Linux
- Fedora and related distributions with glibc 2.28 or newer

The AppImage is the easiest Linux download. The tarball is included as a fallback for systems where AppImage/FUSE support is unavailable.

### AppImage use

```bash
chmod +x RHOPlanner-Linux-x86_64.AppImage
./RHOPlanner-Linux-x86_64.AppImage
```

If FUSE is unavailable:

```bash
./RHOPlanner-Linux-x86_64.AppImage --appimage-extract-and-run
```

Alternatively, extract the tarball and run:

```bash
./RHOPlanner/RHOPlanner
```

## Local builds

Always build on the operating system being targeted.

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-build.txt
python -m PyInstaller --noconfirm --clean RHOPlanner.spec
python -m PyInstaller --noconfirm --clean --distpath dist-onefile RHOPlanner_onefile.spec
```

### macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-build.txt
./build_support/make_icns.sh
python -m PyInstaller --noconfirm --clean RHOPlanner.spec
```

### Linux

A local Linux build inherits the glibc version of the build computer. Use the GitHub workflow for the widest compatibility. A simple same-machine build is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-build.txt
python -m PyInstaller --noconfirm --clean RHOPlanner.spec
./build_support/build_linux_appimage.sh
```

## Signing and operating-system warnings

The provided workflow creates unsigned public builds.

- Windows may show Microsoft SmartScreen until the executable has reputation or is Authenticode-signed.
- macOS may block the first launch because the application is not signed with an Apple Developer ID and notarized. A user can right-click the application and choose **Open**, but public releases are better with Developer ID signing and notarization.
- The macOS workflow applies an ad-hoc signature so the app bundle is internally consistent; this is not a substitute for Developer ID signing.

Signing credentials should be stored only as encrypted GitHub Actions secrets. Do not commit certificates, private keys, or passwords to the repository.

## Internet-dependent features

The planner itself runs locally, but several features require internet access:

- SIMBAD/VizieR target resolution and star identification
- SkyView and Pan-STARRS finder images
- National Weather Service cloud information
- Astropy IERS data updates

The frozen application includes a certificate bundle and sets the standard SSL environment variables at startup so HTTPS requests continue to work after packaging.

## Release testing

The workflow checks Python syntax and verifies that the Linux folder bundle remains running in headless mode on Ubuntu 20.04, Linux Mint 22, and Arch Linux. Before a public release, also test manually on real desktops for:

- finder-chart downloads and rotation
- Gaia/Tycho/SIMBAD star identification
- CSV and XLSX import
- exposure-time calculations
- clipboard copying
- high-DPI and smaller-screen layouts
- macOS Intel and Apple Silicon launch behavior
- Windows Defender/SmartScreen behavior

A container smoke test is useful but cannot fully replace a real Cinnamon, GNOME, KDE, Windows, or macOS desktop test.
