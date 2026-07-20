# Which RHO Planner download should I use?

## Windows

**Recommended:** `RHOPlanner-Windows-x64.zip`

Extract the complete folder, open it, and run `RHOPlanner.exe`. Do not move the executable out of its extracted folder.

**Convenience option:** `RHOPlanner-Windows-x64.exe`

This is a single-file portable build. It may start more slowly and can be more likely to trigger antivirus reputation warnings.

## macOS

- Apple Silicon M1/M2/M3/M4 or later: `RHOPlanner-macOS-arm64.dmg`
- Intel Mac: `RHOPlanner-macOS-x86_64.dmg`

Open the DMG and drag **RHOPlanner.app** to Applications. Unsigned development releases may require right-clicking the app and choosing **Open** the first time.

## Linux

**Recommended:** `RHOPlanner-Linux-x86_64.AppImage`

```bash
chmod +x RHOPlanner-Linux-x86_64.AppImage
./RHOPlanner-Linux-x86_64.AppImage
```

**Fallback:** `RHOPlanner-Linux-x86_64.tar.gz`

Extract it and run `RHOPlanner/RHOPlanner`.

The Linux x86-64 build targets glibc 2.28 or newer and is tested in the release workflow on Linux Mint 22, Ubuntu 20.04, and Arch Linux.
