# RETRHO Planner trusted release setup

The release workflow now uses:
- Windows: PyInstaller onedir -> Inno Setup -> Microsoft Artifact Signing.
- macOS: PyInstaller app -> Developer ID + Hardened Runtime -> notarization/stapling -> DMG.
- Linux: existing AppImage/tarball plus a Debian/Ubuntu .deb package.

Manual workflow runs remain unsigned test builds. Tagged v* releases require signing configuration and will fail rather than publish unsigned Windows/macOS packages.

## Windows GitHub configuration

Secrets:
- AZURE_CLIENT_ID
- AZURE_TENANT_ID
- AZURE_SUBSCRIPTION_ID

Variables:
- WINDOWS_SIGNING_ENDPOINT
- WINDOWS_SIGNING_ACCOUNT
- WINDOWS_SIGNING_PROFILE

Create a Microsoft Artifact Signing account and Public Trust certificate profile, configure GitHub OIDC/federated credentials for the repository, and grant the service principal the Artifact Signing Certificate Profile Signer role.

## macOS GitHub configuration

Secrets:
- APPLE_CERTIFICATE_P12_BASE64
- APPLE_CERTIFICATE_PASSWORD
- APPLE_ID
- APPLE_APP_SPECIFIC_PASSWORD

Variables:
- APPLE_TEAM_ID
- APPLE_SIGNING_IDENTITY

APPLE_SIGNING_IDENTITY should be the full Developer ID Application identity shown by:
security find-identity -v -p codesigning

Export that Developer ID Application certificate/private key to a password-protected .p12 and store its base64 content in APPLE_CERTIFICATE_P12_BASE64.

## Public release assets

- RHOPlanner-Windows-x64-Setup.exe
- RHOPlanner-macOS-arm64.dmg
- RHOPlanner-macOS-arm64.zip
- RHOPlanner-macOS-x86_64.dmg
- RHOPlanner-macOS-x86_64.zip
- RHOPlanner-Linux-x86_64.AppImage
- RHOPlanner-Linux-x86_64.tar.gz
- RHOPlanner-Linux-amd64.deb
- SHA256SUMS.txt

Linux runtime/Qt behavior is intentionally unchanged by this packaging update.
