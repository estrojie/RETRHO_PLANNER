"""Build-time application version.

GitHub Actions rewrites this file in the build workspace before PyInstaller
runs. Source checkouts intentionally retain the development fallback.
"""

APP_VERSION = "0.0.0-dev"
