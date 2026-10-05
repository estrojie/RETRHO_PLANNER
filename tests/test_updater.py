from __future__ import annotations

import unittest

import updater


class UpdaterTests(unittest.TestCase):
    def test_version_comparison(self):
        self.assertTrue(updater.version_is_newer("1.5", "1.4"))
        self.assertTrue(updater.version_is_newer("1.10", "1.9"))
        self.assertTrue(updater.version_is_newer("2.0.1", "2.0"))
        self.assertFalse(updater.version_is_newer("1.4", "1.4"))
        self.assertFalse(updater.version_is_newer("1.3", "1.4"))
        self.assertFalse(updater.version_is_newer("1.5", "dev-abcdef0"))

    def test_release_asset_selection(self):
        self.assertEqual(
            updater.release_asset_name(
                platform_name="win32", machine="AMD64", kind="windows-exe"
            ),
            "RHOPlanner-Windows-x64.exe",
        )
        self.assertEqual(
            updater.release_asset_name(
                platform_name="darwin", machine="arm64", kind="macos-app"
            ),
            "RHOPlanner-macOS-arm64.zip",
        )
        self.assertEqual(
            updater.release_asset_name(
                platform_name="darwin", machine="x86_64", kind="macos-app"
            ),
            "RHOPlanner-macOS-x86_64.zip",
        )
        self.assertEqual(
            updater.release_asset_name(
                platform_name="linux", machine="x86_64", kind="linux-appimage"
            ),
            "RHOPlanner-Linux-x86_64.AppImage",
        )
        self.assertEqual(
            updater.release_asset_name(
                platform_name="linux", machine="x86_64", kind="linux-tar"
            ),
            "RHOPlanner-Linux-x86_64.tar.xz",
        )

    def test_checksum_parser(self):
        text = (
            "a" * 64 + "  RHOPlanner-Windows-x64.exe\n"
            + "b" * 64 + " *RHOPlanner-Linux-x86_64.AppImage\n"
        )
        values = updater.parse_checksum_file(text)
        self.assertEqual(values["RHOPlanner-Windows-x64.exe"], "a" * 64)
        self.assertEqual(values["RHOPlanner-Linux-x86_64.AppImage"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
