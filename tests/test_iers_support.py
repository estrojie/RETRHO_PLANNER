"""Regression coverage for offline IERS planning and safe updates."""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import astropy.units as u
import requests
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

import iers_support


class IERSMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.cache = Path(self.tempdir.name) / "finals2000A.all"
        cache_patcher = mock.patch.object(
            iers_support, "_cache_path", return_value=self.cache
        )
        cache_patcher.start()
        self.addCleanup(cache_patcher.stop)
        self.previous_table = iers.earth_orientation_table.get()
        self.addCleanup(iers.earth_orientation_table.set, self.previous_table)

    def _save_bundled_table(self, stale=False):
        shutil.copyfile(iers.IERS_A_FILE, self.cache)
        if stale:
            old = time.time() - 9 * 24 * 60 * 60
            os.utime(self.cache, (old, old))

    def test_offline_configuration_does_not_reject_old_predictions(self):
        iers_support.configure_iers()
        self.assertFalse(iers.conf.auto_download)
        self.assertIsNone(iers.conf.auto_max_age)

        # Future date deliberately outside the packaged IERS prediction span.
        location = EarthLocation(lat=29.400041 * u.deg, lon=-82.585953 * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alt = SkyCoord(10 * u.deg, 20 * u.deg).transform_to(
                AltAz(obstime=Time("2040-10-27"), location=location)
            ).alt.deg
        self.assertTrue(math.isfinite(float(alt)))

    def test_recent_cached_file_is_used_without_network(self):
        self._save_bundled_table()
        with mock.patch.object(iers_support.requests, "get") as get:
            result = iers_support.refresh_iers_now()
        self.assertEqual(result, "cached")
        get.assert_not_called()
        self.assertIn("current", iers_support.status_message())

    def test_offline_uses_stale_cached_file_without_deleting_it(self):
        self._save_bundled_table(stale=True)
        with mock.patch.object(
            iers_support.requests, "get",
            side_effect=requests.ConnectionError("network unavailable"),
        ) as get:
            result = iers_support.refresh_iers_now()
        self.assertEqual(result, "offline")
        self.assertEqual(get.call_count, 2)
        self.assertTrue(self.cache.exists())
        self.assertIn("Cached", iers_support.status_message())

    def test_offline_first_run_uses_bundled_table(self):
        with mock.patch.object(
            iers_support.requests, "get",
            side_effect=requests.ConnectionError("offline"),
        ):
            self.assertEqual(iers_support.refresh_iers_now(), "offline")
        self.assertFalse(self.cache.exists())
        self.assertIn("Bundled", iers_support.status_message())

    def test_valid_download_replaces_cache_atomically(self):
        self._save_bundled_table(stale=True)
        bundled_bytes = Path(iers.IERS_A_FILE).read_bytes()
        response = mock.Mock(status_code=200, content=bundled_bytes)
        with mock.patch.object(iers_support.requests, "get", return_value=response):
            result = iers_support.refresh_iers_now()
        self.assertEqual(result, "updated")
        self.assertEqual(self.cache.read_bytes(), bundled_bytes)
        self.assertIn("online", iers_support.status_message())

    def test_malformed_download_does_not_destroy_cached_table(self):
        self._save_bundled_table(stale=True)
        original = self.cache.read_bytes()
        response = mock.Mock(status_code=200, content=b"invalid data")
        with mock.patch.object(iers_support.requests, "get", return_value=response):
            self.assertEqual(iers_support.refresh_iers_now(), "offline")
        self.assertEqual(self.cache.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
