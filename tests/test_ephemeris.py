from __future__ import annotations

import unittest
from datetime import date

import numpy as np
from astropy.table import Table

import ephemeris


class _FakeHorizons:
    last_init = None
    last_query = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs

    def ephemerides(self, **kwargs):
        type(self).last_query = kwargs
        return Table({
            "targetname": ["433 Eros", "433 Eros"],
            "datetime_jd": [2461321.375, 2461321.378472222],
            "RA_app": [12.0, 12.1],
            "DEC_app": [20.0, 20.1],
            "RA_rate": [60.0, 120.0],
            "DEC_rate": [0.0, 0.0],
            "AZ": [110.0, 111.0],
            "EL": [30.0, 31.0],
            "airmass": [1.8, 1.7],
            "V": [11.2, 11.2],
            "delta": [0.5, 0.5],
            "delta_rate": [-2.0, -2.0],
            "elong": [100.0, 100.2],
            "lunar_elong": [70.0, 70.1],
            "lunar_illum": [15.0, 15.0],
            "Sky_motion": [1.0, 2.0],
            "Sky_mot_PA": [90.0, 91.0],
            "RelVel-ANG": [-10.0, -9.0],
        })


class EphemerisTests(unittest.TestCase):
    def _request(self):
        return ephemeris.EphemerisRequest(
            target="433",
            planning_date=date(2026, 10, 7),
            site_lat_deg=29.400041,
            site_lon_deg=-82.585953,
            site_elevation_m=31.0,
            timezone_name="US/Eastern",
            step_minutes=5,
            id_type="smallbody",
        )

    def test_planning_window_is_converted_from_local_to_utc(self):
        epochs = ephemeris.horizons_epochs(self._request())
        self.assertEqual(epochs["start"], "2026-10-07 21:00:00")
        self.assertEqual(epochs["stop"], "2026-10-08 11:00:00")
        self.assertEqual(epochs["step"], "5m")

    def test_query_uses_topocentric_site_and_requested_quantities(self):
        result = ephemeris.query_horizons_ephemeris(
            self._request(),
            horizons_cls=_FakeHorizons,
        )
        location = _FakeHorizons.last_init["location"]
        self.assertAlmostEqual(location["lat"], 29.400041)
        self.assertAlmostEqual(location["lon"], -82.585953)
        self.assertAlmostEqual(location["elevation"], 0.031)
        self.assertEqual(_FakeHorizons.last_init["id_type"], "smallbody")
        self.assertEqual(
            _FakeHorizons.last_query["quantities"],
            ephemeris.HORIZONS_QUANTITIES,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "target"], "433 Eros")

    def test_quantity_47_sky_motion_is_preserved(self):
        result = ephemeris.query_horizons_ephemeris(
            self._request(),
            horizons_cls=_FakeHorizons,
        )
        np.testing.assert_allclose(
            result["sky_motion_arcsec_min"].to_numpy(),
            [1.0, 2.0],
        )

    def test_sky_motion_falls_back_to_rate_components(self):
        table = _FakeHorizons().ephemerides()
        table.remove_column("Sky_motion")
        result = ephemeris.normalize_horizons_table(
            table,
            "US/Eastern",
            "433",
        )
        np.testing.assert_allclose(
            result["sky_motion_arcsec_min"].to_numpy(),
            [1.0, 2.0],
        )

    def test_altitude_windows_respect_planner_limits(self):
        result = ephemeris.query_horizons_ephemeris(
            self._request(),
            horizons_cls=_FakeHorizons,
        )
        windows = ephemeris.altitude_windows(result, 30.5, 62.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], result.loc[1, "time_local"])
        self.assertEqual(windows[0][1], result.loc[1, "time_local"])


if __name__ == "__main__":
    unittest.main()
