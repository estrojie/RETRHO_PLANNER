import unittest
from unittest.mock import patch

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import planner_core as core


class ExposureRegressionTests(unittest.TestCase):
    def test_same_band_color_is_rejected(self):
        ok, reason = core.color_constraint_status("Sloan g", "Sloan g", "r")
        self.assertFalse(ok)
        self.assertIn("same", reason)

    def test_extreme_color_extrapolation_is_rejected(self):
        ok, reason = core.color_constraint_status("2MASS J", "2MASS Ks", "g")
        self.assertFalse(ok)
        self.assertIn("outside", reason)

    def test_surface_line_flux_increases_extended_narrowband_signal(self):
        cfg = core.ExposureCalculatorConfig(
            desired_peak_counts_adu=None,
            max_narrowband_subexposure_s=300.0,
        )
        base = core.ExposureTarget(
            reference_mag_ab=24.0,
            reference_band="Sloan r",
            source_type="extended",
            measurement_area_arcsec2=100.0,
            target_snr=20.0,
        )
        with_line = core.ExposureTarget(
            reference_mag_ab=24.0,
            reference_band="Sloan r",
            source_type="extended",
            measurement_area_arcsec2=100.0,
            target_snr=20.0,
            line_surface_fluxes_erg_s_cm2_arcsec2={"H-alpha": 1e-15},
        )
        r0 = {r["filter"]: r for r in core.calculate_exposure_times(cfg, base)}
        r1 = {r["filter"]: r for r in core.calculate_exposure_times(cfg, with_line)}
        self.assertGreater(r1["H-alpha"]["source_rate_e_s"], r0["H-alpha"]["source_rate_e_s"])

    def test_rayleigh_conversion_is_positive_and_wavelength_dependent(self):
        ha = core.rayleigh_to_erg_s_cm2_arcsec2(100.0, 656.3)
        hb = core.rayleigh_to_erg_s_cm2_arcsec2(100.0, 486.1)
        self.assertGreater(ha, 0.0)
        self.assertGreater(hb, ha)

    def test_adc_limit_caps_saturation(self):
        target = core.ExposureTarget(
            reference_mag_ab=8.0,
            reference_band="Johnson V",
            target_snr=20.0,
        )
        high_adc = core.ExposureCalculatorConfig(
            adc_max_adu=65535.0,
            desired_peak_counts_adu=None,
        )
        low_adc = core.ExposureCalculatorConfig(
            adc_max_adu=5000.0,
            desired_peak_counts_adu=None,
        )
        hi = {r["filter"]: r for r in core.calculate_exposure_times(high_adc, target)}
        lo = {r["filter"]: r for r in core.calculate_exposure_times(low_adc, target)}
        self.assertLess(lo["r"]["saturation_time_s"], hi["r"]["saturation_time_s"])

    def test_binning_full_well_does_not_reduce_capacity(self):
        target = core.ExposureTarget(
            reference_mag_ab=8.0,
            reference_band="Johnson V",
            target_snr=20.0,
        )
        cfg1 = core.ExposureCalculatorConfig(
            adc_max_adu=1e9,
            binning_factor=1,
            pixel_scale_arcsec=0.54,
            desired_peak_counts_adu=None,
        )
        cfg3 = core.ExposureCalculatorConfig(
            adc_max_adu=1e9,
            binning_factor=3,
            pixel_scale_arcsec=1.62,
            desired_peak_counts_adu=None,
        )
        a = {r["filter"]: r for r in core.calculate_exposure_times(cfg1, target)}
        b = {r["filter"]: r for r in core.calculate_exposure_times(cfg3, target)}
        self.assertGreaterEqual(b["r"]["saturation_time_s"], a["r"]["saturation_time_s"])

    def test_rectangular_finder_render_tracks_requested_shape(self):
        data = np.arange(200 * 200, dtype=float).reshape(200, 200)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crpix = [100.5, 100.5]
        wcs.wcs.crval = [180.0, 20.0]
        wcs.wcs.cdelt = [-10.0 / 3600.0, 10.0 / 3600.0]
        coord = SkyCoord(180.0 * u.deg, 20.0 * u.deg, frame="icrs")

        fig = core.render_finder_figure_from_data(
            coord, "Test", data, wcs, 15, "local", fov_h_arcmin=20
        )
        try:
            ax = fig.axes[0]
            ny, nx = ax._rho_data_shape
            self.assertGreater(ny, nx)
        finally:
            core.plt.close(fig)


class FinderIdentificationTests(unittest.TestCase):
    def test_friendly_simbad_name_beats_gaia_identifier(self):
        candidate = {
            "main_id": "Gaia DR3 123456789",
            "aliases": ["Gaia DR3 123456789"],
            "catalog": "Gaia DR3",
            "coord": SkyCoord(133.1492 * u.deg, 28.3308 * u.deg, frame="icrs"),
        }
        with patch.object(
            core,
            "_simbad_identifier_aliases",
            return_value=["NAME Copernicus", "55 Cnc", "HD 75732"],
        ):
            enriched = core._enrich_candidate_identifiers(candidate)
        self.assertEqual(enriched["main_id"], "Copernicus")
        self.assertIn("55 Cnc", enriched["aliases"])

    def test_placeholder_figure_uses_dark_background(self):
        fig = core.placeholder_figure("No data yet")
        try:
            rgba = fig.get_facecolor()
            self.assertLess(rgba[0], 0.2)
            self.assertLess(rgba[1], 0.2)
            self.assertLess(rgba[2], 0.2)
        finally:
            core.plt.close(fig)


class PhotometryInputTests(unittest.TestCase):
    def test_common_aliases(self):
        self.assertEqual(core.get_reference_magnitude_band("APASS V").key, "Johnson V")
        self.assertEqual(core.get_reference_magnitude_band("PS1 r").key, "Pan-STARRS r")
        self.assertEqual(core.get_reference_magnitude_band("Ks").key, "2MASS Ks")


if __name__ == "__main__":
    unittest.main()
