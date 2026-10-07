"""JPL Horizons ephemeris support for RHO Planner.

This module deliberately contains no Qt code. Horizons queries are kept here so
the GUI can run them in a worker thread and the normalization logic can be
regression tested without network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time


HORIZONS_QUANTITIES = "2,3,4,8,9,20,23,25,47"


@dataclass(frozen=True)
class EphemerisRequest:
    target: str
    planning_date: date
    site_lat_deg: float
    site_lon_deg: float
    site_elevation_m: float
    timezone_name: str
    step_minutes: int = 5
    id_type: Optional[str] = None

    def validate(self) -> None:
        if not str(self.target).strip():
            raise ValueError("Enter a Horizons target name, number, or designation.")
        if not 1 <= int(self.step_minutes) <= 60:
            raise ValueError("Ephemeris step must be between 1 and 60 minutes.")
        if not -90.0 <= float(self.site_lat_deg) <= 90.0:
            raise ValueError("Observer latitude is outside the valid range.")
        if not -180.0 <= float(self.site_lon_deg) <= 180.0:
            raise ValueError("Observer longitude is outside the valid range.")
        ZoneInfo(str(self.timezone_name))


def planning_window_utc(
    planning_date: date,
    timezone_name: str,
) -> tuple[datetime, datetime]:
    """Return the planner's 17:00-07:00 local observing window in UTC."""
    tz = ZoneInfo(str(timezone_name))
    start_local = datetime(
        planning_date.year,
        planning_date.month,
        planning_date.day,
        17,
        0,
        0,
        tzinfo=tz,
    )
    stop_local = start_local + timedelta(hours=14)
    return (
        start_local.astimezone(timezone.utc),
        stop_local.astimezone(timezone.utc),
    )


def horizons_epochs(request: EphemerisRequest) -> dict[str, str]:
    """Build the UTC range dictionary expected by astroquery.jplhorizons."""
    request.validate()
    start_utc, stop_utc = planning_window_utc(
        request.planning_date,
        request.timezone_name,
    )
    return {
        "start": start_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "stop": stop_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "step": f"{int(request.step_minutes)}m",
    }


def horizons_location(request: EphemerisRequest) -> dict[str, float]:
    """Return a topocentric Horizons site dictionary."""
    request.validate()
    return {
        "lon": float(request.site_lon_deg),
        "lat": float(request.site_lat_deg),
        "elevation": float(request.site_elevation_m) / 1000.0,
    }


def _column(frame: pd.DataFrame, *names: str) -> Optional[str]:
    direct = {str(c): str(c) for c in frame.columns}
    for name in names:
        if name in direct:
            return direct[name]

    lowered = {str(c).strip().lower(): str(c) for c in frame.columns}
    for name in names:
        found = lowered.get(str(name).strip().lower())
        if found is not None:
            return found
    return None


def _numeric(frame: pd.DataFrame, *names: str) -> pd.Series:
    col = _column(frame, *names)
    if col is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _string(frame: pd.DataFrame, *names: str, default: str = "") -> pd.Series:
    col = _column(frame, *names)
    if col is None:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[col].where(frame[col].notna(), default).astype(str)


def _format_ra(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return Angle(float(value) * u.deg).to_string(
        unit=u.hourangle,
        sep=":",
        precision=2,
        pad=True,
    )


def _format_dec(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return Angle(float(value) * u.deg).to_string(
        unit=u.deg,
        sep=":",
        precision=1,
        pad=True,
        alwayssign=True,
    )


def normalize_horizons_table(
    table,
    timezone_name: str,
    fallback_target_name: str = "",
) -> pd.DataFrame:
    """Convert an astroquery Horizons table into RHO Planner columns."""
    if table is None or len(table) == 0:
        raise ValueError("JPL Horizons returned no ephemeris rows.")

    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("JPL Horizons returned no ephemeris rows.")

    jd = _numeric(frame, "datetime_jd")
    if jd.isna().all():
        raise ValueError("Horizons response did not contain usable UTC epochs.")

    utc_datetimes = Time(
        jd.to_numpy(dtype=float),
        format="jd",
        scale="utc",
    ).to_datetime(timezone=timezone.utc)
    time_utc = pd.DatetimeIndex(pd.to_datetime(utc_datetimes, utc=True))
    time_local = time_utc.tz_convert(ZoneInfo(str(timezone_name)))

    ra_deg = _numeric(frame, "RA_app", "RA", "RA_ICRF_app")
    dec_deg = _numeric(frame, "DEC_app", "DEC", "DEC_ICRF_app")
    ra_rate = _numeric(frame, "RA_rate", "RA_ICRF_rate_app")
    dec_rate = _numeric(frame, "DEC_rate", "DEC_ICRF_rate_app")
    sky_motion = _numeric(frame, "Sky_motion", "sky_motion")

    fallback_motion = np.hypot(
        ra_rate.to_numpy(dtype=float),
        dec_rate.to_numpy(dtype=float),
    ) / 60.0
    sky_values = sky_motion.to_numpy(dtype=float)
    missing = ~np.isfinite(sky_values)
    sky_values[missing] = fallback_motion[missing]

    target_name = _string(frame, "targetname", default=str(fallback_target_name))
    target_name = target_name.replace("", str(fallback_target_name))

    result = pd.DataFrame({
        "target": target_name.to_numpy(dtype=object),
        "time_local": time_local,
        "time_utc": time_utc,
        "ra_deg": ra_deg.to_numpy(dtype=float),
        "dec_deg": dec_deg.to_numpy(dtype=float),
        "az_deg": _numeric(frame, "AZ").to_numpy(dtype=float),
        "alt_deg": _numeric(frame, "EL").to_numpy(dtype=float),
        "airmass": _numeric(frame, "airmass").to_numpy(dtype=float),
        "v_mag": _numeric(frame, "V").to_numpy(dtype=float),
        "ra_rate_arcsec_hr": ra_rate.to_numpy(dtype=float),
        "dec_rate_arcsec_hr": dec_rate.to_numpy(dtype=float),
        "sky_motion_arcsec_min": sky_values,
        "sky_motion_pa_deg": _numeric(
            frame, "Sky_mot_PA", "sky_mot_pa"
        ).to_numpy(dtype=float),
        "relative_velocity_angle_deg": _numeric(
            frame, "RelVel-ANG", "RelVel_ANG"
        ).to_numpy(dtype=float),
        "observer_range_au": _numeric(frame, "delta").to_numpy(dtype=float),
        "observer_range_rate_km_s": _numeric(
            frame, "delta_rate"
        ).to_numpy(dtype=float),
        "solar_elongation_deg": _numeric(frame, "elong").to_numpy(dtype=float),
        "moon_separation_deg": _numeric(
            frame, "lunar_elong"
        ).to_numpy(dtype=float),
        "moon_illumination_pct": _numeric(
            frame, "lunar_illum"
        ).to_numpy(dtype=float),
        "solar_presence": _string(
            frame, "solar_presence", default=""
        ).to_numpy(dtype=object),
        "lunar_presence": _string(
            frame, "lunar_presence", default=""
        ).to_numpy(dtype=object),
    })

    result["ra_hms"] = [_format_ra(v) for v in result["ra_deg"]]
    result["dec_dms"] = [_format_dec(v) for v in result["dec_deg"]]
    return result


def query_horizons_ephemeris(
    request: EphemerisRequest,
    *,
    horizons_cls=None,
) -> pd.DataFrame:
    """Query JPL Horizons and return a normalized topocentric ephemeris."""
    request.validate()
    if horizons_cls is None:
        from astroquery.jplhorizons import Horizons

        horizons_cls = Horizons

    kwargs = {
        "id": str(request.target).strip(),
        "location": horizons_location(request),
        "epochs": horizons_epochs(request),
    }
    if request.id_type:
        kwargs["id_type"] = str(request.id_type)

    obj = horizons_cls(**kwargs)
    table = obj.ephemerides(
        quantities=HORIZONS_QUANTITIES,
        extra_precision=True,
        cache=True,
    )
    return normalize_horizons_table(
        table,
        request.timezone_name,
        fallback_target_name=str(request.target).strip(),
    )


def altitude_windows(
    frame: pd.DataFrame,
    min_alt_deg: float,
    max_alt_deg: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return sampled local-time intervals inside the RHO altitude limits."""
    if frame is None or frame.empty:
        return []

    alt = pd.to_numeric(frame["alt_deg"], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(alt)
        & (alt >= float(min_alt_deg))
        & (alt <= float(max_alt_deg))
    )
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []

    split_at = np.where(np.diff(positions) != 1)[0] + 1
    groups = np.split(positions, split_at)
    times = pd.DatetimeIndex(frame["time_local"])
    return [(times[g[0]], times[g[-1]]) for g in groups if len(g)]
