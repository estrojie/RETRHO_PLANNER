# planner_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo
import re
import math

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.utils import iers
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from io import BytesIO, StringIO
from astroplan import Observer, FixedTarget
# RHO Planner ships astropy-iers-data, so use the bundled table at startup.
# This avoids first-run network downloads delaying the GUI by several seconds.
iers.conf.auto_download = False

DEFAULT_LAT          = 29.400041
DEFAULT_LON          = -82.585953
DEFAULT_HEIGHT_M     = 31
DEFAULT_TZ           = "US/Eastern"
DEFAULT_SITE_NAME    = "RHO (Bronson, FL)"
DEFAULT_MIN_ALT_DEG  = 26.0
DEFAULT_MAX_ALT_DEG  = 62.0
DEFAULT_FOV1_ARCMIN  = 90
DEFAULT_FOV2_ARCMIN  = 20
DEC_WARNING_LIMIT_DEG = 60.0

SKYVIEW_SURVEYS       = ["DSS2 Red", "DSS2 Blue", "DSS"]
PANSTARRS_FILTER      = "r"
PANSTARRS_FILENAME_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PANSTARRS_FITSCUT_URL  = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

NWS_HEADERS = {
    "User-Agent": "RHOPlanner/1.0",
    "Accept": "application/geo+json,application/json",
}

_custom_simbad = None
_vizier_client = None
_skyview_class = None


def _get_simbad():
    """Create the SIMBAD client only when a catalogue lookup is requested."""
    global _custom_simbad
    if _custom_simbad is None:
        from astroquery.simbad import Simbad

        client = Simbad()
        try:
            client.add_votable_fields("flux(V)")
        except Exception:
            pass
        _custom_simbad = client
    return _custom_simbad


def _get_vizier_client():
    """Create the VizieR client only when the finder inspector needs it."""
    global _vizier_client
    if _vizier_client is None:
        from astroquery.vizier import Vizier

        _vizier_client = Vizier(columns=["*"], row_limit=200)
    return _vizier_client


def _get_skyview_class():
    """Import and compatibility-patch SkyView on first finder-chart request."""
    global _skyview_class
    if _skyview_class is None:
        import inspect
        from astroquery.skyview import SkyView

        if "grid" not in inspect.signature(SkyView.get_images).parameters:
            original_get_images = SkyView.get_images

            def _get_images_no_grid(*args, **kwargs):
                kwargs.pop("grid", None)
                return original_get_images(*args, **kwargs)

            SkyView.get_images = _get_images_no_grid

        _skyview_class = SkyView
    return _skyview_class

def _normalize_to_uint8(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(vmin) or vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)
    return np.round(np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)


def _rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s,  c]], dtype=float)

def _build_output_wcs(
    ref_wcs: WCS,
    center_coord: SkyCoord,
    shape: tuple[int, int],
    roll_deg: float = 0.0,
) -> WCS:
    ny, nx = int(shape[0]), int(shape[1])
    out = WCS(naxis=2)

    try:
        ctype = list(ref_wcs.wcs.ctype)
    except Exception:
        ctype = ["RA---TAN", "DEC--TAN"]

    try:
        cunit = [str(cu) if str(cu).strip() else "deg" for cu in ref_wcs.wcs.cunit]
    except Exception:
        cunit = ["deg", "deg"]

    try:
        scales_deg = np.abs(proj_plane_pixel_scales(ref_wcs.celestial))
    except Exception:
        if getattr(ref_wcs.wcs, "cdelt", None) is not None and len(ref_wcs.wcs.cdelt) >= 2:
            scales_deg = np.abs(np.asarray(ref_wcs.wcs.cdelt[:2], dtype=float))
        else:
            scales_deg = np.array([1.0 / 3600.0, 1.0 / 3600.0], dtype=float)

    cdelt = np.array([-float(scales_deg[0]), float(scales_deg[1])], dtype=float)

    rot = _rotation_matrix_deg(float(roll_deg))
    out.wcs.pc    = rot
    out.wcs.ctype = ctype
    out.wcs.cunit = cunit
    out.wcs.cdelt = cdelt
    out.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    out.wcs.crval = [float(center_coord.icrs.ra.deg), float(center_coord.icrs.dec.deg)]
    try:
        out.wcs.set()
    except Exception:
        pass
    return out

def _reproject_to_output_wcs(
    data: np.ndarray,
    input_wcs: WCS,
    output_wcs: WCS,
    shape_out: tuple[int, int],
    fill_value: float = np.nan,
) -> np.ndarray:
    arr  = np.asarray(data, dtype=float)
    ny, nx = int(shape_out[0]), int(shape_out[1])
    yy, xx = np.indices((ny, nx), dtype=float)

    world = output_wcs.pixel_to_world(xx, yy)
    xin, yin = input_wcs.world_to_pixel(world)
    xin = np.asarray(xin, dtype=float)
    yin = np.asarray(yin, dtype=float)

    # Bilinear sampling implemented with NumPy.  This replaces the single
    # scipy.ndimage.map_coordinates use in the application, removing SciPy from
    # the desktop bundle while preserving smooth finder-chart reprojection.
    fv = float(fill_value) if np.isfinite(fill_value) else 0.0
    invalid = (~np.isfinite(xin)) | (~np.isfinite(yin)) | \
              (xin < -0.5) | (yin < -0.5) | \
              (xin > (arr.shape[1] - 0.5)) | (yin > (arr.shape[0] - 0.5))

    safe_x = np.where(np.isfinite(xin), xin, 0.0)
    safe_y = np.where(np.isfinite(yin), yin, 0.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    dx = safe_x - x0
    dy = safe_y - y0

    # A one-pixel constant border reproduces map_coordinates(mode="constant")
    # near the image edge without an additional compiled dependency.
    padded = np.pad(arr, 1, mode="constant", constant_values=fv)
    x0p = np.clip(x0 + 1, 0, padded.shape[1] - 1)
    y0p = np.clip(y0 + 1, 0, padded.shape[0] - 1)
    x1p = np.clip(x0p + 1, 0, padded.shape[1] - 1)
    y1p = np.clip(y0p + 1, 0, padded.shape[0] - 1)

    sampled = (
        padded[y0p, x0p] * (1.0 - dx) * (1.0 - dy)
        + padded[y0p, x1p] * dx * (1.0 - dy)
        + padded[y1p, x0p] * (1.0 - dx) * dy
        + padded[y1p, x1p] * dx * dy
    )
    sampled = np.asarray(sampled, dtype=float)
    sampled[invalid] = np.nan
    return sampled

def regenerate_rolled_finder_reprojected(
    data: np.ndarray,
    input_wcs: WCS,
    center_coord: SkyCoord,
    final_shape: tuple[int, int],
    roll_deg: float,
) -> tuple[np.ndarray, WCS]:
    out_wcs = _build_output_wcs(input_wcs, center_coord, final_shape, roll_deg=roll_deg)
    finite  = data[np.isfinite(data)]
    fv      = float(np.nanmedian(finite)) if finite.size else 0.0
    reproj  = _reproject_to_output_wcs(data, input_wcs, out_wcs, final_shape, fill_value=fv)
    return reproj, out_wcs

def rotate_point_about_center(
    x: float, y: float,
    width: float, height: float,
    angle_deg: float,
) -> tuple[float, float]:
    cx = (float(width)  - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    theta = np.deg2rad(float(angle_deg))
    c, s  = float(np.cos(theta)), float(np.sin(theta))
    dx, dy = float(x) - cx, float(y) - cy
    return dx * c - dy * s + cx, dx * s + dy * c + cy

@dataclass
class SiteConfig:
    lat:      float = DEFAULT_LAT
    lon:      float = DEFAULT_LON
    height_m: float = DEFAULT_HEIGHT_M
    timezone: str   = DEFAULT_TZ
    name:     str   = DEFAULT_SITE_NAME

_CURRENT_SITE:     SiteConfig     = SiteConfig()
_CURRENT_OBSERVER: Observer | None = None
_PLANNING_DATE:    date           = date.today()

def set_site(lat: float, lon: float, height_m: float, timezone_str: str, name: str | None = None) -> None:
    global _CURRENT_SITE, _CURRENT_OBSERVER
    _CURRENT_SITE = SiteConfig(
        lat      = float(lat),
        lon      = float(lon),
        height_m = float(height_m),
        timezone = str(timezone_str),
        name     = str(name) if name else f"Site ({lat:.4f}, {lon:.4f})",
    )
    _CURRENT_OBSERVER = None  

def set_planning_date(d: date) -> None:
    global _PLANNING_DATE
    _PLANNING_DATE = d

def get_planning_date() -> date:
    return _PLANNING_DATE

def get_observer() -> Observer:
    global _CURRENT_OBSERVER
    if _CURRENT_OBSERVER is None:
        loc = EarthLocation(
            lat    = _CURRENT_SITE.lat    * u.deg,
            lon    = _CURRENT_SITE.lon    * u.deg,
            height = _CURRENT_SITE.height_m * u.m,
        )
        _CURRENT_OBSERVER = Observer(
            location = loc,
            timezone = _CURRENT_SITE.timezone,
            name     = _CURRENT_SITE.name,
        )
    return _CURRENT_OBSERVER

def get_site_config() -> SiteConfig:
    return _CURRENT_SITE

def _parse_iso_duration_to_seconds(dur: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    if not m:
        return 0
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)

def _parse_valid_time(valid_time: str):
    try:
        start_str, dur_str = valid_time.split("/")
        start   = pd.to_datetime(start_str, utc=True).to_pydatetime()
        seconds = _parse_iso_duration_to_seconds(dur_str)
        return start, start + timedelta(seconds=seconds)
    except Exception:
        return None, None

def get_cloud_cover_now_next(lat: float, lon: float, timeout_s: float = 6.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cloud_now_pct": None, "cloud_now_valid": None,
        "cloud_next_pct": None, "cloud_next_valid": None,
    }
    try:
        meta_resp = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            timeout=timeout_s, headers=NWS_HEADERS,
        )
        meta_resp.raise_for_status()
        props = meta_resp.json().get("properties", {})
        grid, x, y = props.get("gridId"), props.get("gridX"), props.get("gridY")
        if None in (grid, x, y):
            return out

        grid_resp = requests.get(
            f"https://api.weather.gov/gridpoints/{grid}/{x},{y}",
            timeout=timeout_s, headers=NWS_HEADERS,
        )
        grid_resp.raise_for_status()
        sky_vals = grid_resp.json().get("properties", {}).get("skyCover", {}).get("values", [])
        if not sky_vals:
            return out

        now_utc = datetime.now(timezone.utc)
        parsed  = []
        for entry in sky_vals:
            vt, val = entry.get("validTime"), entry.get("value")
            if vt is None or val is None:
                continue
            start, end = _parse_valid_time(vt)
            if start and end:
                parsed.append((start, end, float(val), vt))
        if not parsed:
            return out

        parsed.sort(key=lambda t: t[0])
        idx = next((i for i, (s, e, *_) in enumerate(parsed) if s <= now_utc <= e), None)
        if idx is None:
            idx = min(range(len(parsed)), key=lambda i: abs((parsed[i][0] - now_utc).total_seconds()))

        out["cloud_now_pct"],  out["cloud_now_valid"]  = parsed[idx][2],     parsed[idx][3]
        if idx + 1 < len(parsed):
            out["cloud_next_pct"], out["cloud_next_valid"] = parsed[idx+1][2], parsed[idx+1][3]
    except Exception:
        pass
    return out

@dataclass
class ResolvedTarget:
    display_name: str
    coord:        SkyCoord
    vmag:         Any
    method:       str

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def parse_radec(ra: str, dec: str) -> SkyCoord:
    ra  = (ra  or "").strip().replace(" ", ":")
    dec = (dec or "").strip().replace(" ", ":")
    if not ra or not dec:
        raise ValueError("RA/Dec required for manual fallback.")
    ra_unit = u.deg if (_is_number(ra) and 0.0 <= float(ra) <= 360.0) else u.hourangle
    return SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame="icrs")

def resolve_target(name: str, ra: str, dec: str) -> ResolvedTarget:
    name = (name or "").strip()
    ra   = (ra   or "").strip()
    dec  = (dec  or "").strip()

    if name:
        try:
            coord = SkyCoord.from_name(name)
            vmag  = "N/A"
            try:
                result = _get_simbad().query_object(name)
                if result is not None:
                    for col_try in ("FLUX_V", "V", "flux(V)", "flux_V"):
                        if col_try in result.colnames:
                            vmag = result[col_try][0]
                            break
            except Exception:
                pass
            return ResolvedTarget(display_name=name, coord=coord, vmag=vmag, method="SIMBAD name")
        except Exception:
            pass

    coord = parse_radec(ra, dec)
    return ResolvedTarget(
        display_name = name if name else "Unnamed Target",
        coord        = coord,
        vmag         = "N/A",
        method       = "Manual RA/Dec",
    )

def planning_window_times(step_min: int = 2) -> Time:
    obs = get_observer()
    tz  = ZoneInfo(get_site_config().timezone)
    d   = get_planning_date()
    start_local = datetime(d.year, d.month, d.day, 17, 0, 0, tzinfo=tz)
    start = Time(start_local)
    end   = Time(start_local + timedelta(hours=14))
    n     = int(np.floor(((end - start).to(u.min).value) / step_min))
    return start + np.arange(0, n + 1) * step_min * u.min


def compute_visibility_windows(
    coord: SkyCoord,
    min_alt_deg: float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg: float = DEFAULT_MAX_ALT_DEG,
    step_min:    int   = 2,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    obs   = get_observer()
    times = planning_window_times(step_min=step_min)
    alt   = obs.altaz(times, coord).alt.deg
    mask  = (alt >= min_alt_deg) & (alt <= max_alt_deg)

    if not np.any(mask):
        return []

    idx    = np.where(mask)[0]
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    tz     = obs.timezone

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for g in groups:
        if len(g) == 0:
            continue
        t1 = pd.Timestamp(times[g[ 0]].to_datetime(timezone=tz))
        t2 = pd.Timestamp(times[g[-1]].to_datetime(timezone=tz))
        windows.append((t1, t2))
    return windows

def compute_visibility_window(
    coord: SkyCoord,
    min_alt_deg: float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg: float = DEFAULT_MAX_ALT_DEG,
    step_min:    int   = 2,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    windows = compute_visibility_windows(coord, min_alt_deg, max_alt_deg, step_min=step_min)
    if not windows:
        return None, None
    return windows[0][0], windows[-1][1]

def format_visibility_windows(
    windows:  List[Tuple[pd.Timestamp, pd.Timestamp]],
    time_fmt: str = "%H:%M",
) -> str:
    if not windows:
        return "—"
    return "; ".join(f"{t1.strftime(time_fmt)}–{t2.strftime(time_fmt)}" for t1, t2 in windows)

def sky_conditions(
    timeout_s: float = 6.0,
    site: SiteConfig | None = None,
    planning_date: date | None = None,
) -> Dict[str, Any]:
    """Return sky conditions for an immutable site/date snapshot."""
    out = {}
    site_cfg = site or get_site_config()
    d = planning_date or get_planning_date()

    if site is None:
        obs = get_observer()
    else:
        loc = EarthLocation(
            lat=site_cfg.lat * u.deg,
            lon=site_cfg.lon * u.deg,
            height=site_cfg.height_m * u.m,
        )
        obs = Observer(
            location=loc,
            timezone=site_cfg.timezone,
            name=site_cfg.name,
        )

    out.update(
        get_cloud_cover_now_next(
            site_cfg.lat, site_cfg.lon, timeout_s=timeout_s
        )
    )

    tz = ZoneInfo(site_cfg.timezone)
    anchor = Time(datetime(d.year, d.month, d.day, 20, 0, 0, tzinfo=tz))

    try:
        out["sunset_local"] = obs.sun_set_time(anchor, which="nearest").to_datetime(timezone=obs.timezone)
    except Exception:
        out["sunset_local"] = None

    try:
        out["moon_alt_deg"] = float(obs.moon_altaz(anchor).alt.deg)
    except Exception:
        out["moon_alt_deg"] = None

    try:
        out["moon_illum_frac"] = float(obs.moon_illumination(anchor))
    except Exception:
        out["moon_illum_frac"] = None

    return out

def _mask_to_spans(dt_list, mask: np.ndarray):
    spans = []
    mask  = np.asarray(mask, dtype=bool)
    i, n  = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        spans.append((dt_list[i], dt_list[j]))
        i = j + 1
    return spans

def plot_altitudes(
    coords:       List[SkyCoord],
    names:        List[str],
    min_alt_deg:  float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg:  float = DEFAULT_MAX_ALT_DEG,
    y_mode:       str   = "altitude",
    only_names:   Optional[List[str]] = None,
    display_tz:   str   = "local",
) -> plt.Figure:
    obs   = get_observer()
    times = planning_window_times(step_min=2)

    if str(display_tz).lower() in ("utc", "z"):
        tz_disp, xlab = timezone.utc, "UTC"
    else:
        tz_disp, xlab = obs.timezone, "Local Time"

    dts      = [t.to_datetime(timezone=tz_disp) for t in times]
    sun_alt  = np.array(obs.sun_altaz(times).alt.deg)
    moon_alt = np.array(obs.moon_altaz(times).alt.deg)

    if only_names is not None:
        keep   = [(c, n) for c, n in zip(coords, names) if n in set(only_names)]
        coords = [c for c, _ in keep]
        names  = [n for _, n in keep]

    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for spans, alpha in [
        (_mask_to_spans(dts, sun_alt > 0),                  0.10),
        (_mask_to_spans(dts, (sun_alt <= 0)  & (sun_alt > -6)),  0.15),
        (_mask_to_spans(dts, (sun_alt <= -6) & (sun_alt > -12)), 0.20),
        (_mask_to_spans(dts, (sun_alt <=-12) & (sun_alt > -18)), 0.25),
        (_mask_to_spans(dts, sun_alt <= -18),                0.30),
    ]:
        for a, b in spans:
            ax.axvspan(a, b, alpha=alpha)

    def alt_to_airmass(alt: np.ndarray) -> np.ndarray:
        r = np.deg2rad(alt)
        am = np.full_like(alt, np.nan, dtype=float)
        good = np.sin(r) > 0
        am[good] = 1.0 / np.sin(r[good])
        return am

    for coord, nm in zip(coords[:5], names[:5]):
        alt = obs.altaz(times, coord).alt.deg
        y   = alt_to_airmass(alt) if y_mode.lower().startswith("air") else alt
        ax.plot(dts, y, "-", label=nm, linewidth=2.0)

    ax.plot(dts, moon_alt, "--", linewidth=1.7, label="Moon (alt)")

    if y_mode.lower().startswith("air"):
        ax.set_ylabel("Airmass (sec z)", color="white")
        ax.set_ylim(1.0, 6.0)
        ax.invert_yaxis()
        ax.axhline(1.0 / np.sin(np.deg2rad(max_alt_deg)), linestyle="--", alpha=0.7)
        ax.axhline(1.0 / np.sin(np.deg2rad(min_alt_deg)), linestyle="--", alpha=0.7)
    else:
        ax.axhline(max_alt_deg, linestyle="--", alpha=0.7)
        ax.axhline(min_alt_deg, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 90)
        ax.set_ylabel("Altitude (°)", color="white")

    ax.set_xlabel(xlab, color="white", labelpad=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz_disp))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz_disp))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(colors="white")
    fig.autofmt_xdate(rotation=30)
    ax.margins(x=0.015, y=0.04)

    leg = ax.legend(framealpha=0.75, fontsize=8, ncol=1,
                    handlelength=1.5, labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color("white")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.24)
    plt.close(fig)
    return fig

def _get_skyview_hdu(coord: SkyCoord, fov_arcmin: int, pixels: int, surveys: List[str]):
    for survey in surveys:
        try:
            hdus = _get_skyview_class().get_images(
                position = coord,
                survey   = [survey],
                height   = fov_arcmin * u.arcmin,
                width    = fov_arcmin * u.arcmin,
                pixels   = pixels,
            )
            if hdus and len(hdus) > 0 and len(hdus[0]) > 0:
                return survey, hdus[0][0]
        except Exception:
            continue
    return None, None

def _get_panstarrs_hdu(coord: SkyCoord, fov_arcmin: int, pixels: int, filt: str = PANSTARRS_FILTER):
    try:
        params = {
            "ra": f"{float(coord.icrs.ra.deg):.8f}",
            "dec": f"{float(coord.icrs.dec.deg):.8f}",
            "filters": str(filt), "type": "stack", "sep": "comma",
        }
        resp = requests.get(PANSTARRS_FILENAME_URL, params=params, timeout=12)
        resp.raise_for_status()
        text = (resp.text or "").strip()
        if not text:
            return None

        df = pd.read_csv(StringIO(text))
        if df.empty or "filename" not in df.columns:
            return None

        filename = str(df.iloc[0]["filename"]).strip()
        if not filename:
            return None

        cutout_resp = requests.get(PANSTARRS_FITSCUT_URL, params={
            "ra":     f"{float(coord.icrs.ra.deg):.8f}",
            "dec":    f"{float(coord.icrs.dec.deg):.8f}",
            "size":   int(pixels),
            "format": "fits",
            "red":    filename,
        }, timeout=20)
        cutout_resp.raise_for_status()
        hdul = fits.open(BytesIO(cutout_resp.content))
        try:
            return hdul[0].copy()
        finally:
            hdul.close()
    except Exception:
        return None

FOV_MIN_ARCMIN: int = 1
FOV_MAX_ARCMIN: int = 180 
FINDER_PADDING_FACTOR: float = 1.25
FINDER_PADDING_MIN_ARCMIN: float = 6.0
FINDER_FETCH_MAX_ARCMIN: int = 320
FINDER_FETCH_MAX_PIXELS: int = 1800

def _clamp_fov(v: int) -> int:
    return max(FOV_MIN_ARCMIN, min(FOV_MAX_ARCMIN, int(v)))

def _wcs_coverage_arcmin(data: np.ndarray, wcs: WCS) -> tuple[float, float]:
    """Approximate width/height of an image footprint in the tangent plane."""
    arr = np.squeeze(np.asarray(data))
    if arr.ndim != 2:
        return 0.0, 0.0
    try:
        scales = np.abs(proj_plane_pixel_scales(wcs.celestial)) * 60.0
        return float(arr.shape[1] * scales[0]), float(arr.shape[0] * scales[1])
    except Exception:
        return 0.0, 0.0

def fetch_finder_raw(
    coord:        SkyCoord,
    fov_w_arcmin: int,
    mode:         str,
    fov_h_arcmin: int | None = None,
    pixels:       int        = 800,
) -> tuple[np.ndarray | None, WCS | None, str]:
    import math

    fov_w = _clamp_fov(fov_w_arcmin)
    fov_h = _clamp_fov(fov_h_arcmin if fov_h_arcmin is not None else fov_w)

    diagonal_arcmin = math.hypot(float(fov_w), float(fov_h))
    safety_arcmin = max(
        FINDER_PADDING_MIN_ARCMIN,
        diagonal_arcmin * (FINDER_PADDING_FACTOR - 1.0),
    )
    fov_padded = int(math.ceil(diagonal_arcmin + safety_arcmin))
    fov_padded = min(fov_padded, FINDER_FETCH_MAX_ARCMIN)

    requested_max_axis = max(float(fov_w), float(fov_h), 1.0)
    fetch_pixels = int(math.ceil(float(pixels) * fov_padded / requested_max_axis))
    fetch_pixels = max(int(pixels), min(fetch_pixels, FINDER_FETCH_MAX_PIXELS))

    mode = str(mode or "DSS").strip()
    if mode == "SkyView":
        mode = "DSS"

    if mode in {"DSS", "DSS2 Red", "DSS2 Blue"}:
        used, hdu = _get_skyview_hdu(coord, fov_padded, fetch_pixels, [mode])
        if hdu is None:
            return None, None, ""
        data = np.asarray(hdu.data, dtype=float)
        wcs  = WCS(hdu.header).celestial
        return data, wcs, used

    if mode == "Pan-STARRS":
        hdu = _get_panstarrs_hdu(coord, fov_padded, fetch_pixels, filt=PANSTARRS_FILTER)
        if hdu is not None:
            data = np.asarray(hdu.data, dtype=float)
            if data.ndim > 2:
                data = np.squeeze(data)
            wcs = WCS(hdu.header).celestial
            cov_w, cov_h = _wcs_coverage_arcmin(data, wcs)
            if min(cov_w, cov_h) >= 0.98 * fov_padded:
                return data, wcs, "Pan-STARRS"

        used, fallback_hdu = _get_skyview_hdu(
            coord, fov_padded, fetch_pixels, ["DSS2 Red", "DSS"]
        )
        if fallback_hdu is None:
            return None, None, ""
        data = np.asarray(fallback_hdu.data, dtype=float)
        wcs = WCS(fallback_hdu.header).celestial
        return data, wcs, f"{used} (Pan-STARRS fallback)"

    return None, None, ""

def format_finder_cursor(
    wcs: WCS | None,
    x: float,
    y: float,
    include_pixel: bool = True,
) -> str:
    try:
        xf, yf = float(x), float(y)
    except Exception:
        return ""
    if not np.isfinite(xf) or not np.isfinite(yf):
        return ""

    if wcs is not None:
        try:
            world = wcs.pixel_to_world(xf, yf)
            if isinstance(world, SkyCoord):
                c = world.icrs
                if np.isfinite(c.ra.deg) and np.isfinite(c.dec.deg):
                    ra_s = c.ra.to_string(
                        unit=u.hourangle, sep=":", precision=3, pad=True,
                    )
                    dec_s = c.dec.to_string(
                        unit=u.deg, sep=":", precision=2, pad=True,
                        alwayssign=True,
                    )
                    text = f"RA {ra_s}    Dec {dec_s}"
                    if include_pixel:
                        text += f"    Pixel x={xf:.2f}, y={yf:.2f}"
                    return text
        except Exception:
            pass

    return f"Pixel x={xf:.2f}, y={yf:.2f}" if include_pixel else ""

def render_finder_figure_from_data(
    coord:        SkyCoord,
    name:         str,
    data:         np.ndarray,
    input_wcs:    WCS,
    fov_w_arcmin: int,
    survey_label: str,
    roll_deg:     float      = 0.0,
    fov_h_arcmin: int | None = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> plt.Figure:
    fov_w = _clamp_fov(fov_w_arcmin)
    fov_h = _clamp_fov(fov_h_arcmin if fov_h_arcmin is not None else fov_w)
    target = coord.icrs

    try:
        scale_deg_arr = np.abs(proj_plane_pixel_scales(input_wcs.celestial))
        scale_deg     = float(np.mean(scale_deg_arr))
    except Exception:
        try:
            scale_deg = float(np.mean(np.abs(input_wcs.wcs.cdelt[:2])))
        except Exception:
            scale_deg = 1.0 / 3600.0 

    scale_arcmin = scale_deg * 60.0
    MAX_PX       = 1200   

    nx_out = min(MAX_PX, max(4, int(round(fov_w / scale_arcmin))))
    ny_out = min(MAX_PX, max(4, int(round(fov_h / scale_arcmin))))
    shape_out = (ny_out, nx_out)

    out_wcs = _build_output_wcs(input_wcs, target, shape_out, roll_deg=float(roll_deg))

    arr    = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    fv     = float(np.nanmedian(finite)) if finite.size else 0.0
    data_plot = _reproject_to_output_wcs(arr, input_wcs, out_wcs, shape_out, fill_value=fv)
    wcs_plot  = out_wcs

    plot_arr    = np.asarray(data_plot, dtype=float)
    finite_out  = plot_arr[np.isfinite(plot_arr)]
    vmin, vmax  = ZScaleInterval().get_limits(finite_out) if finite_out.size else (0.0, 1.0)
    plot_masked = np.ma.masked_invalid(plot_arr)

    aspect  = nx_out / max(1, ny_out)   
    BASE_H  = 7.0   
    if aspect >= 1.0:
        fig_w = min(12.0, BASE_H * aspect)
        fig_h = BASE_H
    else:
        fig_w = max(3.5, BASE_H * aspect)
        fig_h = BASE_H
    fig_w = max(4.0, min(fig_w, 12.0))
    fig_h = max(3.0, min(fig_h, 10.0))

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")
    ax  = fig.add_subplot(111, projection=wcs_plot)
    ax.set_facecolor("black")
    ax.imshow(plot_masked, origin="lower", vmin=vmin, vmax=vmax,
              cmap="gray", interpolation="nearest")
    if flip_horizontal:
        ax.invert_xaxis()
    if flip_vertical:
        ax.invert_yaxis()

    try:
        for axis_idx, pos in ((0, "b"), (1, "l")):
            ax.coords[axis_idx].set_ticks_position(pos)
            ax.coords[axis_idx].set_ticklabel_position(pos)
            ax.coords[axis_idx].set_axislabel_position(pos)
    except Exception:
        pass

    ax.coords.grid(color="white", alpha=0.20, linestyle="--")
    ax.coords[0].set_axislabel("RA",  color="white")
    ax.coords[1].set_axislabel("Dec", color="white")
    ax.coords[0].set_ticklabel(color="white")
    ax.coords[1].set_ticklabel(color="white")

    fov_str = f"FOV={fov_w}×{fov_h}′" if fov_w != fov_h else f"FOV={fov_w}′"
    title   = f"{name} — {survey_label} — {fov_str}"
    if abs(float(roll_deg)) > 1e-9:
        title += f" — Roll={float(roll_deg):.1f}°"
    flips = []
    if flip_horizontal:
        flips.append("H")
    if flip_vertical:
        flips.append("V")
    if flips:
        title += " — Flip " + "+".join(flips)
    fig.suptitle(title, color="white", fontsize=13, y=0.985)

    ax._rho_roll_deg     = 0.0
    ax._rho_data_shape   = np.asarray(data_plot).shape
    ax._rho_wcs          = wcs_plot
    ax._rho_display_roll = float(roll_deg)
    ax._rho_flip_horizontal = bool(flip_horizontal)
    ax._rho_flip_vertical = bool(flip_vertical)

    ax.format_coord = lambda x, y: format_finder_cursor(wcs_plot, x, y, include_pixel=True)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08)
    plt.close(fig)
    return fig

def _empty_finder_figure(
    name:         str,
    fov_w_arcmin: int,
    fov_h_arcmin: int | None = None,
    reason:       str        = "unavailable",
) -> plt.Figure:
    fov_h   = fov_h_arcmin if fov_h_arcmin is not None else fov_w_arcmin
    fov_str = f"FOV={fov_w_arcmin}×{fov_h}′" if fov_w_arcmin != fov_h else f"FOV={fov_w_arcmin}′"
    fig = plt.figure(figsize=(7.0, 7.0))
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.text(0.5, 0.5, f"Finder chart {reason}\n{fov_str}",
            ha="center", va="center", color="white", fontsize=12)
    ax.set_title(str(name), color="white", fontsize=13, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    ax._rho_roll_deg     = 0.0
    ax._rho_data_shape   = (1, 1)
    ax._rho_wcs          = None
    ax._rho_display_roll = 0.0
    plt.close(fig)
    return fig

def finder_figure(
    coord:        SkyCoord,
    name:         str,
    fov_w_arcmin: int,
    mode:         str,
    roll_deg:     float      = 0.0,
    fov_h_arcmin: int | None = None,
) -> plt.Figure:
    data, wcs, label = fetch_finder_raw(coord, fov_w_arcmin, mode, fov_h_arcmin=fov_h_arcmin)
    if data is None or wcs is None:
        return _empty_finder_figure(name, fov_w_arcmin, fov_h_arcmin)
    return render_finder_figure_from_data(coord, name, data, wcs, fov_w_arcmin, label,
                                          roll_deg, fov_h_arcmin=fov_h_arcmin)

def _catalog_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.ma.is_masked(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            value = str(value)
    s = re.sub(r"\s+", " ", str(value)).strip()
    if s.lower() in ("", "none", "nan", "--", "masked"):
        return ""

    if s.upper().startswith("NAME "):
        s = s[5:].strip()
    return s

def _catalog_float(value: Any) -> float | None:
    try:
        if np.ma.is_masked(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None

def _row_first(row: Any, names: tuple[str, ...]) -> Any:
    try:
        cols = list(row.colnames)
    except Exception:
        try:
            cols = list(row.keys())
        except Exception:
            cols = []
    lower = {str(c).lower(): c for c in cols}
    for name in names:
        key = lower.get(str(name).lower())
        if key is not None:
            try:
                return row[key]
            except Exception:
                continue
    return None

def _identifier_priority(identifier: str, catalog: str) -> int:
    ident = _catalog_text(identifier)
    up = ident.upper()
    if not ident:
        return 99
    if any(up.startswith(prefix) for prefix in (
        "HD ", "HIP ", "HR ", "BD", "CD", "CPD", "NGC ", "IC ",
        "M ", "MESSIER ", "SAO ", "WDS ", "GJ ", "GL ", "LHS ",
        "LTT ", "* ", "V* ", "SN ", "PN ", "NAME ",
    )):
        return 0
    if catalog == "SIMBAD" and not up.startswith(("GAIA ", "TYC ")):
        return 1
    if up.startswith("TYC ") or catalog == "Tycho-2":
        return 2
    if up.startswith("GAIA ") or catalog == "Gaia DR3":
        return 4
    return 3

def _make_candidate(
    click_coord: SkyCoord,
    obj_coord: SkyCoord,
    identifier: str,
    catalog: str,
    magnitude: float | None = None,
    mag_band: str | None = None,
) -> Dict[str, Any] | None:
    try:
        sep = float(click_coord.separation(obj_coord).arcsec)
    except Exception:
        return None
    if not np.isfinite(sep):
        return None
    identifier = _catalog_text(identifier)
    if not identifier:
        return None
    return {
        "coord": obj_coord.icrs,
        "sep_arcsec": sep,
        "main_id": identifier,
        "catalog": str(catalog),
        "mag": _catalog_float(magnitude),
        "mag_band": _catalog_text(mag_band) or None,
        "aliases": [identifier],
        "catalogs": [str(catalog)],
        "id_priority": _identifier_priority(identifier, str(catalog)),
        "astrometry_priority": {"Gaia DR3": 0, "Tycho-2": 1, "SIMBAD": 2}.get(str(catalog), 9),
    }

def _merge_catalog_candidates(
    click_coord: SkyCoord,
    candidates: List[Dict[str, Any]],
    merge_radius_arcsec: float = 3.0,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: float(c.get("sep_arcsec", np.inf))):
        match = None
        for group in merged:
            try:
                if cand["coord"].separation(group["coord"]).arcsec <= merge_radius_arcsec:
                    match = group
                    break
            except Exception:
                continue
        if match is None:
            merged.append(dict(cand))
            continue

        for alias in cand.get("aliases", []):
            if alias and alias not in match["aliases"]:
                match["aliases"].append(alias)
        for cat in cand.get("catalogs", []):
            if cat and cat not in match["catalogs"]:
                match["catalogs"].append(cat)

        if int(cand.get("id_priority", 99)) < int(match.get("id_priority", 99)):
            match["main_id"] = cand["main_id"]
            match["catalog"] = cand["catalog"]
            match["id_priority"] = cand["id_priority"]

        cmag = cand.get("mag")
        mmag = match.get("mag")
        if cmag is not None and (mmag is None or float(cmag) < float(mmag)):
            match["mag"] = float(cmag)
            match["mag_band"] = cand.get("mag_band")

        if int(cand.get("astrometry_priority", 9)) < int(match.get("astrometry_priority", 9)):
            match["coord"] = cand["coord"]
            match["astrometry_priority"] = cand["astrometry_priority"]

        match["sep_arcsec"] = float(click_coord.separation(match["coord"]).arcsec)

    return merged

def identify_star_at_coord(
    coord: SkyCoord,
    radius_arcsec: float = 30.0,
) -> Dict[str, Any]:
    click = coord.icrs
    radius_arcsec = max(1.0, float(radius_arcsec))
    rad = radius_arcsec * u.arcsec
    candidates: List[Dict[str, Any]] = []

    try:
        tables = _get_vizier_client().query_region(click, radius=rad, catalog="I/355/gaiadr3")
        if tables and len(tables) > 0:
            tbl = tables[0]
            for row in tbl:
                ra = _catalog_float(_row_first(row, ("RA_ICRS", "_RAJ2000", "RAJ2000")))
                dec = _catalog_float(_row_first(row, ("DE_ICRS", "_DEJ2000", "DEJ2000")))
                source = _catalog_text(_row_first(row, ("Source", "source_id")))
                if ra is None or dec is None or not source:
                    continue
                obj = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                cand = _make_candidate(
                    click, obj, f"Gaia DR3 {source}", "Gaia DR3",
                    _row_first(row, ("Gmag", "phot_g_mean_mag")), "G",
                )
                if cand is not None:
                    candidates.append(cand)
    except Exception:
        pass

    try:
        tables = _get_vizier_client().query_region(click, radius=rad, catalog="I/259/tyc2")
        if tables and len(tables) > 0:
            tbl = tables[0]
            for row in tbl:
                ra = _catalog_float(_row_first(row, ("RAmdeg", "_RAJ2000", "RAJ2000")))
                dec = _catalog_float(_row_first(row, ("DEmdeg", "_DEJ2000", "DEJ2000")))
                if ra is None or dec is None:
                    continue
                try:
                    tyc1 = int(float(_row_first(row, ("TYC1",))))
                    tyc2 = int(float(_row_first(row, ("TYC2",))))
                    tyc3 = int(float(_row_first(row, ("TYC3",))))
                    identifier = f"TYC {tyc1}-{tyc2}-{tyc3}"
                except Exception:
                    identifier = _catalog_text(_row_first(row, ("TYC", "Name")))
                if not identifier:
                    continue
                obj = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                cand = _make_candidate(
                    click, obj, identifier, "Tycho-2",
                    _row_first(row, ("VTmag", "Vmag")), "VT",
                )
                if cand is not None:
                    candidates.append(cand)
    except Exception:
        pass

    try:
        simbad_tap = Simbad()
        ra_deg = float(click.ra.deg)
        dec_deg = float(click.dec.deg)
        rad_deg = radius_arcsec / 3600.0
        adql = f"""
            SELECT TOP 100 main_id, ra, dec, flux_v,
                DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_deg}, {dec_deg})) AS dist_deg
            FROM basic
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra_deg}, {dec_deg}, {rad_deg})
            )
            ORDER BY dist_deg ASC
        """
        table = simbad_tap.query_tap(adql)
        if table is not None:
            for row in table:
                ra = _catalog_float(_row_first(row, ("ra", "RA")))
                dec = _catalog_float(_row_first(row, ("dec", "DEC")))
                identifier = _catalog_text(_row_first(row, ("main_id", "MAIN_ID")))
                if ra is None or dec is None or not identifier:
                    continue
                obj = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                cand = _make_candidate(
                    click, obj, identifier, "SIMBAD",
                    _row_first(row, ("flux_v", "FLUX_V", "V")), "V",
                )
                if cand is not None:
                    candidates.append(cand)
    except Exception:
        try:
            simbad_tap = Simbad()
            ra_deg = float(click.ra.deg)
            dec_deg = float(click.dec.deg)
            rad_deg = radius_arcsec / 3600.0
            adql = f"""
                SELECT TOP 100 main_id, ra, dec,
                    DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_deg}, {dec_deg})) AS dist_deg
                FROM basic
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {ra_deg}, {dec_deg}, {rad_deg})
                )
                ORDER BY dist_deg ASC
            """
            table = simbad_tap.query_tap(adql)
            if table is not None:
                for row in table:
                    ra = _catalog_float(_row_first(row, ("ra", "RA")))
                    dec = _catalog_float(_row_first(row, ("dec", "DEC")))
                    identifier = _catalog_text(_row_first(row, ("main_id", "MAIN_ID")))
                    if ra is None or dec is None or not identifier:
                        continue
                    obj = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                    cand = _make_candidate(click, obj, identifier, "SIMBAD", None, None)
                    if cand is not None:
                        candidates.append(cand)
        except Exception:
            pass

    candidates = [c for c in candidates if float(c.get("sep_arcsec", np.inf)) <= radius_arcsec]
    merged = _merge_catalog_candidates(click, candidates, merge_radius_arcsec=3.0)
    if not merged:
        return {"ok": False, "msg": f"No catalogue object within {radius_arcsec:.0f}\""}

    click_aperture = min(radius_arcsec, max(12.0, 0.67 * radius_arcsec))
    close = [c for c in merged if float(c["sep_arcsec"]) <= click_aperture]
    pool = close if close else merged
    selection_radius = click_aperture if close else radius_arcsec

    def _brightness_key(c: Dict[str, Any]):
        mag = c.get("mag")
        return (
            1 if mag is None else 0,
            float(mag) if mag is not None else np.inf,
            float(c.get("sep_arcsec", np.inf)),
            int(c.get("id_priority", 99)),
        )

    best = min(pool, key=_brightness_key)
    mag = best.get("mag")
    band = best.get("mag_band")
    return {
        "ok": True,
        "main_id": best.get("main_id", "Unknown"),
        "catalog": best.get("catalog", ""),
        "mag": float(mag) if mag is not None else None,
        "mag_band": band,
        "vmag": float(mag) if mag is not None else None,
        "sep_arcsec": float(best["sep_arcsec"]),
        "coord": best["coord"],
        "aliases": list(best.get("aliases", [])),
        "catalogs": list(best.get("catalogs", [])),
        "candidate_count": len(merged),
        "selection_radius_arcsec": float(selection_radius),
        "selection_mode": "brightest in click aperture" if close else "brightest in search radius",
    }

_PLANCK_H = 6.62607015e-34       
_LIGHT_C  = 299792458.0          
_BOLTZ_K  = 1.380649e-23         
_AB_ZERO_W_M2_HZ = 3631.0e-26    

@dataclass(frozen=True)
class ExposureFilterSpec:
    name: str
    central_nm: float
    width_nm: float
    relative_throughput: float
    sky_ab_mag_arcsec2: float
    extinction_mag_airmass: float
    line_key: str | None = None

EXPOSURE_FILTERS: Dict[str, ExposureFilterSpec] = {
    "g":       ExposureFilterSpec("g",       477.0, 137.0, 0.86, 21.8, 0.20),
    "r":       ExposureFilterSpec("r",       623.0, 138.0, 0.90, 21.0, 0.12),
    "i":       ExposureFilterSpec("i",       763.0, 153.0, 0.86, 20.0, 0.08),
    "z":       ExposureFilterSpec("z",       913.0,  95.0, 0.72, 18.7, 0.07),
    "H-alpha": ExposureFilterSpec("H-alpha", 656.3,   7.0, 0.78, 20.8, 0.10, "H-alpha"),
    "H-beta":  ExposureFilterSpec("H-beta",  486.1,   7.0, 0.72, 21.7, 0.18, "H-beta"),
    "OIII":    ExposureFilterSpec("OIII",    500.7,   7.0, 0.74, 21.5, 0.16, "OIII"),
    "SII":     ExposureFilterSpec("SII",     672.4,   7.0, 0.74, 20.6, 0.09, "SII"),
}

@dataclass(frozen=True)
class ReferenceMagnitudeBandSpec:
    key: str
    display_name: str
    family: str
    central_nm: float
    zero_point_jy: float
    magnitude_system: str

REFERENCE_MAGNITUDE_BANDS: Dict[str, ReferenceMagnitudeBandSpec] = {
    "Johnson U": ReferenceMagnitudeBandSpec(
        "Johnson U", "Johnson U (Vega)", "Johnson / Cousins", 360.0, 1755.0, "Vega"),
    "Johnson B": ReferenceMagnitudeBandSpec(
        "Johnson B", "Johnson B (Vega)", "Johnson / Cousins", 440.0, 4000.87, "Vega"),
    "Johnson V": ReferenceMagnitudeBandSpec(
        "Johnson V", "Johnson V (Vega)", "Johnson / Cousins", 550.0, 3597.28, "Vega"),
    "Johnson R": ReferenceMagnitudeBandSpec(
        "Johnson R", "Johnson R (Vega)", "Johnson / Cousins", 700.0, 3080.0, "Vega"),
    "Johnson I": ReferenceMagnitudeBandSpec(
        "Johnson I", "Johnson I (Vega)", "Johnson / Cousins", 900.0, 2550.0, "Vega"),
    "Cousins Rc": ReferenceMagnitudeBandSpec(
        "Cousins Rc", "Cousins R_C (Vega)", "Johnson / Cousins", 710.0, 3080.0, "Vega"),
    "Cousins Ic": ReferenceMagnitudeBandSpec(
        "Cousins Ic", "Cousins I_C (Vega)", "Johnson / Cousins", 790.0, 2432.84, "Vega"),

    "Gaia G": ReferenceMagnitudeBandSpec(
        "Gaia G", "Gaia DR3 G (Vega)", "Gaia DR3", 622.0, 3229.0, "Vega"),
    "Gaia BP": ReferenceMagnitudeBandSpec(
        "Gaia BP", "Gaia DR3 BP (Vega)", "Gaia DR3", 511.0, 3552.0, "Vega"),
    "Gaia RP": ReferenceMagnitudeBandSpec(
        "Gaia RP", "Gaia DR3 RP (Vega)", "Gaia DR3", 777.0, 2555.0, "Vega"),

    "Sloan u": ReferenceMagnitudeBandSpec(
        "Sloan u", "Sloan u (AB)", "Sloan", 355.0, 3631.0, "AB"),
    "Sloan g": ReferenceMagnitudeBandSpec(
        "Sloan g", "Sloan g (AB)", "Sloan", 477.0, 3631.0, "AB"),
    "Sloan r": ReferenceMagnitudeBandSpec(
        "Sloan r", "Sloan r (AB)", "Sloan", 623.0, 3631.0, "AB"),
    "Sloan i": ReferenceMagnitudeBandSpec(
        "Sloan i", "Sloan i (AB)", "Sloan", 763.0, 3631.0, "AB"),
    "Sloan z": ReferenceMagnitudeBandSpec(
        "Sloan z", "Sloan z (AB)", "Sloan", 913.0, 3631.0, "AB"),

    "Pan-STARRS g": ReferenceMagnitudeBandSpec(
        "Pan-STARRS g", "Pan-STARRS g (AB)", "Pan-STARRS", 481.0, 3631.0, "AB"),
    "Pan-STARRS r": ReferenceMagnitudeBandSpec(
        "Pan-STARRS r", "Pan-STARRS r (AB)", "Pan-STARRS", 617.0, 3631.0, "AB"),
    "Pan-STARRS i": ReferenceMagnitudeBandSpec(
        "Pan-STARRS i", "Pan-STARRS i (AB)", "Pan-STARRS", 752.0, 3631.0, "AB"),
    "Pan-STARRS z": ReferenceMagnitudeBandSpec(
        "Pan-STARRS z", "Pan-STARRS z (AB)", "Pan-STARRS", 866.0, 3631.0, "AB"),
    "Pan-STARRS y": ReferenceMagnitudeBandSpec(
        "Pan-STARRS y", "Pan-STARRS y (AB)", "Pan-STARRS", 962.0, 3631.0, "AB"),

    "2MASS J": ReferenceMagnitudeBandSpec(
        "2MASS J", "2MASS J (Vega)", "2MASS", 1235.0, 1594.0, "Vega"),
    "2MASS H": ReferenceMagnitudeBandSpec(
        "2MASS H", "2MASS H (Vega)", "2MASS", 1662.0, 1024.0, "Vega"),
    "2MASS Ks": ReferenceMagnitudeBandSpec(
        "2MASS Ks", "2MASS K_s (Vega)", "2MASS", 2159.0, 666.7, "Vega"),

    "H-alpha": ReferenceMagnitudeBandSpec(
        "H-alpha", "H-alpha (AB)", "Narrowband", 656.3, 3631.0, "AB"),
    "H-beta": ReferenceMagnitudeBandSpec(
        "H-beta", "H-beta (AB)", "Narrowband", 486.1, 3631.0, "AB"),
    "OIII": ReferenceMagnitudeBandSpec(
        "OIII", "OIII (AB)", "Narrowband", 500.7, 3631.0, "AB"),
    "SII": ReferenceMagnitudeBandSpec(
        "SII", "SII (AB)", "Narrowband", 672.4, 3631.0, "AB"),
}

_REFERENCE_BAND_ALIASES_EXACT: Dict[str, str] = {
    "U": "Johnson U", "B": "Johnson B", "V": "Johnson V",
    "R": "Johnson R", "I": "Johnson I",
    "Rc": "Cousins Rc", "Ic": "Cousins Ic",
    "G": "Gaia G", "BP": "Gaia BP", "RP": "Gaia RP",
    "u": "Sloan u", "g": "Sloan g", "r": "Sloan r",
    "i": "Sloan i", "z": "Sloan z",
    "J": "2MASS J", "H": "2MASS H", "Ks": "2MASS Ks",
}

_REFERENCE_BAND_ALIASES_NORMALIZED: Dict[str, str] = {
    "johnson u": "Johnson U", "uj": "Johnson U",
    "johnson b": "Johnson B", "bj": "Johnson B",
    "johnson v": "Johnson V", "vj": "Johnson V",
    "johnson r": "Johnson R", "rj": "Johnson R",
    "johnson i": "Johnson I", "ij": "Johnson I",
    "cousins r": "Cousins Rc", "cousins rc": "Cousins Rc", "rc": "Cousins Rc",
    "cousins i": "Cousins Ic", "cousins ic": "Cousins Ic", "ic": "Cousins Ic",
    "gaia g": "Gaia G", "gaia dr3 g": "Gaia G", "gmag": "Gaia G",
    "gaia bp": "Gaia BP", "gaia dr3 bp": "Gaia BP", "gbp": "Gaia BP", "bpmag": "Gaia BP",
    "gaia rp": "Gaia RP", "gaia dr3 rp": "Gaia RP", "grp": "Gaia RP", "rpmag": "Gaia RP",
    "sloan u": "Sloan u", "sdss u": "Sloan u", "uprime": "Sloan u",
    "sloan g": "Sloan g", "sdss g": "Sloan g", "gprime": "Sloan g",
    "sloan r": "Sloan r", "sdss r": "Sloan r", "rprime": "Sloan r",
    "sloan i": "Sloan i", "sdss i": "Sloan i", "iprime": "Sloan i",
    "sloan z": "Sloan z", "sdss z": "Sloan z", "zprime": "Sloan z",
    "pan starrs g": "Pan-STARRS g", "ps1 g": "Pan-STARRS g", "ps g": "Pan-STARRS g",
    "pan starrs r": "Pan-STARRS r", "ps1 r": "Pan-STARRS r", "ps r": "Pan-STARRS r",
    "pan starrs i": "Pan-STARRS i", "ps1 i": "Pan-STARRS i", "ps i": "Pan-STARRS i",
    "pan starrs z": "Pan-STARRS z", "ps1 z": "Pan-STARRS z", "ps z": "Pan-STARRS z",
    "pan starrs y": "Pan-STARRS y", "ps1 y": "Pan-STARRS y", "ps y": "Pan-STARRS y",
    "2mass j": "2MASS J", "jmag": "2MASS J",
    "2mass h": "2MASS H", "hmag": "2MASS H",
    "2mass ks": "2MASS Ks", "2mass k": "2MASS Ks", "ksmag": "2MASS Ks",
    "apass b": "Johnson B", "apass v": "Johnson V",
    "apass g": "Sloan g", "apass r": "Sloan r", "apass i": "Sloan i",
    "halpha": "H-alpha", "h alpha": "H-alpha", "h-alpha": "H-alpha",
    "hbeta": "H-beta", "h beta": "H-beta", "h-beta": "H-beta",
    "oiii": "OIII", "o iii": "OIII", "sii": "SII", "s ii": "SII",
}

def _normalize_reference_band_name(value: str) -> str:
    text = str(value or "").strip()
    if text in REFERENCE_MAGNITUDE_BANDS:
        return text
    if text in _REFERENCE_BAND_ALIASES_EXACT:
        return _REFERENCE_BAND_ALIASES_EXACT[text]

    norm = text.lower().replace("_", " ").replace("'", "prime")
    norm = norm.replace("(", " ").replace(")", " ")
    norm = re.sub(r"[^a-z0-9+\- ]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in _REFERENCE_BAND_ALIASES_NORMALIZED:
        return _REFERENCE_BAND_ALIASES_NORMALIZED[norm]
    raise ValueError(
        f"Unsupported reference magnitude band: {value!r}. "
        "Choose a Johnson/Cousins, Gaia DR3, Sloan, or listed narrowband band."
    )

def get_reference_magnitude_band(value: str) -> ReferenceMagnitudeBandSpec:
    """Return the canonical input-band definition for *value*."""
    return REFERENCE_MAGNITUDE_BANDS[_normalize_reference_band_name(value)]

def reference_magnitude_band_groups() -> List[Tuple[str, List[ReferenceMagnitudeBandSpec]]]:
    """Ordered groups used by the exposure-calculator combo box."""
    groups: List[Tuple[str, List[ReferenceMagnitudeBandSpec]]] = []
    for family in ("Johnson / Cousins", "Gaia DR3", "Sloan", "Pan-STARRS", "2MASS", "Narrowband"):
        specs = [s for s in REFERENCE_MAGNITUDE_BANDS.values() if s.family == family]
        if specs:
            groups.append((family, specs))
    return groups

REFERENCE_WAVELENGTHS_NM: Dict[str, float] = {
    key: spec.central_nm for key, spec in REFERENCE_MAGNITUDE_BANDS.items()
}
REFERENCE_WAVELENGTHS_NM.update({
    alias: REFERENCE_MAGNITUDE_BANDS[key].central_nm
    for alias, key in _REFERENCE_BAND_ALIASES_EXACT.items()
})


@dataclass
class ExposureCalculatorConfig:
    aperture_m: float = 0.356
    central_obstruction_fraction: float = 0.0
    base_system_throughput: float = 0.40
    pixel_scale_arcsec: float = 1.62
    read_noise_e: float = 9.3
    gain_e_per_adu: float = 0.37
    adc_max_adu: float = 65535.0
    binning_factor: int = 1
    dark_current_e_s_pix: float = 1.35
    full_well_e: float = 25500.0
    saturation_fraction: float = 0.80
    seeing_fwhm_arcsec: float = 10.3
    aperture_radius_fwhm: float = 1.5
    max_subexposure_s: float = 300.0
    max_narrowband_subexposure_s: float = 900.0
    minimum_practical_exposure_s: float = 0.10
    desired_peak_counts_adu: Optional[float] = 40000.0


@dataclass
class ExposureTarget:
    reference_mag_ab: float = 12.0
    reference_band: str = "Johnson V"
    spectrum_model: str = "blackbody"
    effective_temperature_k: float = 5800.0
    target_snr: float = 100.0
    airmass: float = 1.2
    line_fluxes_erg_s_cm2: Optional[Dict[str, float]] = None
    line_surface_fluxes_erg_s_cm2_arcsec2: Optional[Dict[str, float]] = None
    peak_line_factor: float = 1.0
    second_reference_mag: Optional[float] = None
    second_reference_band: Optional[str] = None
    source_type: str = "point"
    measurement_area_arcsec2: Optional[float] = None
    peak_surface_brightness_mag_arcsec2: Optional[float] = None


def _planck_bnu_at_lambda(lambda_nm: float, temperature_k: float) -> float:
    lam = float(lambda_nm) * 1e-9
    temp = max(float(temperature_k), 1.0)
    nu = _LIGHT_C / lam
    x = _PLANCK_H * nu / (_BOLTZ_K * temp)
    if x > 700.0:
        return 0.0
    return (2.0 * _PLANCK_H * nu**3 / _LIGHT_C**2) / np.expm1(x)


def reference_magnitude_fnu_w_m2_hz(
    magnitude: float,
    reference_band: str,
) -> float:
    spec = get_reference_magnitude_band(reference_band)
    fnu0 = float(spec.zero_point_jy) * 1.0e-26
    return float(fnu0 * 10.0 ** (-0.4 * float(magnitude)))

COLOR_MIN_WAVELENGTH_SEPARATION_RATIO = 1.03
COLOR_MAX_EXTRAPOLATION_RATIO = 1.80


def color_constraint_status(
    reference_band: str,
    second_reference_band: Optional[str],
    filter_name: str,
) -> tuple[bool, str]:
    """Return whether a two-band color is safe to use for the requested filter."""
    if not second_reference_band:
        return False, "no second band"

    ref_spec = get_reference_magnitude_band(reference_band)
    second_spec = get_reference_magnitude_band(second_reference_band)
    if ref_spec.key == second_spec.key:
        return False, "same photometric band"

    lam1 = float(ref_spec.central_nm)
    lam2 = float(second_spec.central_nm)
    lo, hi = min(lam1, lam2), max(lam1, lam2)
    if hi / max(lo, 1e-12) < COLOR_MIN_WAVELENGTH_SEPARATION_RATIO:
        return False, "bands are too close in wavelength"

    lamf = float(EXPOSURE_FILTERS[str(filter_name)].central_nm)
    if lo <= lamf <= hi:
        return True, "interpolation"

    nearest = lo if lamf < lo else hi
    ratio = max(lamf, nearest) / max(min(lamf, nearest), 1e-12)
    if ratio <= COLOR_MAX_EXTRAPOLATION_RATIO:
        return True, "moderate extrapolation"
    return False, "target filter is too far outside the color baseline"


def _color_powerlaw_alpha(
    reference_mag: float,
    reference_band: str,
    second_reference_mag: float,
    second_reference_band: str,
) -> Optional[float]:
    ref_spec = get_reference_magnitude_band(reference_band)
    second_spec = get_reference_magnitude_band(second_reference_band)
    if ref_spec.key == second_spec.key:
        return None

    fnu_ref = reference_magnitude_fnu_w_m2_hz(reference_mag, ref_spec.key)
    fnu_second = reference_magnitude_fnu_w_m2_hz(
        second_reference_mag, second_spec.key
    )
    nu_ref = _LIGHT_C / (float(ref_spec.central_nm) * 1e-9)
    nu_second = _LIGHT_C / (float(second_spec.central_nm) * 1e-9)
    denom = np.log(nu_second / nu_ref)
    if (
        fnu_ref <= 0.0
        or fnu_second <= 0.0
        or not np.isfinite(fnu_ref + fnu_second + denom)
        or abs(float(denom)) <= 1e-12
    ):
        return None
    alpha = np.log(fnu_second / fnu_ref) / denom
    return float(alpha) if np.isfinite(alpha) else None


def estimate_filter_ab_magnitude(
    reference_mag_ab: float,
    reference_band: str,
    filter_name: str,
    spectrum_model: str = "blackbody",
    effective_temperature_k: float = 5800.0,
    second_reference_mag: Optional[float] = None,
    second_reference_band: Optional[str] = None,
) -> float:
    """Estimate an AB magnitude in an RHO filter.

    When a second observed magnitude is supplied, the two calibrated flux
    densities define a local power-law color slope in f_nu.  This is preferable
    to guessing a stellar temperature or flat spectrum when catalog color
    information is available.  Without a second band, the selected blackbody
    or flat-f_nu fallback is used.
    """
    ref_spec = get_reference_magnitude_band(reference_band)
    fnu_ref = reference_magnitude_fnu_w_m2_hz(reference_mag_ab, ref_spec.key)
    filt_nm = EXPOSURE_FILTERS[str(filter_name)].central_nm

    fnu_filter = None
    if second_reference_mag is not None and second_reference_band:
        try:
            allowed, _reason = color_constraint_status(
                ref_spec.key, second_reference_band, str(filter_name)
            )
            alpha = (
                _color_powerlaw_alpha(
                    float(reference_mag_ab),
                    ref_spec.key,
                    float(second_reference_mag),
                    second_reference_band,
                )
                if allowed
                else None
            )
            if alpha is not None:
                nu_ref = _LIGHT_C / (float(ref_spec.central_nm) * 1e-9)
                nu_filter = _LIGHT_C / (float(filt_nm) * 1e-9)
                fnu_filter = fnu_ref * (nu_filter / nu_ref) ** alpha
        except Exception:
            fnu_filter = None

    if fnu_filter is None:
        model = str(spectrum_model or "flat_fnu").strip().lower()
        if model in {"flat", "flat_fnu", "flat fnu", "constant fnu"}:
            fnu_filter = fnu_ref
        else:
            b_ref = _planck_bnu_at_lambda(ref_spec.central_nm, effective_temperature_k)
            b_fil = _planck_bnu_at_lambda(filt_nm, effective_temperature_k)
            if b_ref <= 0.0 or b_fil <= 0.0 or not np.isfinite(b_ref + b_fil):
                fnu_filter = fnu_ref
            else:
                fnu_filter = fnu_ref * (b_fil / b_ref)

    if not np.isfinite(fnu_filter) or fnu_filter <= 0.0:
        return float("inf")
    return float(-2.5 * np.log10(fnu_filter / _AB_ZERO_W_M2_HZ))

def ab_magnitude_photon_flux_m2_s(mag_ab: float, central_nm: float, width_nm: float) -> float:
    m = float(mag_ab)
    fnu = _AB_ZERO_W_M2_HZ * 10.0 ** (-0.4 * m)
    lam0 = float(central_nm)
    half = max(float(width_nm), 1e-6) / 2.0
    lam_lo = max(0.1, lam0 - half) * 1e-9
    lam_hi = (lam0 + half) * 1e-9
    return float(fnu / _PLANCK_H * np.log(lam_hi / lam_lo))

def emission_line_photon_flux_m2_s(line_flux_erg_s_cm2: float, wavelength_nm: float) -> float:
    flux_w_m2 = max(0.0, float(line_flux_erg_s_cm2)) * 1e-3
    photon_energy_j = _PLANCK_H * _LIGHT_C / (float(wavelength_nm) * 1e-9)
    return flux_w_m2 / photon_energy_j if photon_energy_j > 0 else 0.0

def _solve_exposure_time_s(
    source_rate_e_s: float,
    background_rate_e_s: float,
    read_variance_e2: float,
    target_snr: float,
) -> float:
    s = float(source_rate_e_s)
    b = max(0.0, float(background_rate_e_s))
    r2 = max(0.0, float(read_variance_e2))
    q = max(float(target_snr), 1e-6) ** 2
    if not np.isfinite(s) or s <= 0.0:
        return float("inf")
    linear = q * (s + b)
    disc = linear**2 + 4.0 * s**2 * q * r2
    return float((linear + np.sqrt(max(0.0, disc))) / (2.0 * s**2))

def _required_exposures_for_snr(
    source_rate_e_s: float,
    background_rate_e_s: float,
    read_variance_e2: float,
    subexposure_s: float,
    target_snr: float,
) -> int:
    s = float(source_rate_e_s)
    b = max(0.0, float(background_rate_e_s))
    r2 = max(0.0, float(read_variance_e2))
    t = float(subexposure_s)
    q = max(float(target_snr), 1e-6) ** 2
    if not np.isfinite(s) or s <= 0.0 or not np.isfinite(t) or t <= 0.0:
        return 0
    signal_per = s * t
    variance_per = (s + b) * t + r2
    required = q * variance_per / max(signal_per**2, 1e-300)
    return max(1, int(np.ceil(required)))

def _stack_snr(
    source_rate_e_s: float,
    background_rate_e_s: float,
    read_variance_e2: float,
    subexposure_s: float,
    n_exposures: int,
) -> float:
    n = max(0, int(n_exposures))
    t = float(subexposure_s)
    s = float(source_rate_e_s)
    b = max(0.0, float(background_rate_e_s))
    r2 = max(0.0, float(read_variance_e2))
    if n <= 0 or t <= 0.0 or s <= 0.0:
        return 0.0
    signal = n * s * t
    variance = n * ((s + b) * t + r2)
    return float(signal / np.sqrt(variance)) if variance > 0.0 else float("inf")

def calculate_exposure_times(
    config: ExposureCalculatorConfig,
    target: ExposureTarget,
) -> List[Dict[str, Any]]:
    d = float(config.aperture_m)
    if d <= 0.0:
        raise ValueError("Telescope aperture must be positive.")
    obstruction = float(config.central_obstruction_fraction)
    if not 0.0 <= obstruction < 1.0:
        raise ValueError("Central obstruction must be between 0 and 1.")
    if config.pixel_scale_arcsec <= 0.0:
        raise ValueError("Pixel scale must be positive.")
    if config.seeing_fwhm_arcsec <= 0.0:
        raise ValueError("Seeing FWHM must be positive.")
    if target.target_snr <= 0.0:
        raise ValueError("Target S/N must be positive.")
    if config.gain_e_per_adu <= 0.0:
        raise ValueError("Camera gain must be positive.")
    if config.adc_max_adu <= 0.0:
        raise ValueError("Camera ADC maximum must be positive.")
    if int(config.binning_factor) < 1:
        raise ValueError("Binning factor must be at least 1.")
    if target.peak_line_factor <= 0.0:
        raise ValueError("Peak/mean line-brightness factor must be positive.")
    if (config.desired_peak_counts_adu is not None
            and float(config.desired_peak_counts_adu) <= 0.0):
        raise ValueError("Desired peak counts must be positive or disabled.")

    collecting_area_m2 = np.pi * (d / 2.0) ** 2 * (1.0 - obstruction**2)
    seeing = float(config.seeing_fwhm_arcsec)
    radius_arcsec = max(0.1, float(config.aperture_radius_fwhm) * seeing)
    sigma_arcsec = seeing / 2.354820045
    encircled = 1.0 - np.exp(-(radius_arcsec**2) / (2.0 * sigma_arcsec**2))

    source_type = str(target.source_type or "point").strip().lower()
    is_extended = source_type in {"extended", "diffuse", "nebula", "galaxy"}
    if is_extended:
        aperture_area_arcsec2 = max(
            1e-6,
            float(target.measurement_area_arcsec2 or 0.0),
        )
        if aperture_area_arcsec2 <= 1e-6:
            raise ValueError(
                "Extended/diffuse sources require a positive measurement area."
            )
    else:
        aperture_area_arcsec2 = np.pi * radius_arcsec**2

    n_pix = max(1.0, aperture_area_arcsec2 / float(config.pixel_scale_arcsec) ** 2)

    p = float(config.pixel_scale_arcsec)
    peak_fraction = float(math.erf(p / (2.0 * np.sqrt(2.0) * sigma_arcsec)) ** 2)
    peak_fraction = float(np.clip(peak_fraction, 1e-6, 1.0))

    line_fluxes = target.line_fluxes_erg_s_cm2 or {}
    line_surface_fluxes = target.line_surface_fluxes_erg_s_cm2_arcsec2 or {}
    results: List[Dict[str, Any]] = []

    for name, spec in EXPOSURE_FILTERS.items():
        color_used = False
        color_reason = ""
        if target.second_reference_mag is not None and target.second_reference_band:
            try:
                color_used, color_reason = color_constraint_status(
                    target.reference_band, target.second_reference_band, name
                )
            except Exception as exc:
                color_reason = str(exc)

        mag = estimate_filter_ab_magnitude(
            target.reference_mag_ab,
            target.reference_band,
            name,
            target.spectrum_model,
            target.effective_temperature_k,
            target.second_reference_mag,
            target.second_reference_band,
        )

        atmospheric_transmission = 10.0 ** (
            -0.4 * spec.extinction_mag_airmass * max(float(target.airmass), 0.0)
        )
        throughput = (
            float(config.base_system_throughput)
            * spec.relative_throughput
            * atmospheric_transmission
        )
        throughput = float(np.clip(throughput, 0.0, 1.0))

        continuum_photons = ab_magnitude_photon_flux_m2_s(
            mag, spec.central_nm, spec.width_nm
        )
        line_flux = float(line_fluxes.get(spec.line_key or "", 0.0) or 0.0)
        line_surface_flux = float(
            line_surface_fluxes.get(spec.line_key or "", 0.0) or 0.0
        )
        effective_line_flux = line_flux
        if is_extended and line_surface_flux > 0.0:
            effective_line_flux += line_surface_flux * aperture_area_arcsec2

        line_photons = (
            emission_line_photon_flux_m2_s(effective_line_flux, spec.central_nm)
            if spec.line_key else 0.0
        )

        continuum_source_e_s = continuum_photons * collecting_area_m2 * throughput
        line_source_e_s = line_photons * collecting_area_m2 * throughput

        if is_extended:
            # The input magnitude is interpreted as surface brightness.
            # Continuum is integrated across the measurement area; line inputs
            # have already been converted to an integrated aperture flux above.
            continuum_total_e_s = continuum_source_e_s * aperture_area_arcsec2
            total_source_e_s = continuum_total_e_s + line_source_e_s
            source_rate_e_s = total_source_e_s
        else:
            total_source_e_s = continuum_source_e_s + line_source_e_s
            source_rate_e_s = total_source_e_s * encircled

        sky_photons_arcsec2 = ab_magnitude_photon_flux_m2_s(
            spec.sky_ab_mag_arcsec2, spec.central_nm, spec.width_nm
        )
        sky_rate_e_s_arcsec2 = sky_photons_arcsec2 * collecting_area_m2 * throughput
        sky_rate_e_s = sky_rate_e_s_arcsec2 * aperture_area_arcsec2
        dark_rate_e_s = max(0.0, float(config.dark_current_e_s_pix)) * n_pix
        background_rate_e_s = sky_rate_e_s + dark_rate_e_s
        read_variance_e2 = n_pix * max(0.0, float(config.read_noise_e)) ** 2

        if is_extended:
            # S/N uses the mean surface brightness over the measurement area.
            # Saturation can instead use an independently supplied peak surface
            # brightness, which handles bright nuclei, knots, and filaments.
            if target.peak_surface_brightness_mag_arcsec2 is not None:
                peak_mag = estimate_filter_ab_magnitude(
                    float(target.peak_surface_brightness_mag_arcsec2),
                    target.reference_band,
                    name,
                    target.spectrum_model,
                    target.effective_temperature_k,
                    target.second_reference_mag,
                    target.second_reference_band,
                )
                peak_continuum_photons_arcsec2 = ab_magnitude_photon_flux_m2_s(
                    peak_mag, spec.central_nm, spec.width_nm
                )
                source_surface_rate_e_s_arcsec2 = (
                    peak_continuum_photons_arcsec2
                    * collecting_area_m2
                    * throughput
                )
            else:
                source_surface_rate_e_s_arcsec2 = continuum_source_e_s

            peak_line_rate_e_s_arcsec2 = 0.0
            if spec.line_key:
                peak_line_surface_flux = line_surface_flux
                if peak_line_surface_flux <= 0.0 and effective_line_flux > 0.0:
                    peak_line_surface_flux = effective_line_flux / aperture_area_arcsec2
                if peak_line_surface_flux > 0.0:
                    peak_line_photons_arcsec2 = emission_line_photon_flux_m2_s(
                        peak_line_surface_flux * float(target.peak_line_factor),
                        spec.central_nm,
                    )
                    peak_line_rate_e_s_arcsec2 = (
                        peak_line_photons_arcsec2
                        * collecting_area_m2
                        * throughput
                    )

            peak_rate_e_s = (
                (source_surface_rate_e_s_arcsec2 + peak_line_rate_e_s_arcsec2) * p**2
                + sky_rate_e_s_arcsec2 * p**2
                + max(0.0, float(config.dark_current_e_s_pix))
            )
        else:
            peak_rate_e_s = (
                total_source_e_s * peak_fraction
                + sky_rate_e_s_arcsec2 * p**2
                + max(0.0, float(config.dark_current_e_s_pix))
            )
        binning_charge_factor = float(max(1, int(config.binning_factor)) ** 2)
        usable_well_e = max(
            1.0,
            float(config.full_well_e)
            * binning_charge_factor
            * float(config.saturation_fraction),
        )
        usable_adc_e = max(
            1.0,
            float(config.adc_max_adu)
            * float(config.gain_e_per_adu)
            * float(config.saturation_fraction),
        )
        usable_signal_e = min(usable_well_e, usable_adc_e)
        saturation_s = (
            usable_signal_e / peak_rate_e_s
            if peak_rate_e_s > 0
            else float("inf")
        )
        is_narrowband = name in {"H-alpha", "H-beta", "OIII", "SII"}
        configured_max = (
            float(config.max_narrowband_subexposure_s)
            if is_narrowband else float(config.max_subexposure_s)
        )
        gain = float(config.gain_e_per_adu)
        desired_counts = (
            None if config.desired_peak_counts_adu is None
            else float(config.desired_peak_counts_adu)
        )
        count_target_s = (
            desired_counts * gain / peak_rate_e_s
            if desired_counts is not None and peak_rate_e_s > 0.0
            else float("inf")
        )

        if desired_counts is not None:
            suggested_sub_s = min(
                max(configured_max, 1e-6), saturation_s, count_target_s
            )
            suggested_n = _required_exposures_for_snr(
                source_rate_e_s, background_rate_e_s, read_variance_e2,
                suggested_sub_s, target.target_snr,
            )
            total_time_s = (
                suggested_n * suggested_sub_s
                if suggested_n > 0 and np.isfinite(suggested_sub_s)
                else float("inf")
            )
        else:
            max_sub = min(max(configured_max, 1e-6), saturation_s)
            suggested_n = 1
            total_time_s = float("inf")
            if max_sub > 0.0:
                for _ in range(20):
                    total_time_s = _solve_exposure_time_s(
                        source_rate_e_s, background_rate_e_s,
                        read_variance_e2 * suggested_n, target.target_snr,
                    )
                    if not np.isfinite(total_time_s):
                        suggested_n = 0
                        break
                    new_n = max(1, int(np.ceil(total_time_s / max_sub)))
                    if new_n == suggested_n:
                        break
                    suggested_n = new_n

            if suggested_n <= 0 or not np.isfinite(total_time_s):
                suggested_sub_s = float("inf")
            else:
                suggested_sub_s = total_time_s / suggested_n

        predicted_peak_counts_adu = (
            peak_rate_e_s * suggested_sub_s / gain
            if np.isfinite(suggested_sub_s) and gain > 0.0
            else float("inf")
        )
        achieved_snr = _stack_snr(
            source_rate_e_s, background_rate_e_s, read_variance_e2,
            suggested_sub_s, suggested_n,
        )

        notes: List[str] = []
        if is_extended:
            notes.append(
                f"surface brightness over {aperture_area_arcsec2:.1f} arcsec^2"
            )
            if target.peak_surface_brightness_mag_arcsec2 is not None:
                notes.append("peak surface brightness used for saturation")
            else:
                notes.append("mean surface brightness used for peak counts")
        if target.second_reference_mag is not None and target.second_reference_band:
            if color_used:
                notes.append(f"color-constrained SED ({color_reason})")
            else:
                notes.append(f"color not applied: {color_reason or 'invalid constraint'}")
        if is_extended and line_flux > 0.0 and float(target.peak_line_factor) > 1.0:
            notes.append(
                f"peak line brightness ×{float(target.peak_line_factor):g}"
            )
        if effective_line_flux > 0.0:
            if line_surface_flux > 0.0:
                notes.append("line surface flux included")
            else:
                notes.append("integrated line flux included")
        if desired_counts is not None:
            safe_peak_adu = usable_signal_e / gain
            if desired_counts > float(config.adc_max_adu):
                notes.append("desired counts exceed configured ADC maximum")
            if desired_counts > safe_peak_adu:
                notes.append("desired counts exceed safe-well target; saturation cap used")
            elif count_target_s > configured_max:
                notes.append("maximum subexposure prevents reaching desired counts")
            elif np.isfinite(count_target_s) and np.isfinite(suggested_sub_s):
                if suggested_sub_s >= 0.98 * count_target_s:
                    notes.append("subexposure set by peak-count target")
        if saturation_s < float(config.minimum_practical_exposure_s):
            notes.append("saturates below practical shutter time")
        elif np.isfinite(total_time_s) and saturation_s < total_time_s:
            notes.append("stack subexposures to avoid saturation")
        if np.isfinite(total_time_s) and total_time_s < float(config.minimum_practical_exposure_s):
            notes.append("defocus or use a neutral-density strategy")
        if name in {"H-alpha", "H-beta", "OIII", "SII"} and effective_line_flux <= 0.0:
            notes.append("continuum-only narrowband estimate")

        results.append({
            "filter": name,
            "central_nm": spec.central_nm,
            "width_nm": spec.width_nm,
            "estimated_ab_mag": float(mag),
            "effective_throughput": throughput,
            "source_rate_e_s": float(source_rate_e_s),
            "sky_rate_e_s": float(sky_rate_e_s),
            "n_pix": float(n_pix),
            "total_time_s": float(total_time_s),
            "saturation_time_s": float(saturation_s),
            "suggested_n": int(suggested_n),
            "suggested_subexposure_s": float(suggested_sub_s),
            "desired_peak_counts_adu": desired_counts,
            "count_target_time_s": float(count_target_s),
            "predicted_peak_counts_adu": float(predicted_peak_counts_adu),
            "achieved_snr": float(achieved_snr),
            "notes": "; ".join(notes),
        })

    return results

def package_self_test() -> None:
    """Offline packaged-feature smoke test used by CI release bundles."""
    import importlib

    # Dynamic imports that PyInstaller cannot infer from planner startup alone.
    _get_simbad()
    _get_vizier_client()
    _get_skyview_class()
    importlib.import_module("openpyxl")

    # Exercise point and diffuse ETC paths.
    point_results = calculate_exposure_times(
        ExposureCalculatorConfig(),
        ExposureTarget(reference_mag_ab=12.0, reference_band="Johnson V"),
    )
    if not point_results:
        raise RuntimeError("Point-source exposure self-test returned no results.")

    diffuse_results = calculate_exposure_times(
        ExposureCalculatorConfig(),
        ExposureTarget(
            reference_mag_ab=21.0,
            reference_band="Sloan r",
            source_type="extended",
            measurement_area_arcsec2=100.0,
            peak_surface_brightness_mag_arcsec2=18.5,
            second_reference_mag=21.5,
            second_reference_band="Sloan g",
        ),
    )
    if not diffuse_results:
        raise RuntimeError("Diffuse-source exposure self-test returned no results.")

    # Exercise rectangular finder reprojection without network access.
    data = np.arange(64 * 64, dtype=float).reshape(64, 64)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [32.5, 32.5]
    wcs.wcs.crval = [180.0, 20.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    coord = SkyCoord(180.0 * u.deg, 20.0 * u.deg, frame="icrs")
    fig = render_finder_figure_from_data(
        coord, "Self Test", data, wcs, 15, "local",
        roll_deg=17.0, fov_h_arcmin=20,
        flip_horizontal=True,
    )
    try:
        if not fig.axes:
            raise RuntimeError("Finder self-test did not create an axis.")
    finally:
        plt.close(fig)


def _norm_col(s: str) -> str:
    s = (s or "").replace("\n", " ").strip().lower()
    s = re.sub(r"[\*\(\)\[\]\{\}:,\-_/]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cand_norm = {_norm_col(c) for c in candidates}
    for col in df.columns:
        if _norm_col(str(col)) in cand_norm:
            return col
    raise KeyError(f"No matching column found among: {candidates}")

def _drop_template_first_data_row(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        code_col = next((c for c in df.columns if _norm_col(str(c)) == "code"), None)
        if code_col is not None:
            v = str(df.iloc[0][code_col]).strip()
            if "###" in v or v.upper() == "YYS###":
                return df.iloc[1:].reset_index(drop=True)
        checks = []
        for col_name, token in [
            ("Primary Identifier**", "string"),
            ("V Magnitude**",        "float"),
            ("Priority**",           "integer"),
            ("RA**",                 "hh mm ss"),
            ("Dec**",                "deg min sec"),
        ]:
            try:
                col = find_col(df, [col_name])
                checks.append(token in str(df.iloc[0][col]).lower())
            except Exception:
                pass
        if checks and any(checks):
            return df.iloc[1:].reset_index(drop=True)
    except Exception:
        pass
    return df

def load_targets_from_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        xls   = pd.ExcelFile(path)
        sheet = "TargetMasterSheet" if "TargetMasterSheet" in xls.sheet_names else xls.sheet_names[0]
        df    = pd.read_excel(path, sheet_name=sheet)

    df       = _drop_template_first_data_row(df)
    name_col = find_col(df, ["primary identifier", "Primary Identifier**", "object name", "name"])
    ra_col   = find_col(df, ["ra", "RA**", "radeg", "ra_deg"])
    dec_col  = find_col(df, ["dec", "Dec**", "dedeg", "decdeg", "dec_deg"])

    pr_col = None
    for cand in (["priority", "Priority**"], ["prio"], ["rank"]):
        try:
            pr_col = find_col(df, cand)
            break
        except Exception:
            pass

    def _pick_numeric_col(df, cands):
        cn = {_norm_col(c) for c in cands}
        matches = []
        for col in df.columns:
            if _norm_col(str(col)) not in cn:
                continue
            if pd.api.types.is_bool_dtype(df[col]):
                continue
            score = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            matches.append((score, col))
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1] if matches[0][0] > 0 else None

    vmag_col = _pick_numeric_col(df, ["v magnitude", "V Magnitude**", "vmag", "v_mag", "mag_v", "Vmag", "V"])

    return pd.DataFrame({
        "name":     df[name_col].astype(str),
        "ra":       df[ra_col].astype(str),
        "dec":      df[dec_col].astype(str),
        "priority": pd.to_numeric(df[pr_col], errors="coerce").fillna(3).astype(int) if pr_col else 3,
        "vmag":     pd.to_numeric(df[vmag_col], errors="coerce") if vmag_col else np.nan,
    })
