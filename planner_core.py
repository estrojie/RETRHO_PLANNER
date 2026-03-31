# planner_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo
import re
from astroquery.skyview import SkyView
import inspect

if "grid" not in inspect.signature(SkyView.get_images).parameters:
    _orig_get_images = SkyView.get_images

    def _get_images_no_grid(*args, **kwargs):
        kwargs.pop("grid", None)
        return _orig_get_images(*args, **kwargs)

    SkyView.get_images = _get_images_no_grid

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
from astroplan.plots import plot_finder_image
from astroquery.simbad import Simbad
from PIL import Image
from scipy.ndimage import affine_transform, map_coordinates

# Defaults (RHO)
DEFAULT_LAT = 29.400041
DEFAULT_LON = -82.585953
DEFAULT_HEIGHT_M = 31
DEFAULT_TZ = "US/Eastern"
DEFAULT_SITE_NAME = "RHO (Bronson, FL)"

DEFAULT_MIN_ALT_DEG = 26.0
DEFAULT_MAX_ALT_DEG = 62.0

DEFAULT_FOV1_ARCMIN = 90
DEFAULT_FOV2_ARCMIN = 20

SKYVIEW_SURVEYS = ["DSS2 Red", "DSS2 Blue", "DSS"]
PANSTARRS_FILTER = "r"
PANSTARRS_FILENAME_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PANSTARRS_FITSCUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

NWS_HEADERS = {
    "User-Agent": "RHOPlanner/1.0",
    "Accept": "application/geo+json,application/json",
}

# For notes / UI warnings
DEC_WARNING_LIMIT_DEG = 60.0


def _normalize_to_uint8(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _rotate_grayscale_image(data: np.ndarray, roll_deg: float, vmin: float, vmax: float) -> np.ndarray:
    img8 = _normalize_to_uint8(data, vmin, vmax)
    pil = Image.fromarray(img8, mode="L")
    rotated = pil.rotate(float(roll_deg), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    return np.asarray(rotated, dtype=np.uint8)


def _rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    return np.array([
        [float(np.cos(theta)), -float(np.sin(theta))],
        [float(np.sin(theta)),  float(np.cos(theta))],
    ], dtype=float)


def _rotate_data_with_affine(data: np.ndarray, roll_deg: float, fill_value: float | None = None) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D image array for finder chart rotation.")

    ny, nx = arr.shape
    center = np.array([(ny - 1.0) / 2.0, (nx - 1.0) / 2.0], dtype=float)

    rot_minus = _rotation_matrix_deg(-float(roll_deg))
    # affine_transform uses array-order coordinates: (row, col) == (y, x)
    mat_xy = rot_minus
    mat_rc = np.array([
        [mat_xy[1, 1], mat_xy[1, 0]],
        [mat_xy[0, 1], mat_xy[0, 0]],
    ], dtype=float)
    offset = center - mat_rc @ center

    if fill_value is None:
        finite = arr[np.isfinite(arr)]
        fill_value = float(np.nanmedian(finite)) if finite.size else 0.0

    rotated = affine_transform(
        arr,
        matrix=mat_rc,
        offset=offset,
        output_shape=arr.shape,
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=True,
    )
    return rotated


def _rotated_wcs(wcs: WCS, shape: tuple[int, int], roll_deg: float) -> WCS:
    new_wcs = wcs.deepcopy()
    ny, nx = int(shape[0]), int(shape[1])
    center = np.array([nx / 2.0, ny / 2.0], dtype=float)
    rot_minus = _rotation_matrix_deg(-float(roll_deg))

    try:
        if getattr(new_wcs.wcs, "has_cd", lambda: False)():
            cd = np.array(new_wcs.wcs.cd, dtype=float)
            new_wcs.wcs.cd = cd @ rot_minus
        else:
            pc = np.array(new_wcs.wcs.get_pc(), dtype=float)
            new_wcs.wcs.pc = pc @ rot_minus
    except Exception:
        pc = np.array(new_wcs.wcs.get_pc(), dtype=float)
        new_wcs.wcs.pc = pc @ rot_minus

    try:
        new_wcs.wcs.crpix = center
    except Exception:
        pass

    return new_wcs


def regenerate_rolled_finder_data(data: np.ndarray, wcs: WCS, roll_deg: float) -> tuple[np.ndarray, WCS]:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
    rolled = _rotate_data_with_affine(arr, roll_deg, fill_value=fill_value)
    return rolled, _rotated_wcs(wcs, arr.shape, roll_deg)


def _roll_pad_factor(roll_deg: float, dec_deg: float | None = None, extra_margin: float = 0.35) -> float:
    theta = np.deg2rad(float(roll_deg))
    base = abs(np.cos(theta)) + abs(np.sin(theta)) + float(extra_margin)
    pad = max(1.75, float(base))
    if dec_deg is not None and abs(float(dec_deg)) >= 75.0:
        pad = max(pad, 2.20)
    return pad


def _center_crop(data: np.ndarray, ny: int, nx: int) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(data)
    sy, sx = arr.shape[:2]
    ny = min(int(ny), int(sy))
    nx = min(int(nx), int(sx))
    y0 = max(0, (sy - ny) // 2)
    x0 = max(0, (sx - nx) // 2)
    return arr[y0:y0 + ny, x0:x0 + nx].copy(), (y0, x0)


def _center_crop_wcs(wcs: WCS, y0: int, x0: int) -> WCS:
    new_wcs = wcs.deepcopy()
    try:
        new_wcs.wcs.crpix = np.array(new_wcs.wcs.crpix, dtype=float) - np.array([float(x0), float(y0)], dtype=float)
    except Exception:
        pass
    return new_wcs


def regenerate_rolled_finder_data_with_padding(
    data: np.ndarray,
    wcs: WCS,
    roll_deg: float,
    final_shape: tuple[int, int],
) -> tuple[np.ndarray, WCS]:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
    rolled = _rotate_data_with_affine(arr, roll_deg, fill_value=fill_value)
    rolled_wcs = _rotated_wcs(wcs, arr.shape, roll_deg)
    cropped, (y0, x0) = _center_crop(rolled, int(final_shape[0]), int(final_shape[1]))
    cropped_wcs = _center_crop_wcs(rolled_wcs, y0, x0)
    return cropped, cropped_wcs


def _build_output_wcs_from_reference(ref_wcs: WCS, center_coord: SkyCoord, shape: tuple[int, int], roll_deg: float = 0.0) -> WCS:
    ny, nx = int(shape[0]), int(shape[1])
    out = WCS(naxis=2)

    try:
        ctype = list(ref_wcs.wcs.ctype)
    except Exception:
        ctype = ["RA---TAN", "DEC--TAN"]
    try:
        cunit = list(ref_wcs.wcs.cunit)
    except Exception:
        cunit = [u.deg, u.deg]

    try:
        scales_deg = np.abs(proj_plane_pixel_scales(ref_wcs.celestial)) * 180.0 / np.pi
    except Exception:
        if getattr(ref_wcs.wcs, 'cdelt', None) is not None and len(ref_wcs.wcs.cdelt) >= 2:
            scales_deg = np.abs(np.asarray(ref_wcs.wcs.cdelt[:2], dtype=float))
        else:
            scales_deg = np.array([1.0 / 3600.0, 1.0 / 3600.0], dtype=float)

    # Keep standard astronomical handedness: RA decreases to the right.
    cdelt = np.array([-float(scales_deg[0]), float(scales_deg[1])], dtype=float)

    theta = np.deg2rad(float(roll_deg))
    pc = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=float)

    out.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    out.wcs.crval = [float(center_coord.icrs.ra.deg), float(center_coord.icrs.dec.deg)]
    out.wcs.ctype = ctype
    out.wcs.cunit = cunit
    out.wcs.cdelt = cdelt
    out.wcs.pc = pc
    return out


def _reproject_to_output_wcs(data: np.ndarray, input_wcs: WCS, output_wcs: WCS, shape_out: tuple[int, int], fill_value: float = np.nan) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    ny, nx = int(shape_out[0]), int(shape_out[1])
    yy, xx = np.indices((ny, nx), dtype=float)

    world = output_wcs.pixel_to_world(xx + 0.5, yy + 0.5)
    xin, yin = input_wcs.world_to_pixel(world)
    xin = np.asarray(xin, dtype=float)
    yin = np.asarray(yin, dtype=float)

    # map_coordinates uses array order (row, col) -> (y, x) and zero-based pixels.
    sampled = map_coordinates(
        arr,
        [yin, xin],
        order=1,
        mode='constant',
        cval=float(fill_value) if np.isfinite(fill_value) else np.nan,
        prefilter=True,
    )

    invalid = (~np.isfinite(xin)) | (~np.isfinite(yin)) | (xin < -0.5) | (yin < -0.5) | (xin > (arr.shape[1] - 0.5)) | (yin > (arr.shape[0] - 0.5))
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
    out_wcs = _build_output_wcs_from_reference(input_wcs, center_coord, final_shape, roll_deg=roll_deg)
    reproj = _reproject_to_output_wcs(data, input_wcs, out_wcs, final_shape, fill_value=np.nan)
    return reproj, out_wcs


def _render_figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


def _rotate_rgb_image(rgb: np.ndarray, roll_deg: float) -> np.ndarray:
    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    rotated = pil.rotate(float(roll_deg), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return np.asarray(rotated, dtype=np.uint8)


def rotate_point_about_center(x: float, y: float, width: float, height: float, angle_deg: float) -> tuple[float, float]:
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    dx = float(x) - cx
    dy = float(y) - cy
    xr = dx * cos_t - dy * sin_t
    yr = dx * sin_t + dy * cos_t
    return xr + cx, yr + cy


def viewer_to_original_pixel(x: float, y: float, width: float, height: float, roll_deg: float) -> tuple[float, float]:
    return rotate_point_about_center(x, y, width, height, -float(roll_deg))

# Runtime state (editable from UI)
@dataclass
class SiteConfig:
    lat: float = DEFAULT_LAT
    lon: float = DEFAULT_LON
    height_m: float = DEFAULT_HEIGHT_M
    timezone: str = DEFAULT_TZ
    name: str = DEFAULT_SITE_NAME

_CURRENT_SITE = SiteConfig()
_CURRENT_OBSERVER: Observer | None = None
_PLANNING_DATE: date = date.today()

def set_site(lat: float, lon: float, height_m: float, timezone_str: str, name: str | None = None) -> None:
    global _CURRENT_SITE, _CURRENT_OBSERVER
    _CURRENT_SITE = SiteConfig(
        lat=float(lat),
        lon=float(lon),
        height_m=float(height_m),
        timezone=str(timezone_str),
        name=str(name) if name else f"Site ({lat:.4f}, {lon:.4f})",
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
        iers.conf.auto_download = True
        loc = EarthLocation(
            lat=_CURRENT_SITE.lat * u.deg,
            lon=_CURRENT_SITE.lon * u.deg,
            height=_CURRENT_SITE.height_m * u.m
        )
        _CURRENT_OBSERVER = Observer(location=loc, timezone=_CURRENT_SITE.timezone, name=_CURRENT_SITE.name)
    return _CURRENT_OBSERVER

def get_site_config() -> SiteConfig:
    return _CURRENT_SITE

# SIMBAD
custom_simbad = Simbad()
custom_simbad.add_votable_fields("flux(V)")

@dataclass
class ResolvedTarget:
    display_name: str
    coord: SkyCoord
    vmag: Any
    method: str

# weather.gov validTime parsing
def _parse_iso_duration_to_seconds(dur: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s

def _parse_valid_time(valid_time: str):
    try:
        start_str, dur_str = valid_time.split("/")
        start = pd.to_datetime(start_str, utc=True).to_pydatetime()
        seconds = _parse_iso_duration_to_seconds(dur_str)
        end = start + timedelta(seconds=seconds)
        return start, end
    except Exception:
        return None, None

def get_cloud_cover_now_next(lat: float, lon: float, timeout_s: float = 6.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cloud_now_pct": None,
        "cloud_now_valid": None,
        "cloud_next_pct": None,
        "cloud_next_valid": None,
    }

    try:
        meta_resp = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            timeout=timeout_s,
            headers=NWS_HEADERS,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        props = meta.get("properties", {})
        grid = props.get("gridId")
        x = props.get("gridX")
        y = props.get("gridY")
        if grid is None or x is None or y is None:
            return out

        grid_resp = requests.get(
            f"https://api.weather.gov/gridpoints/{grid}/{x},{y}",
            timeout=timeout_s,
            headers=NWS_HEADERS,
        )
        grid_resp.raise_for_status()
        grid_data = grid_resp.json()

        sky_vals = grid_data.get("properties", {}).get("skyCover", {}).get("values", [])
        if not sky_vals:
            return out

        now_utc = datetime.now(timezone.utc)

        parsed = []
        for entry in sky_vals:
            vt = entry.get("validTime")
            val = entry.get("value")
            if vt is None or val is None:
                continue
            start, end = _parse_valid_time(vt)
            if start is None or end is None:
                continue
            parsed.append((start, end, float(val), vt))

        if not parsed:
            return out

        parsed.sort(key=lambda t: t[0])

        current_idx = None
        for i, (start, end, val, vt) in enumerate(parsed):
            if start <= now_utc <= end:
                current_idx = i
                break
        if current_idx is None:
            current_idx = min(
                range(len(parsed)),
                key=lambda i: abs((parsed[i][0] - now_utc).total_seconds()),
            )

        start, end, val, vt = parsed[current_idx]
        out["cloud_now_pct"] = val
        out["cloud_now_valid"] = vt

        if current_idx + 1 < len(parsed):
            start2, end2, val2, vt2 = parsed[current_idx + 1]
            out["cloud_next_pct"] = val2
            out["cloud_next_valid"] = vt2

        return out
    except Exception:
        return out

# RA/Dec parsing
def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def parse_radec(ra: str, dec: str) -> SkyCoord:
    ra = (ra or "").strip().replace(" ", ":")
    dec = (dec or "").strip().replace(" ", ":")

    if not ra or not dec:
        raise ValueError("RA/Dec required for manual fallback.")

    if _is_number(ra) and 0.0 <= float(ra) <= 360.0:
        ra_unit = u.deg
    else:
        ra_unit = u.hourangle

    return SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame="icrs")

# Resolve target (SIMBAD -> fallback)
def resolve_target(name: str, ra: str, dec: str) -> ResolvedTarget:
    name = (name or "").strip()
    ra = (ra or "").strip()
    dec = (dec or "").strip()

    if name:
        try:
            coord = SkyCoord.from_name(name)
            vmag = "N/A"
            try:
                result = custom_simbad.query_object(name)
                if result is not None and "V" in result.colnames:
                    vmag = result["V"][0]
            except Exception:
                pass
            return ResolvedTarget(display_name=name, coord=coord, vmag=vmag, method="SIMBAD name lookup")
        except Exception:
            pass

    coord = parse_radec(ra, dec)
    disp = name if name else "Unnamed Target"
    return ResolvedTarget(display_name=disp, coord=coord, vmag="N/A", method="Manual RA/Dec")

# Planning time grid for a chosen local date
def planning_window_times(step_min: int = 2) -> Time:
    obs = get_observer()
    tz = ZoneInfo(get_site_config().timezone)

    d = get_planning_date()
    start_local = datetime(d.year, d.month, d.day, 17, 0, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=14)

    start = Time(start_local)
    end = Time(end_local)

    n = int(np.floor(((end - start).to(u.min).value) / step_min))
    return start + np.arange(0, n + 1) * step_min * u.min

def compute_visibility_windows(
    coord: SkyCoord,
    min_alt_deg: float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg: float = DEFAULT_MAX_ALT_DEG,
    step_min: int = 2,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    obs = get_observer()
    times = planning_window_times(step_min=step_min)
    alt = obs.altaz(times, coord).alt.deg
    mask = (alt >= min_alt_deg) & (alt <= max_alt_deg)

    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for g in groups:
        if len(g) == 0:
            continue
        t1 = pd.Timestamp(times[g[0]].to_datetime(timezone=obs.timezone))
        t2 = pd.Timestamp(times[g[-1]].to_datetime(timezone=obs.timezone))
        windows.append((t1, t2))

    return windows


def compute_visibility_window(
    coord: SkyCoord,
    min_alt_deg: float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg: float = DEFAULT_MAX_ALT_DEG,
    step_min: int = 2,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    windows = compute_visibility_windows(coord, min_alt_deg, max_alt_deg, step_min=step_min)
    if not windows:
        return None, None
    return windows[0][0], windows[-1][1]

def format_visibility_windows(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    time_fmt: str = "%H:%M",
) -> str:
    if not windows:
        return "—"

    parts = [f"{t1.strftime(time_fmt)}–{t2.strftime(time_fmt)}" for t1, t2 in windows]
    return "; ".join(parts)

# Sky conditions panel
def sky_conditions(timeout_s: float = 6.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    obs = get_observer()
    site = get_site_config()

    cloud = get_cloud_cover_now_next(site.lat, site.lon, timeout_s=timeout_s)
    out.update(cloud)

    tz = ZoneInfo(site.timezone)
    d = get_planning_date()

    anchor_local = datetime(d.year, d.month, d.day, 20, 0, 0, tzinfo=tz)
    anchor = Time(anchor_local)

    try:
        sunset = obs.sun_set_time(anchor, which="nearest")
        out["sunset_local"] = sunset.to_datetime(timezone=obs.timezone)
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

# Twilight spans helper
def _mask_to_spans(dt_list, mask: np.ndarray):
    spans = []
    mask = np.asarray(mask, dtype=bool)
    i, n = 0, len(mask)
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

# Altitude/Airmass plot
def plot_altitudes(
    coords: List[SkyCoord],
    names: List[str],
    min_alt_deg: float = DEFAULT_MIN_ALT_DEG,
    max_alt_deg: float = DEFAULT_MAX_ALT_DEG,
    y_mode: str = "altitude",
    only_names: Optional[List[str]] = None,
    display_tz: str = "local",  
) -> plt.Figure:
    obs = get_observer()
    times = planning_window_times(step_min=2)

    # choose display timezone
    if str(display_tz).lower() in ("utc", "z"):
        tz_disp = timezone.utc
        xlab = "UTC"
    else:
        tz_disp = obs.timezone  # local site tz from Observer
        xlab = "Local Time"

    dts = [t.to_datetime(timezone=tz_disp) for t in times]

    sun_alt = np.array(obs.sun_altaz(times).alt.deg)
    moon_alt = np.array(obs.moon_altaz(times).alt.deg)

    if only_names is not None:
        keep = [(c, n) for (c, n) in zip(coords, names) if n in set(only_names)]
        coords = [c for c, _ in keep]
        names  = [n for _, n in keep]

    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    day = sun_alt > 0
    civil = (sun_alt <= 0) & (sun_alt > -6)
    nautical = (sun_alt <= -6) & (sun_alt > -12)
    astro = (sun_alt <= -12) & (sun_alt > -18)
    night = sun_alt <= -18

    for a, b in _mask_to_spans(dts, day): ax.axvspan(a, b, alpha=0.10)
    for a, b in _mask_to_spans(dts, civil): ax.axvspan(a, b, alpha=0.15)
    for a, b in _mask_to_spans(dts, nautical): ax.axvspan(a, b, alpha=0.20)
    for a, b in _mask_to_spans(dts, astro): ax.axvspan(a, b, alpha=0.25)
    for a, b in _mask_to_spans(dts, night): ax.axvspan(a, b, alpha=0.30)

    def alt_to_airmass(alt_deg: np.ndarray) -> np.ndarray:
        alt_rad = np.deg2rad(alt_deg)
        sin_alt = np.sin(alt_rad)
        am = np.full_like(alt_deg, np.nan, dtype=float)
        good = sin_alt > 0
        am[good] = 1.0 / sin_alt[good]
        return am

    # Limit number of targets in preview
    MAX_PREVIEW_OBJECTS = 5

    coords_plot = coords[:MAX_PREVIEW_OBJECTS]
    names_plot = names[:MAX_PREVIEW_OBJECTS]

    for coord, nm in zip(coords_plot, names_plot):
        alt = obs.altaz(times, coord).alt.deg
        if y_mode.lower().startswith("air"):
            y = alt_to_airmass(alt)
            ax.plot(dts, y, "-", label=nm, linewidth=2.0)
        else:
            ax.plot(dts, alt, "-", label=nm, linewidth=2.0)

    ax.plot(dts, moon_alt, "--", linewidth=1.7, label="Moon (alt)")

    if y_mode.lower().startswith("air"):
        ax.set_ylabel("Airmass (sec z)", color="white")
        ax.set_ylim(1.0, 6.0)
        ax.invert_yaxis()
        am_min = 1.0 / np.sin(np.deg2rad(max_alt_deg))
        am_max = 1.0 / np.sin(np.deg2rad(min_alt_deg))
        ax.axhline(am_min, linestyle="--", alpha=0.7)
        ax.axhline(am_max, linestyle="--", alpha=0.7)
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

    leg = ax.legend(
        framealpha=0.75,
        fontsize=8,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.25,
    )
    for t in leg.get_texts():
        t.set_color("white")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.24)
    plt.close(fig)
    return fig

# Finder charts
def finder_figure_astroplan(coord: SkyCoord, name: str, fov_arcmin: int, survey: str, roll_deg: float = 0.0) -> plt.Figure:
    fixed = FixedTarget(coord=coord, name=name)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    try:
        plot_finder_image(fixed, survey=survey, fov_radius=fov_arcmin * u.arcmin, ax=ax)
        ax.set_title(f"{name} — {survey} — FOV={fov_arcmin}′", color="white", fontsize=13, pad=10)
    except Exception:
        ax.text(0.5, 0.5, f"Finder chart unavailable\n{survey}\nFOV={fov_arcmin}′",
                ha="center", va="center", color="white")
        ax.set_title(name, color="white", fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.04)

    if abs(float(roll_deg)) < 1e-9:
        plt.close(fig)
        return fig

    rgb = _render_figure_to_array(fig)
    plt.close(fig)
    rgb_rot = _rotate_rgb_image(rgb, roll_deg)

    fig2, ax2 = plt.subplots(figsize=(7.8, 6.4))
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    ax2.imshow(rgb_rot, origin="lower")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"{name} — {survey} — FOV={fov_arcmin}′ — Roll={float(roll_deg):.1f}°", color="white", fontsize=13, pad=10)
    ax2._rho_roll_deg = float(roll_deg)
    ax2._rho_data_shape = rgb.shape[:2]
    ax2._rho_wcs = None

    fig2.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.04)
    plt.close(fig2)
    return fig2


def _get_skyview_hdu(coord: SkyCoord, fov_arcmin: int, pixels: int, surveys: List[str]):
    for survey in surveys:
        try:
            hdus = SkyView.get_images(
                position=coord,
                survey=[survey],
                height=fov_arcmin * u.arcmin,
                width=fov_arcmin * u.arcmin,
                pixels=pixels,
            )
            if hdus and len(hdus) > 0 and len(hdus[0]) > 0:
                return survey, hdus[0][0]
        except Exception:
            continue
    return None, None

def _get_panstarrs_hdu(coord: SkyCoord, fov_arcmin: int, pixels: int, filt: str = PANSTARRS_FILTER):
    """Fetch a Pan-STARRS FITS cutout with WCS when available.

    Pan-STARRS is deeper than DSS but is not full-sky; requests can fail for
    positions outside the survey footprint.
    """
    try:
        params = {
            "ra": f"{float(coord.icrs.ra.deg):.8f}",
            "dec": f"{float(coord.icrs.dec.deg):.8f}",
            "filters": str(filt),
            "type": "stack",
            "sep": "comma",
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

        cutout_params = {
            "ra": f"{float(coord.icrs.ra.deg):.8f}",
            "dec": f"{float(coord.icrs.dec.deg):.8f}",
            "size": int(pixels),
            "format": "fits",
            "red": filename,
        }
        cutout_resp = requests.get(PANSTARRS_FITSCUT_URL, params=cutout_params, timeout=20)
        cutout_resp.raise_for_status()
        hdul = fits.open(BytesIO(cutout_resp.content))
        try:
            return hdul[0].copy()
        finally:
            hdul.close()
    except Exception:
        return None


def finder_figure_panstarrs(
    coord: SkyCoord,
    name: str,
    fov_arcmin: int,
    pixels: int = 700,
    roll_deg: float = 0.0,
) -> plt.Figure:
    applied_roll = float(roll_deg)
    if abs(applied_roll) > 1e-9:
        pad = _roll_pad_factor(applied_roll, dec_deg=float(coord.icrs.dec.deg))
        fetch_fov_arcmin = int(np.ceil(float(fov_arcmin) * pad))
        fetch_pixels = int(np.ceil(int(pixels) * pad))
    else:
        fetch_fov_arcmin = int(fov_arcmin)
        fetch_pixels = int(pixels)

    hdu = _get_panstarrs_hdu(coord, fetch_fov_arcmin, fetch_pixels, filt=PANSTARRS_FILTER)

    fig = plt.figure(figsize=(7.8, 6.4))
    fig.patch.set_facecolor("black")

    if hdu is None:
        ax = fig.add_subplot(111)
        ax.set_facecolor("black")
        ax.text(0.5, 0.5, "Pan-STARRS cutout unavailable\n(outside footprint or service unavailable)",
                ha="center", va="center", color="white")
        ax.set_title(f"{name} — Pan-STARRS", color="white", fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.close(fig)
        return fig

    data = np.asarray(hdu.data, dtype=float)
    if data.ndim > 2:
        data = np.squeeze(data)
    wcs = WCS(hdu.header).celestial

    if abs(applied_roll) > 1e-9:
        data_plot, wcs_plot = regenerate_rolled_finder_data_with_padding(
            data, wcs, applied_roll, (int(pixels), int(pixels))
        )
    else:
        data_plot, wcs_plot = data, wcs
        applied_roll = 0.0

    plot_arr = np.asarray(data_plot, dtype=float)
    finite = plot_arr[np.isfinite(plot_arr)]
    if finite.size:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    else:
        vmin, vmax = 0.0, 1.0
    plot_masked = np.ma.masked_invalid(plot_arr)

    ax = fig.add_subplot(111, projection=wcs_plot)
    ax.set_facecolor("black")
    ax.imshow(plot_masked, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")

    try:
        ax.coords[0].set_ticks_position('b')
        ax.coords[0].set_ticklabel_position('b')
        ax.coords[0].set_axislabel_position('b')
        ax.coords[1].set_ticks_position('l')
        ax.coords[1].set_ticklabel_position('l')
        ax.coords[1].set_axislabel_position('l')
    except Exception:
        pass

    if abs(applied_roll) < 1e-9:
        ax.coords.grid(color='white', alpha=0.20, linestyle='--')
    ax.coords[0].set_axislabel('RA', color='white')
    ax.coords[1].set_axislabel('Dec', color='white')
    ax.coords[0].set_ticklabel(color='white')
    ax.coords[1].set_ticklabel(color='white')

    title = f"{name} — Pan-STARRS — FOV={fov_arcmin}′"
    if abs(applied_roll) > 1e-9:
        title += f" — Roll={applied_roll:.1f}°"
    fig.suptitle(title, color='white', fontsize=13, y=0.985)

    ax._rho_roll_deg = 0.0
    ax._rho_data_shape = np.asarray(data_plot).shape
    ax._rho_wcs = wcs_plot

    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08)
    plt.close(fig)
    return fig


def finder_figure_skyview(
    coord: SkyCoord,
    name: str,
    fov_arcmin: int,
    pixels: int = 700,
    roll_deg: float = 0.0,
    surveys: List[str] | None = None,
) -> plt.Figure:
    survey_list = list(surveys) if surveys else SKYVIEW_SURVEYS

    applied_roll = float(roll_deg)
    if abs(applied_roll) > 1e-9:
        pad = _roll_pad_factor(applied_roll, dec_deg=float(coord.icrs.dec.deg))
        fetch_fov_arcmin = int(np.ceil(float(fov_arcmin) * pad))
        fetch_pixels = int(np.ceil(int(pixels) * pad))
    else:
        fetch_fov_arcmin = int(fov_arcmin)
        fetch_pixels = int(pixels)

    used, hdu = _get_skyview_hdu(coord, fetch_fov_arcmin, fetch_pixels, survey_list)

    fig = plt.figure(figsize=(7.8, 6.4))
    fig.patch.set_facecolor('black')

    if hdu is None:
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.text(0.5, 0.5, f"Finder chart unavailable\nFOV={fov_arcmin}′",
                ha='center', va='center', color='white')
        ax.set_title(f"{name}", color='white', fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.close(fig)
        return fig

    data = np.asarray(hdu.data, dtype=float)
    wcs = WCS(hdu.header).celestial

    if abs(applied_roll) > 1e-9:
        # Use a padded fetch + centered affine rotation path here. It is less brittle
        # than the experimental reprojection path and keeps the displayed scale sane.
        data_plot, wcs_plot = regenerate_rolled_finder_data_with_padding(
            data, wcs, applied_roll, (int(pixels), int(pixels))
        )
    else:
        data_plot, wcs_plot = data, wcs
        applied_roll = 0.0

    plot_arr = np.asarray(data_plot, dtype=float)
    finite = plot_arr[np.isfinite(plot_arr)]
    if finite.size:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    else:
        vmin, vmax = 0.0, 1.0
    plot_masked = np.ma.masked_invalid(plot_arr)

    ax = fig.add_subplot(111, projection=wcs_plot)
    ax.set_facecolor('black')
    ax.imshow(plot_masked, origin='lower', vmin=vmin, vmax=vmax, cmap='gray', interpolation='nearest')

    try:
        ax.coords[0].set_ticks_position('b')
        ax.coords[0].set_ticklabel_position('b')
        ax.coords[0].set_axislabel_position('b')
        ax.coords[1].set_ticks_position('l')
        ax.coords[1].set_ticklabel_position('l')
        ax.coords[1].set_axislabel_position('l')
    except Exception:
        pass

    if abs(applied_roll) < 1e-9:
        ax.coords.grid(color='white', alpha=0.20, linestyle='--')
    ax.coords[0].set_axislabel('RA', color='white')
    ax.coords[1].set_axislabel('Dec', color='white')
    ax.coords[0].set_ticklabel(color='white')
    ax.coords[1].set_ticklabel(color='white')

    title = f"{name} — {used} — FOV={fov_arcmin}′"
    if abs(applied_roll) > 1e-9:
        title += f" — Roll={applied_roll:.1f}°"
    fig.suptitle(title, color='white', fontsize=13, y=0.985)

    ax._rho_roll_deg = 0.0
    ax._rho_data_shape = np.asarray(data_plot).shape
    ax._rho_wcs = wcs_plot

    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08)
    plt.close(fig)
    return fig


def finder_figure(coord: SkyCoord, name: str, fov_arcmin: int, mode: str, roll_deg: float = 0.0) -> plt.Figure:
    mode = str(mode or "DSS").strip()

    if mode in {"DSS", "DSS2 Red", "DSS2 Blue", "SkyView"}:
        # Keep legacy "SkyView" support by mapping it to the generic DSS option.
        if mode == "SkyView":
            mode = "DSS"
        fig = finder_figure_skyview(coord, name, fov_arcmin, pixels=700, roll_deg=roll_deg, surveys=[mode])
        return fig

    if mode == "Pan-STARRS":
        return finder_figure_panstarrs(coord, name, fov_arcmin, pixels=700, roll_deg=roll_deg)

    # Fallback for any custom/unsupported Astroplan survey names.
    return finder_figure_astroplan(coord, name, fov_arcmin, survey=mode, roll_deg=roll_deg)


# Upload parsing
def _norm_col(s: str) -> str:
    s = (s or "")
    s = s.replace("\n", " ").strip().lower()
    s = re.sub(r"[\*\(\)\[\]\{\}:,\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
        code_col = None
        for c in df.columns:
            if _norm_col(str(c)) == "code":
                code_col = c
                break

        if code_col is not None:
            v = str(df.iloc[0][code_col]).strip()
            if "###" in v or v.upper() == "YYS###":
                return df.iloc[1:].reset_index(drop=True)

        checks = []
        for col_name, token in [
            ("Primary Identifier**", "string"),
            ("V Magnitude**", "float"),
            ("Priority**", "integer"),
            ("RA**", "hh mm ss"),
            ("Dec**", "deg min sec"),
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
        xls = pd.ExcelFile(path)
        sheet = "TargetMasterSheet" if "TargetMasterSheet" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)

    df = _drop_template_first_data_row(df)

    name_col = find_col(df, ["primary identifier", "Primary Identifier**", "object name", "name"])
    ra_col   = find_col(df, ["ra", "RA**", "radeg", "ra_deg"])
    dec_col  = find_col(df, ["dec", "Dec**", "dedeg", "decdeg", "dec_deg", "decdeg", "dec_deg"])

    pr_col = None
    for cand in (["priority", "Priority**"], ["prio"], ["rank"]):
        try:
            pr_col = find_col(df, cand)
            break
        except Exception:
            pass

    def _pick_numeric_column(df: pd.DataFrame, candidate_names: List[str]) -> str | None:
        cand_norm = {_norm_col(c) for c in candidate_names}

        matches: List[tuple[int, str]] = []
        for col in df.columns:
            if _norm_col(str(col)) not in cand_norm:
                continue

            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                continue

            num = pd.to_numeric(s, errors="coerce")
            score = int(num.notna().sum())
            matches.append((score, col))

        if not matches:
            return None

        matches.sort(reverse=True, key=lambda t: t[0])
        best_score, best_col = matches[0]
        if best_score == 0:
            return None
        return best_col

    vmag_col = _pick_numeric_column(
        df,
        ["v magnitude", "V Magnitude**", "vmag", "v_mag", "mag_v", "Vmag", "V"]
    )

    out = pd.DataFrame({
        "name": df[name_col].astype(str),
        "ra": df[ra_col].astype(str),
        "dec": df[dec_col].astype(str),
        "priority": pd.to_numeric(df[pr_col], errors="coerce").fillna(3).astype(int) if pr_col else 3,
        "vmag": pd.to_numeric(df[vmag_col], errors="coerce") if vmag_col else np.nan,
    })

    return out
