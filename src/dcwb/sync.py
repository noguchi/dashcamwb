from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def unwrap_yaw_rate(headings_deg: np.ndarray, dt: float) -> np.ndarray:
    """deg/s yaw rate from a heading series, robust to 0/360 wraparound."""
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, float))))
    return np.gradient(unwrapped, dt)

def resample_uniform(t: np.ndarray, v: np.ndarray, rate_hz: float):
    """Linearly resample irregular (t, v) onto a uniform grid at rate_hz."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    grid_t = np.arange(t[0], t[-1] + 1e-9, 1.0 / rate_hz)
    grid_v = np.interp(grid_t, t, v)
    return grid_t, grid_v

def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Return (best_lag, peak) maximizing normalized cross-correlation of a vs b
    over integer lags in [-max_lag, max_lag]. Positive lag means b lags a."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    best_lag, best = 0, -np.inf
    n = min(len(a), len(b))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:n - lag], b[lag:n]
        else:
            x, y = a[-lag:n], b[:n + lag]
        if len(x) < 2:
            continue
        score = float(np.dot(x, y) / len(x))
        if score > best:
            best, best_lag = score, lag
    return best_lag, best


@dataclass
class MotionSeries:
    t: np.ndarray        # absolute seconds, monotonic
    yaw_rate: np.ndarray
    accel_x: np.ndarray

@dataclass
class SyncResult:
    delta_s: float       # insta.t[0]-tesla.t[0] + correlation lag; see note
    confidence: float    # cross-correlation peak in [-1, 1]
    signal: str          # "yaw_rate" or "accel_x"
    anchor_guess: float

def compute_offset(tesla: MotionSeries, insta: MotionSeries,
                   anchor_guess: float, window_s: float, rate_hz: float) -> SyncResult:
    """Refine the coarse anchor to a fine delta via cross-correlation.

    delta_s is defined as (insta.t[0] - tesla.t[0]) plus the residual correlation
    lag, i.e. the offset that maps the insta timeline onto the tesla timeline.
    The rendering sign convention is calibrated against real data in the CLI task.
    Tries yaw-rate first; falls back to accel_x when its correlation peak is weaker.
    """
    base = float(insta.t[0] - tesla.t[0])
    max_lag = int((window_s + abs(anchor_guess)) * rate_hz)

    def _corr(field: str):
        _, ta = resample_uniform(tesla.t - tesla.t[0], getattr(tesla, field), rate_hz)
        _, ia = resample_uniform(insta.t - insta.t[0], getattr(insta, field), rate_hz)
        lag, peak = normalized_xcorr(ta, ia, max_lag=max_lag)
        return base + lag / rate_hz, peak

    delta_y, peak_y = _corr("yaw_rate")
    delta_a, peak_a = _corr("accel_x")
    if peak_y >= peak_a:
        return SyncResult(delta_y, peak_y, "yaw_rate", anchor_guess)
    return SyncResult(delta_a, peak_a, "accel_x", anchor_guess)


import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
_FRONT_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-front\.mp4$")

def _front_start(name: str):
    m = _FRONT_RE.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=JST)

def select_front_clips(day_dir: Path, start: datetime, end: datetime,
                       seg_seconds: float = 60.0) -> list[Path]:
    """Front clips whose [clip_start, clip_start+seg] overlaps [start, end]."""
    out = []
    for p in sorted(day_dir.glob("*-front.mp4")):
        cs = _front_start(p.name)
        if cs is None:
            continue
        ce = cs + timedelta(seconds=seg_seconds)
        if ce > start and cs < end:
            out.append(p)
    return out


def _ass_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def telemetry_ass(rows, play_w: int, play_h: int) -> str:
    """Build an ASS subtitle (speed/steer/gear) with one event per row.
    rows: list of (t_seconds, speed_mps, steering_deg, gear)."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, "
        "Bold, Alignment, MarginL, MarginR, MarginV, BorderStyle, Outline, Shadow\n"
        "Style: tele,DejaVu Sans Mono,42,&H00FFFFFF,&H80000000,1,1,40,40,40,3,2,0\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
    )
    lines = []
    for i, (t, mps, steer, gear) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 1.0
        kmh = round(mps * 3.6)
        text = f"{kmh} km/h   steer {steer:+.0f}\\N{gear}"
        lines.append(f"Dialogue: 0,{_ass_ts(t)},{_ass_ts(end)},tele,{text}")
    return header + "\n".join(lines) + "\n"
