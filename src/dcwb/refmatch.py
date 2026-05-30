"""Reference-camera match gain: replace the B (scene-light) layer with a gain
that matches the Tesla cameras' tone to a consumer reference camera
(Insta360 / iPhone / ...), so composites of the two read as visually
consistent. See docs/superpowers/specs/2026-05-31-reference-camera-match-gain-design.md
and docs/adr/0001-consumer-camera-reference-color-matching.md.

The goal is perceptual consistency for compositing, NOT colorimetric accuracy:
the reference need not be a calibrated standard.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dcwb.profile import Profile
from dcwb.awb import shades_of_gray
from dcwb.daylight import is_daytime, TOKYO_LAT, TOKYO_LON
from dcwb.ffmpeg_wrap import probe_duration, concat_clips, reframe_insv, extract_frame
from dcwb.sync import detect_visual_offset, _front_start

# v360 forward-flat reframe used for .insv references — same framing the sync
# player uses so the extracted ride-view matches what gets composited.
DEFAULT_REFRAME = dict(yaw=180.0, pitch=14.0, roll=180.0, h_fov=82.0, v_fov=52.0)
# Center outdoor crop (fractions of w/h): avoid the Insta cabin / Tesla bonnet
# at the bottom and the often-blown-out sky band at the very top.
DEFAULT_CROP = (0.25, 0.20, 0.50, 0.40)  # x, y, w, h
_EOF_GUARD_SEC = 0.3


class ReferenceGainError(RuntimeError):
    """Raised when a reference match gain cannot be computed (no sync / no
    daytime driving frames)."""


def _center_crop(img: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    fx, fy, fw, fh = crop
    x0 = int(w * fx); y0 = int(h * fy)
    x1 = min(w, x0 + int(w * fw)); y1 = min(h, y0 + int(h * fh))
    sub = img[y0:y1, x0:x1]
    return sub if sub.size else img


def reference_gain(
    tesla_sog: tuple[float, float, float],
    ref_sog: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Match gain that pulls A-corrected Tesla toward the reference's rendition.

    Both inputs are Shades-of-Gray gains (the ``awb.shades_of_gray`` return
    form: green-normalised ``(g_r, g_g, g_b)`` that *neutralise* their camera).
    ``shades_of_gray`` returns the gain that neutralises a camera, so the
    residual illuminant of A-corrected Tesla is ``1/g_T`` and the reference's is
    ``1/g_R``. Applying gain ``G`` to Tesla makes its residual ``G/g_T``; to
    match the reference's ``1/g_R`` we need ``G = g_T / g_R`` (per channel).

    The result is re-normalised so ``g_g == 1`` (the downstream
    ``compose_clip_matrix`` uses ``gain_min``/``gain_max`` symmetric around 1).
    A neutral reference (g_R = 1,1,1) collapses ``G`` to ``g_T`` — i.e. the
    plain gray-world neutralisation (continuity with the no-reference path).
    """
    g = tuple(t / r for t, r in zip(tesla_sog, ref_sog))
    g_g = g[1]
    return g[0] / g_g, 1.0, g[2] / g_g


def _sog_after_a(img_rgb: np.ndarray, a_matrix: np.ndarray | None,
                 sat_high: float, sat_low: float, p: int) -> tuple[float, float, float]:
    """Shades-of-Gray gains of an RGB frame, optionally after applying matrix A
    (row-vector convention: ``flat @ A.T``, clipped to 0..1)."""
    if a_matrix is not None:
        flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
        flat = flat @ a_matrix.T
        np.clip(flat, 0.0, 1.0, out=flat)
        img_rgb = (flat.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
    return shades_of_gray(img_rgb, p=p, sat_high=sat_high, sat_low=sat_low)


def compute_reference_gain(
    reference: Path,
    fronts: list[Path],
    front_profile: Profile,
    *,
    start_jst: datetime,
    work_dir: Path,
    samples: int = 10,
    max_window: float | None = 600.0,
    is_insv: bool | None = None,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    reframe: dict | None = None,
    crop: tuple[float, float, float, float] = DEFAULT_CROP,
    lat: float = TOKYO_LAT,
    lon: float = TOKYO_LON,
    sat_high: float = 0.97,
    sat_low: float = 0.03,
    p: int = 6,
) -> tuple[tuple[float, float, float], float, int]:
    """Orchestrate: reference video + Tesla front clips -> reference match gain.

    Steps (see the design spec):
      1. concat the front clips and visually time-sync the reference to them
         (pure-pixel frame-difference cross-correlation; no telemetry/IMU).
      2. sample ~``samples`` daytime, driving frame pairs across the synced window.
      3. apply front A to the Tesla frame, run Shades-of-Gray on both (center
         outdoor crop), and take ``reference_gain`` per pair.
      4. aggregate the per-pair gains by geometric mean (robust for multiplicative
         gains).

    Returns ``((g_r, 1, g_b), sync_peak, n_pairs)``. Raises ``ReferenceGainError``
    when the reference cannot be synced or no daytime frames are usable.
    """
    if is_insv is None:
        is_insv = reference.suffix.lower() == ".insv"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- coarse anchor from front filenames (file clocks only) ---
    cs0 = _front_start(fronts[0].name)
    if cs0 is None:
        raise ReferenceGainError(f"cannot parse front clip start time: {fronts[0].name}")
    anchor_tesla_lead = -(cs0 - start_jst).total_seconds()

    # --- Tesla concat (covers visual detection + frame sampling) ---
    tesla_cat = work_dir / "tesla-concat.mp4"
    concat_clips(fronts, tesla_cat, encoder=encoder, bitrate_kbps=bitrate_kbps)
    tesla_total = probe_duration(tesla_cat)
    insv_total = probe_duration(reference)
    # Bound the work: a 30-min .insv over a network mount must not be reframed or
    # sampled in full. ~10 daytime frames within the first max_window seconds are
    # plenty for a stable match gain. max_window=None disables the cap.
    ref_dur = insv_total if max_window is None else min(insv_total, float(max_window))

    # --- visual fine alignment: instant at ref video time v -> Tesla epoch v+off,
    # which is Tesla-concat-local tc = v + off + anchor_tesla_lead. ---
    offset, peak = detect_visual_offset(reference, tesla_cat, anchor_tesla_lead, ref_dur)

    # --- reference frames: reframe .insv to forward-flat once, else use as-is ---
    if is_insv:
        ref_flat = work_dir / "ref-flat.mp4"
        rf = {**DEFAULT_REFRAME, **(reframe or {})}
        reframe_insv(reference, ref_flat, encoder=encoder, bitrate_kbps=bitrate_kbps,
                     out_w=1280, out_h=720, start=0.0, duration=ref_dur, **rf)
        ref_src = ref_flat
    else:
        ref_src = reference

    # --- usable reference-time window where both frames exist ---
    lo = max(0.0, -offset - anchor_tesla_lead) + _EOF_GUARD_SEC
    hi = min(ref_dur, tesla_total - offset - anchor_tesla_lead) - _EOF_GUARD_SEC
    if hi <= lo:
        raise ReferenceGainError(
            f"no overlapping window between reference and Tesla front "
            f"(ref={insv_total:.0f}s tesla={tesla_total:.0f}s offset={offset:.1f}s)")
    usable = hi - lo
    v_list = [lo + usable * (i + 0.5) / samples for i in range(samples)]

    a_matrix = front_profile.matrix_3x3
    pair_gains: list[tuple[float, float, float]] = []
    daytime_seen = False
    for v in v_list:
        abs_jst = start_jst + timedelta(seconds=v)
        if not is_daytime(abs_jst, lat=lat, lon=lon):
            continue
        daytime_seen = True
        tc = v + offset + anchor_tesla_lead
        try:
            ref_img = _center_crop(extract_frame(ref_src, v), crop)
            tesla_img = _center_crop(extract_frame(tesla_cat, tc), crop)
        except RuntimeError:
            continue
        g_r = _sog_after_a(ref_img, None, sat_high, sat_low, p)
        g_t = _sog_after_a(tesla_img, a_matrix, sat_high, sat_low, p)
        pair_gains.append(reference_gain(g_t, g_r))

    if not daytime_seen:
        raise ReferenceGainError(
            "no daytime frames in the synced window (reference B-layer "
            "replacement assumes a daytime drive)")
    if not pair_gains:
        raise ReferenceGainError("no frame pairs could be extracted for matching")

    arr = np.array(pair_gains)
    geo = np.exp(np.log(arr).mean(axis=0))
    gain = (float(geo[0]), float(geo[1]), float(geo[2]))
    return gain, float(peak), len(pair_gains)
