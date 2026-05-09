from __future__ import annotations
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
from dcwb.profile import Profile
from dcwb.matrix import from_diag, compose, Matrix3x3
from dcwb.awb import shades_of_gray
from dcwb.ffmpeg_wrap import probe_duration, extract_frame, render_with_matrix
from dcwb.calibrate import read_event_timestamp, read_event_latlon
from dcwb.daylight import is_daytime

CAMERAS = (
    "front", "back",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
)

def compose_clip_matrix(
    profile: Profile,
    scene_gain: tuple[float, float, float],
    gain_min: float = 0.7,
    gain_max: float = 1.5,
    attenuation: float = 1.0,
) -> Matrix3x3:
    """Combine A (profile) and B (scene gain) into a single 3x3.

    If any scene_gain channel falls outside [gain_min, gain_max], B is dropped
    entirely (fallback to A only). attenuation in [0, 1] linearly weakens B
    toward identity (used for night attenuation).
    """
    g_r, g_g, g_b = scene_gain
    if any(g < gain_min or g > gain_max for g in (g_r, g_g, g_b)):
        return profile.matrix_3x3
    if attenuation < 1.0:
        g_r = 1.0 + (g_r - 1.0) * attenuation
        g_g = 1.0 + (g_g - 1.0) * attenuation
        g_b = 1.0 + (g_b - 1.0) * attenuation
    return compose(from_diag(g_r, g_g, g_b), profile.matrix_3x3)

def _camera_of(clip: Path) -> str:
    # filename format: <ts>-<camera>.mp4 (camera may include _ e.g. left_pillar)
    stem = clip.stem
    for cam in CAMERAS:
        if stem.endswith("-" + cam):
            return cam
    raise ValueError(f"could not infer camera from filename: {clip.name}")

def estimate_scene_gain(
    clip: Path,
    profile: Profile,
    samples_per_clip: int,
    sat_high: float,
    sat_low: float,
    p: int,
) -> tuple[float, float, float]:
    duration = probe_duration(clip)
    ts_list = [duration * (i + 0.5) / samples_per_clip for i in range(samples_per_clip)]
    gains: list[tuple[float, float, float]] = []
    for t in ts_list:
        img_rgb = extract_frame(clip, t)
        # Apply A (profile) before estimating B
        flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
        flat = flat @ profile.matrix_3x3.T
        np.clip(flat, 0.0, 1.0, out=flat)
        img_a = (flat.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
        gains.append(shades_of_gray(img_a, p=p, sat_high=sat_high, sat_low=sat_low))
    g = np.array(gains).mean(axis=0)
    return float(g[0]), float(g[1]), float(g[2])

def render_event(
    event_dir: Path,
    out_root: Path,
    profiles_dir: Path,
    pipeline_cfg: dict,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
) -> None:
    """Render every camera mp4 in event_dir to out_root/<event_name>/."""
    awb_cfg = pipeline_cfg["awb"]
    out_dir = out_root / event_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = {
        cam: Profile.from_json(profiles_dir / f"{cam}.json")
        for cam in CAMERAS
    }

    # Event-level night attenuation: pull from event.json timestamp + lat/lon.
    ts = read_event_timestamp(event_dir)
    lat, lon = read_event_latlon(event_dir)
    if ts is None:
        attenuation = 1.0
    else:
        attenuation = 1.0 if is_daytime(ts, lat=lat, lon=lon) else float(awb_cfg["night_attenuation"])

    snapshot: dict = {
        "event": event_dir.name,
        "attenuation": attenuation,
        "clips": [],
    }

    for clip in sorted(event_dir.glob("*.mp4")):
        try:
            cam = _camera_of(clip)
            prof = profiles[cam]
            scene_gain = estimate_scene_gain(
                clip, prof,
                samples_per_clip=int(awb_cfg["samples_per_clip"]),
                sat_high=float(awb_cfg["saturation_high"]),
                sat_low=float(awb_cfg["saturation_low"]),
                p=int(awb_cfg["minkowski_p"]),
            )
            final_matrix = compose_clip_matrix(
                prof, scene_gain,
                gain_min=float(awb_cfg["gain_min"]),
                gain_max=float(awb_cfg["gain_max"]),
                attenuation=attenuation,
            )
            render_with_matrix(
                clip, out_dir / clip.name, final_matrix,
                bitrate_kbps=bitrate_kbps, encoder=encoder,
            )
            snapshot["clips"].append({
                "clip": clip.name, "camera": cam,
                "scene_gain": list(scene_gain),
                "final_matrix": final_matrix.tolist(),
            })
        except Exception as e:
            print(f"[render] FAILED {clip.name}: {e}", file=sys.stderr)
            try:
                cam_for_err = _camera_of(clip)
            except Exception:
                cam_for_err = None
            snapshot["clips"].append({
                "clip": clip.name,
                "camera": cam_for_err,
                "error": str(e),
            })
            continue

    # event.json copy
    src_meta = event_dir / "event.json"
    if src_meta.exists():
        try:
            shutil.copy2(src_meta, out_dir / "event.json")
        except Exception as e:
            print(f"[render] FAILED event.json copy: {e}", file=sys.stderr)

    # thumb.png: render through the front camera matrix if present
    src_thumb = event_dir / "thumb.png"
    if src_thumb.exists():
        try:
            import cv2
            img_bgr = cv2.imread(str(src_thumb), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
                corrected = flat @ profiles["front"].matrix_3x3.T
                np.clip(corrected, 0.0, 1.0, out=corrected)
                out_rgb = (corrected.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / "thumb.png"), out_bgr)
        except Exception as e:
            print(f"[render] FAILED thumb.png: {e}", file=sys.stderr)

    # snapshot
    snapshot["rendered_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "_pipeline.json").write_text(json.dumps(snapshot, indent=2))
