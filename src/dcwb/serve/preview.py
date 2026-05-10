from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from dcwb.profile import Profile
from dcwb.ffmpeg_wrap import probe_duration, extract_frame
from dcwb.render import compose_clip_matrix, estimate_scene_gain


@dataclass
class FrameTriple:
    before: np.ndarray   # uint8 RGB, central frame
    a_only: np.ndarray   # uint8 RGB after profile matrix only
    full: np.ndarray     # uint8 RGB after profile × scene-gain
    scene_gain: tuple[float, float, float]


def _apply_matrix(img_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    out = flat @ matrix.T
    np.clip(out, 0.0, 1.0, out=out)
    return (out.reshape(img_rgb.shape) * 255.0).astype(np.uint8)


def compute_frame_triple(
    clip: Path,
    profile: Profile,
    awb_cfg: dict,
) -> FrameTriple:
    duration = probe_duration(clip)
    before = extract_frame(clip, duration / 2.0)
    a_only = _apply_matrix(before, profile.matrix_3x3)
    scene_gain = estimate_scene_gain(
        clip, profile,
        samples_per_clip=int(awb_cfg["samples_per_clip"]),
        sat_high=float(awb_cfg["saturation_high"]),
        sat_low=float(awb_cfg["saturation_low"]),
        p=int(awb_cfg["minkowski_p"]),
    )
    final_m = compose_clip_matrix(
        profile, scene_gain,
        gain_min=float(awb_cfg["gain_min"]),
        gain_max=float(awb_cfg["gain_max"]),
    )
    full = _apply_matrix(before, final_m)
    return FrameTriple(before=before, a_only=a_only, full=full, scene_gain=scene_gain)


def to_png_bytes(img_rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()
