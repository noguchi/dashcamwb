from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dcwb.profile import Profile
from dcwb.matrix import from_diag
from dcwb.ffmpeg_wrap import probe_duration, extract_frame
from dcwb.awb import shades_of_gray
from dcwb.render import compose_clip_matrix, _camera_of, CAMERAS, estimate_scene_gain

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Defaults mirror project's pipeline.json so callers can omit pipeline_cfg.
_DEFAULT_PIPELINE_CFG: dict = {
    "awb": {
        "method": "shades_of_gray",
        "minkowski_p": 6,
        "samples_per_clip": 10,
        "saturation_high": 0.97,
        "saturation_low": 0.03,
        "gain_min": 0.7,
        "gain_max": 1.5,
        "night_attenuation": 0.5,
    }
}

def _to_base64_png(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _apply_matrix(img_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    out = flat @ matrix.T
    np.clip(out, 0.0, 1.0, out=out)
    return (out.reshape(img_rgb.shape) * 255.0).astype(np.uint8)

def generate_verify_report(
    event_dir: Path,
    profiles_dir: Path,
    out_html: Path,
    encoder: str = "h264_videotoolbox",
    pipeline_cfg: dict | None = None,
) -> None:
    cfg = pipeline_cfg if pipeline_cfg is not None else _DEFAULT_PIPELINE_CFG
    awb_cfg = cfg["awb"]
    profiles = {
        cam: Profile.from_json(profiles_dir / f"{cam}.json")
        for cam in CAMERAS
    }
    rows = []
    for clip in sorted(event_dir.glob("*.mp4")):
        cam = _camera_of(clip)
        prof = profiles[cam]
        # 1 フレームを中央時刻から (表示用)
        duration = probe_duration(clip)
        before = extract_frame(clip, duration / 2.0)
        a_only = _apply_matrix(before, prof.matrix_3x3)
        # B は render と同じ N-フレーム平均で推定 (pipeline_cfg 由来)
        scene_gain = estimate_scene_gain(
            clip, prof,
            samples_per_clip=int(awb_cfg["samples_per_clip"]),
            sat_high=float(awb_cfg["saturation_high"]),
            sat_low=float(awb_cfg["saturation_low"]),
            p=int(awb_cfg["minkowski_p"]),
        )
        final_m = compose_clip_matrix(
            prof, scene_gain,
            gain_min=float(awb_cfg["gain_min"]),
            gain_max=float(awb_cfg["gain_max"]),
        )
        full = _apply_matrix(before, final_m)
        rows.append({
            "camera": cam,
            "scene_gain": [round(g, 3) for g in scene_gain],
            "before": _to_base64_png(before),
            "a_only": _to_base64_png(a_only),
            "full": _to_base64_png(full),
        })
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tmpl = env.get_template("verify.html.j2")
    out_html.write_text(tmpl.render(event_name=event_dir.name, rows=rows))
