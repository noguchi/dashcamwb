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
from dcwb.render import compose_clip_matrix, _camera_of, CAMERAS

TEMPLATES_DIR = Path(__file__).parent / "templates"

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
) -> None:
    profiles = {
        cam: Profile.from_json(profiles_dir / f"{cam}.json")
        for cam in CAMERAS
    }
    rows = []
    for clip in sorted(event_dir.glob("*.mp4")):
        cam = _camera_of(clip)
        prof = profiles[cam]
        # 1 フレームを中央時刻から
        duration = probe_duration(clip)
        before = extract_frame(clip, duration / 2.0)
        a_only = _apply_matrix(before, prof.matrix_3x3)
        # B を 1 サンプルで簡易計算
        scene_gain = shades_of_gray(a_only, p=6)
        final_m = compose_clip_matrix(prof, scene_gain)
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
