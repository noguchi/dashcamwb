from __future__ import annotations
import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dcwb.profile import Profile
from dcwb.render import CAMERAS, _camera_of
from dcwb.serve.preview import compute_frame_triple, to_png_bytes

TEMPLATES_DIR = Path(__file__).parent / "templates"

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


def _b64(img_rgb) -> str:
    return base64.b64encode(to_png_bytes(img_rgb)).decode("ascii")


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
        triple = compute_frame_triple(clip, profiles[cam], awb_cfg)
        rows.append({
            "camera": cam,
            "scene_gain": [round(g, 3) for g in triple.scene_gain],
            "before": _b64(triple.before),
            "a_only": _b64(triple.a_only),
            "full": _b64(triple.full),
        })
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tmpl = env.get_template("verify.html.j2")
    out_html.write_text(tmpl.render(event_name=event_dir.name, rows=rows))
