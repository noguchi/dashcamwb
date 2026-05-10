from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from dcwb.profile import Profile
from dcwb.ffmpeg_wrap import probe_duration, extract_frame
from dcwb.render import compose_clip_matrix, estimate_scene_gain
from dcwb.serve.index import Event, CAMERAS


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


@dataclass
class PreviewResult:
    paths: dict[str, dict[str, Path]]      # cam -> {"before": Path, "after": Path}
    scene_gains: dict[str, list[float]]    # cam -> [r, g, b]
    errors: dict[str, str | None]


def _camera_of(clip: Path) -> str:
    stem = clip.stem
    for cam in CAMERAS:
        if stem.endswith("-" + cam):
            return cam
    raise ValueError(f"could not infer camera from filename: {clip.name}")


def _clip_for_camera(event: Event, cam: str) -> Path | None:
    # return the first clip whose camera suffix matches; for RecentClips
    # pseudo-events this picks the chronologically-first clip of the cam.
    for clip in event.clips:
        try:
            if _camera_of(clip) == cam:
                return clip
        except ValueError:
            continue
    return None


def _cache_dir(event: Event, cache_root: Path) -> Path:
    return cache_root / "previews" / event.source / event.name


def _profile_mtimes(profiles_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for cam in CAMERAS:
        p = profiles_dir / f"{cam}.json"
        out[cam] = p.stat().st_mtime if p.exists() else 0.0
    return out


def _pipeline_cfg_hash(awb_cfg: dict) -> str:
    import hashlib
    blob = json.dumps(awb_cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _meta_current(meta: dict, profile_mtimes: dict[str, float], cfg_hash: str) -> bool:
    if meta.get("profile_mtimes") != profile_mtimes:
        return False
    if meta.get("pipeline_cfg_hash") != cfg_hash:
        return False
    return True


def _write_placeholder(path: Path, label: str) -> None:
    h, w = 240, 320
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    cv2.putText(img, label, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _render_one_camera(
    cam: str,
    clip: Path | None,
    profiles_dir: Path,
    awb_cfg: dict,
    cache_dir: Path,
) -> tuple[str, tuple[float, float, float] | None, str | None]:
    before_p = cache_dir / f"{cam}_before.png"
    after_p = cache_dir / f"{cam}_after.png"
    if clip is None:
        _write_placeholder(before_p, "no clip")
        _write_placeholder(after_p, "no clip")
        return cam, None, "no clip for camera"
    try:
        prof = Profile.from_json(profiles_dir / f"{cam}.json")
        triple = compute_frame_triple(clip, prof, awb_cfg)
        before_p.write_bytes(to_png_bytes(triple.before))
        after_p.write_bytes(to_png_bytes(triple.full))
        return cam, tuple(float(g) for g in triple.scene_gain), None
    except Exception as e:
        _write_placeholder(before_p, "error")
        _write_placeholder(after_p, "error")
        return cam, None, str(e)


def ensure_previews(
    event: Event,
    profiles_dir: Path,
    pipeline_cfg: dict,
    cache_root: Path,
) -> PreviewResult:
    awb_cfg = pipeline_cfg["awb"] if "awb" in pipeline_cfg else pipeline_cfg
    cache_dir = _cache_dir(event, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    profile_mtimes = _profile_mtimes(profiles_dir)
    cfg_hash = _pipeline_cfg_hash(awb_cfg)

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if _meta_current(meta, profile_mtimes, cfg_hash):
                paths = {
                    cam: {
                        "before": cache_dir / f"{cam}_before.png",
                        "after": cache_dir / f"{cam}_after.png",
                    }
                    for cam in CAMERAS
                }
                if all(p.exists() for cam_paths in paths.values() for p in cam_paths.values()):
                    return PreviewResult(
                        paths=paths,
                        scene_gains={cam: list(meta.get("scene_gains", {}).get(cam, [1.0, 1.0, 1.0])) for cam in CAMERAS},
                        errors={cam: meta.get("errors", {}).get(cam) for cam in CAMERAS},
                    )
        except (OSError, json.JSONDecodeError):
            pass

    scene_gains: dict[str, list[float]] = {}
    errors: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(_render_one_camera, cam, _clip_for_camera(event, cam),
                        profiles_dir, awb_cfg, cache_dir)
            for cam in CAMERAS
        ]
        for fut in as_completed(futs):
            cam, gain, err = fut.result()
            scene_gains[cam] = list(gain) if gain is not None else [1.0, 1.0, 1.0]
            errors[cam] = err

    meta_path.write_text(json.dumps({
        "profile_mtimes": profile_mtimes,
        "pipeline_cfg_hash": cfg_hash,
        "scene_gains": scene_gains,
        "errors": errors,
    }, indent=2))

    return PreviewResult(
        paths={
            cam: {
                "before": cache_dir / f"{cam}_before.png",
                "after": cache_dir / f"{cam}_after.png",
            }
            for cam in CAMERAS
        },
        scene_gains=scene_gains,
        errors=errors,
    )
