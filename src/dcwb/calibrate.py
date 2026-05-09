from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from datetime import timedelta
from dcwb.profile import Profile, CalibrationMeta
from dcwb.daylight import is_daytime, TOKYO_LAT, TOKYO_LON
from dcwb.ffmpeg_wrap import probe_duration, extract_frame

# Tesla の event.json は naive ISO format (no tz)。実車設定のローカル時刻を保存する想定。
# Tokyo ベースの利用が前提なので JST (UTC+9) として解釈する。
JST = timezone(timedelta(hours=9))

NEUTRAL_V_MIN = 0.7
# 0.25 を採用: 実カメラの ±10% 程度のキャスト (例: R=1.10, B=0.90 →
# S≈0.18) を「ニュートラル候補」として捕捉できるようにするため。
NEUTRAL_S_MAX = 0.25
SAT_MAX = 250
SHADOW_V_MIN = 0.2

def find_neutral_pixels(image_rgb: np.ndarray) -> np.ndarray:
    """Return Nx3 uint8 array of pixels passing the neutral-candidate mask."""
    if image_rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    flat = image_rgb.reshape(-1, 3)
    s_flat = s.reshape(-1)
    v_flat = v.reshape(-1)
    mask = (
        (v_flat > NEUTRAL_V_MIN)
        & (s_flat < NEUTRAL_S_MAX)
        & (v_flat > SHADOW_V_MIN)
        & np.all(flat <= SAT_MAX, axis=1)
    )
    return flat[mask]

def is_multicolor(image_rgb: np.ndarray, threshold: float = 0.05) -> bool:
    """True if the scene exhibits enough chroma diversity.

    全ピクセルが同一彩度 (例: 単色ベタ塗り、または 3 色等帯) でも
    彩度の標準偏差はゼロになり判定不能なため、RGB 各チャンネルの
    画素値ばらつきと、彩度の最大値を併用する。
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    rgb = image_rgb.astype(np.float32) / 255.0
    # 各チャンネルの画素ばらつきの最大値 (色帯があれば高い)
    channel_std_max = float(rgb.std(axis=(0, 1)).max())
    s_std = float(s.std())
    return max(channel_std_max, s_std) > threshold

def geometric_median(points: np.ndarray, eps: float = 1e-5, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld's algorithm for the geometric median of N points in R^d."""
    y = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
        nz = d > eps
        if not np.any(nz):
            return y
        w = 1.0 / d[nz]
        y_new = (points[nz] * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y

def _list_clips_for_camera(source_root: Path, camera: str) -> list[Path]:
    paths: list[Path] = []
    for sub in ("SentryClips", "RecentClips", "SavedClips"):
        root = source_root / sub
        if not root.exists():
            continue
        paths.extend(root.rglob(f"*-{camera}.mp4"))
    return sorted(paths)

def _event_dir_of(clip: Path) -> Path:
    return clip.parent

def read_event_timestamp(event_dir: Path) -> datetime | None:
    """Read event.json["timestamp"]. Naive timestamps are interpreted as JST."""
    ev = event_dir / "event.json"
    if not ev.exists():
        return None
    try:
        d = json.loads(ev.read_text())
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=JST)
        return ts
    except Exception:
        return None

def read_event_latlon(event_dir: Path) -> tuple[float, float]:
    ev = event_dir / "event.json"
    if not ev.exists():
        return TOKYO_LAT, TOKYO_LON
    try:
        d = json.loads(ev.read_text())
        return float(d.get("est_lat") or TOKYO_LAT), float(d.get("est_lon") or TOKYO_LON)
    except Exception:
        return TOKYO_LAT, TOKYO_LON

def calibrate_camera(
    camera: str,
    source_root: Path,
    max_per_event: int = 3,
) -> Profile:
    """Mine neutral pixels across all daytime clips for one camera and build a Profile."""
    clips = _list_clips_for_camera(source_root, camera)
    all_pixels: list[np.ndarray] = []
    events_seen: set[Path] = set()
    for clip in clips:
        ev_dir = _event_dir_of(clip)
        ts = read_event_timestamp(ev_dir)
        lat, lon = read_event_latlon(ev_dir)
        if ts is not None and not is_daytime(ts, lat=lat, lon=lon):
            continue
        try:
            duration = probe_duration(clip)
        except Exception:
            continue
        # 等間隔に最大 max_per_event 枚
        n = max_per_event
        ts_list = [duration * (i + 0.5) / n for i in range(n)]
        for t in ts_list:
            try:
                img = extract_frame(clip, t)
            except Exception:
                continue
            if not is_multicolor(img):
                continue
            pixels = find_neutral_pixels(img)
            if pixels.shape[0] == 0:
                continue
            all_pixels.append(pixels)
            events_seen.add(ev_dir)
    if not all_pixels:
        raise RuntimeError(f"no neutral samples found for camera={camera}")
    stacked = np.concatenate(all_pixels, axis=0).astype(np.float64)
    white_point = geometric_median(stacked)
    meta = CalibrationMeta(
        samples_used=int(stacked.shape[0]),
        events_sampled=len(events_seen),
        method="robust_white_patch_median",
        calibrated_at=datetime.now(timezone.utc),
        samples_per_event_max=max_per_event,
    )
    return Profile.from_white_point(camera=camera, rgb_white=white_point, meta=meta)
