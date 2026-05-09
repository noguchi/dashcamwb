"""Generate synthetic Tesla DashCam-style mp4s for testing.

Creates short (3s) 320x240 H.264 mp4s with a known constant color cast applied.
This is enough to test:
- ffmpeg/cv2 frame extraction
- statistical mining recovers the cast
- render pipeline neutralises the cast
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

CAMERAS = [
    "front", "back",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
]

def make_clip(
    out_path: Path,
    cast_rgb: tuple[float, float, float],
    duration_sec: float = 3.0,
    width: int = 320,
    height: int = 240,
    base_gray: int = 180,
) -> None:
    """Generate one mp4 with a near-uniform color (base_gray * cast).

    cast_rgb: per-channel multiplier representing the camera's intrinsic cast
    (e.g. (1.10, 1.00, 0.92) means R is 10% high, B is 8% low).
    """
    r = int(np.clip(base_gray * cast_rgb[0], 0, 255))
    g = int(np.clip(base_gray * cast_rgb[1], 0, 255))
    b = int(np.clip(base_gray * cast_rgb[2], 0, 255))
    color_str = f"0x{r:02x}{g:02x}{b:02x}"
    # ffmpeg color source → libx264 (no VideoToolbox needed for tests)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color_str}:s={width}x{height}:d={duration_sec}:r=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def make_event(
    event_dir: Path,
    casts: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Generate a 6-camera synthetic event under event_dir."""
    casts = casts or {cam: (1.0, 1.0, 1.0) for cam in CAMERAS}
    event_dir.mkdir(parents=True, exist_ok=True)
    timestamp = "2026-05-05_13-49-39"
    for cam in CAMERAS:
        out = event_dir / f"{timestamp}-{cam}.mp4"
        make_clip(out, casts[cam])
    # 注: Tesla 実車の event.json は timezone なし naive ISO format。
    # calibrate.py の `read_event_timestamp` で JST として解釈する前提。
    (event_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T13:49:56","city":"Tokyo",'
        '"street":"","est_lat":"35.68","est_lon":"139.65",'
        '"reason":"sentry_aware_object_detection","camera":"5"}'
    )
