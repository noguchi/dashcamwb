"""Synthetic .insv-like fixtures: a real mp4 with a known creation_time tag,
plus (Task 3) a synthetic IMU trailer appended to it."""
from __future__ import annotations
import subprocess
from pathlib import Path

def make_insv_header(out_path: Path, creation_utc: str = "2026-05-27T08:17:57Z") -> None:
    """Write a tiny real mp4 carrying creation_time=<creation_utc> (UTC)."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        "-metadata", f"creation_time={creation_utc}",
        "-f", "mp4",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
