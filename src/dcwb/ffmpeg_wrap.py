from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2
from dcwb.matrix import Matrix3x3

def probe_duration(path: Path) -> float:
    """Return clip duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def extract_frame(path: Path, t: float) -> np.ndarray:
    """Decode one frame at timestamp t (seconds) and return RGB uint8 (H, W, 3)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to read frame at t={t} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

def extract_frames(path: Path, times: list[float]) -> list[np.ndarray]:
    """Decode frames at the given timestamps (seconds) in one capture session.

    Returns RGB uint8 (H, W, 3) frames. Frames that fail to decode are skipped,
    so the result may be shorter than `times`.
    """
    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


def render_with_matrix(
    src: Path,
    dst: Path,
    matrix: Matrix3x3,
    bitrate_kbps: int = 12000,
    encoder: str = "h264_videotoolbox",
) -> None:
    """Render src → dst with a 3x3 RGB color transform applied via colorchannelmixer.

    On non-Apple-Silicon systems pass encoder='libx264' explicitly.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, got {matrix.shape}")
    m = matrix
    cm = (
        f"colorchannelmixer="
        f"rr={m[0, 0]:.6f}:rg={m[0, 1]:.6f}:rb={m[0, 2]:.6f}:"
        f"gr={m[1, 0]:.6f}:gg={m[1, 1]:.6f}:gb={m[1, 2]:.6f}:"
        f"br={m[2, 0]:.6f}:bg={m[2, 1]:.6f}:bb={m[2, 2]:.6f}"
    )
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", cm,
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(dst)
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed: {e.stderr.decode('utf-8', errors='replace')[:500]}"
        ) from e
