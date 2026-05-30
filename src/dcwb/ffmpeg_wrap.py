from __future__ import annotations
import dataclasses
import functools
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from dcwb.matrix import Matrix3x3

_ENCODER_LINE = re.compile(r"^\s[A-Z.]{6}\s+(\S+)")


@dataclass(frozen=True)
class LookConfig:
    """Creative 'look' grade applied after the neutral WB matrix.

    A gentle S-curve (deepen shadows, lift highlights) + saturation/gamma to
    counter the flat, muted look of Tesla's forensic-tuned footage. tag_bt709
    writes bt709 color metadata so players stop guessing at the untagged stream.
    """
    scurve: str = "0/0 0.25/0.21 0.5/0.5 0.75/0.82 1/1"
    saturation: float = 1.12
    gamma: float = 1.03
    tag_bt709: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "LookConfig":
        data = data or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def filters(self) -> list[str]:
        fs = [
            f"curves=master='{self.scurve}'",
            f"eq=saturation={self.saturation:.4f}:gamma={self.gamma:.4f}",
        ]
        if self.tag_bt709:
            fs.append(_BT709_SETPARAMS)
        return fs


# setparams tags the frames in the filter graph (output -color_* flags don't
# propagate reliably through a -vf chain). Range is left untouched so the
# luminance interpretation matches the untagged default (no level shift).
_BT709_SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"


@functools.lru_cache(maxsize=1)
def _available_encoders() -> frozenset[str]:
    """Names of video/audio encoders this ffmpeg build exposes.

    Returns an empty set if ffmpeg is missing or the probe fails, so callers
    treat "unknown" as "don't second-guess the requested encoder".
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return frozenset()
    names = {m.group(1) for line in result.stdout.splitlines()
             if (m := _ENCODER_LINE.match(line))}
    return frozenset(names)


def resolve_encoder(encoder: str, fallback: str = "libx264") -> str:
    """Return a usable encoder, falling back to libx264 off Apple Silicon.

    The project default is h264_videotoolbox (Apple Silicon). On other
    platforms that encoder is absent, so substitute `fallback` with a warning.
    If the probe is empty (ffmpeg missing/old) keep the request unchanged.
    """
    available = _available_encoders()
    if not available or encoder in available:
        return encoder
    if fallback in available:
        print(
            f"[ffmpeg] encoder '{encoder}' unavailable; falling back to '{fallback}'",
            file=sys.stderr,
        )
        return fallback
    return encoder

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
    so the result may be shorter than `times`. The returned frame is the nearest
    decodable frame to each `t` (seek accuracy depends on keyframe spacing).
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
    encoder = resolve_encoder(encoder)
    cm = _colorchannelmixer(matrix)
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


def _run_ffmpeg(cmd: list[str], tmp: Path) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed: {e.stderr.decode('utf-8', errors='replace')[:500]}"
        ) from e


def _colorchannelmixer(matrix: Matrix3x3) -> str:
    if matrix.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, got {matrix.shape}")
    m = matrix
    return (
        f"colorchannelmixer="
        f"rr={m[0, 0]:.6f}:rg={m[0, 1]:.6f}:rb={m[0, 2]:.6f}:"
        f"gr={m[1, 0]:.6f}:gg={m[1, 1]:.6f}:gb={m[1, 2]:.6f}:"
        f"br={m[2, 0]:.6f}:bg={m[2, 1]:.6f}:bb={m[2, 2]:.6f}"
    )


def cut_clip(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    matrix: Matrix3x3 | None = None,
    look: LookConfig | None = None,
) -> None:
    """Cut [start_sec, start_sec+duration_sec) from src into dst.

    When `matrix` is given, the 3x3 RGB color transform (white balance) is
    applied via colorchannelmixer. When `look` is given, the creative grade
    (S-curve + saturation/gamma) is appended after it. Both run in the same
    single pass (no second re-encode).
    """
    encoder = resolve_encoder(encoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration_sec:.3f}",
        "-an",
    ]
    vf: list[str] = []
    if matrix is not None:
        vf.append(_colorchannelmixer(matrix))
    if look is not None:
        vf += look.filters()
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += [
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]
    _run_ffmpeg(cmd, tmp)
    tmp.replace(dst)


def concat_clips(
    clips: list[Path],
    dst: Path,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    tag_bt709: bool = False,
) -> None:
    if not clips:
        raise ValueError("concat_clips requires at least one clip")
    encoder = resolve_encoder(encoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    # Use the concat *filter*, not the concat demuxer. Real Tesla front clips are
    # VFR and end up with mismatched per-clip timescales (e.g. 18432 vs 7170000);
    # the demuxer can't reconcile those when re-encoding and smears the output PTS
    # into a multi-hour file. The filter regenerates a single uniform timeline.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    streams = "".join(f"[{i}:v]" for i in range(len(clips)))
    # The concat filter does not reliably carry per-input color tags onto the
    # output, so re-tag the joined stream here when requested.
    graph = f"{streams}concat=n={len(clips)}:v=1:a=0[c];[c]{_BT709_SETPARAMS}[outv]" if tag_bt709 \
        else f"{streams}concat=n={len(clips)}:v=1:a=0[outv]"
    cmd += [
        "-filter_complex", graph,
        "-map", "[outv]",
        "-an",
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]
    _run_ffmpeg(cmd, tmp)
    tmp.replace(dst)
