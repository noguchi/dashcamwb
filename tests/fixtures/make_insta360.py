"""Synthetic .insv-like fixtures: a real mp4 with a known creation_time tag,
plus (Task 3) a synthetic IMU trailer appended to it."""
from __future__ import annotations
import struct
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


INSTA360_MAGIC = b"8db42d694ccc418790edff439fe026bf"  # 32 ASCII bytes
_ACC_RANGE = 32.0
_GYRO_RANGE = 2000.0

def append_imu_trailer(insv: Path, samples: list[tuple[float, ...]]) -> None:
    """Append one offsets-index trailer with a gyro record (id=3) to an mp4.
    samples: list of (t_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
    (accel in g, gyro in deg/s — matching the camera's raw layout)."""
    acc_scale = 32768.0 / _ACC_RANGE
    gyro_scale = 32768.0 / _GYRO_RANGE
    body = b""
    for t_s, ax, ay, az, gx, gy, gz in samples:
        counts = [
            int(round(ax * acc_scale)) + 32768, int(round(ay * acc_scale)) + 32768,
            int(round(az * acc_scale)) + 32768, int(round(gx * gyro_scale)) + 32768,
            int(round(gy * gyro_scale)) + 32768, int(round(gz * gyro_scale)) + 32768,
        ]
        body += struct.pack("<Q6H", int(round(t_s * 1_000_000)), *counts)
    # offsets table: one entry for the gyro record (id=3) at offset 0
    table = struct.pack("<BBII", 3, 0, len(body), 0)
    # 6-byte offsets-record trailer: pad(1) | first_id=0 (u8) | table_size (u32 LE)
    offsets_tail = b"\x00" + b"\x00" + struct.pack("<I", len(table))
    trailer = body + table + offsets_tail
    extra_size = len(trailer) + 72  # +72 header
    header = b"\x00" * 32 + struct.pack("<II", extra_size, 3) + INSTA360_MAGIC
    with open(insv, "ab") as fp:
        fp.write(trailer + header)
