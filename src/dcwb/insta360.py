from __future__ import annotations
import json
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

JST = timezone(timedelta(hours=9))

def read_creation_time(insv: Path) -> datetime:
    """Return the mp4 header creation_time as a tz-aware UTC datetime."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "json", str(insv)],
        check=True, capture_output=True, text=True,
    ).stdout
    tag = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
    if not tag:
        raise ValueError(f"no creation_time in {insv}")
    return datetime.fromisoformat(tag.replace("Z", "+00:00")).astimezone(timezone.utc)

def to_jst(dt: datetime) -> datetime:
    return dt.astimezone(JST)


# ── IMU trailer parser (Task 3) ──────────────────────────────────────────────

_MAGIC = b"8db42d694ccc418790edff439fe026bf"
_HEADER_SIZE = 72  # padding[32] + extra_size:u32 + version:u32 + magic[32]
_OFFSETS_ENTRY = 10
_IMU_RECORD_ID = 3
_METADATA_RECORD_ID = 1
_ITEM = 20  # u64 ts + 6*u16
_G_TO_MS2 = 9.80665
_DEG_TO_RAD = np.pi / 180.0
_DEFAULT_ACC_RANGE = 32.0
_DEFAULT_GYRO_RANGE = 2000.0


@dataclass(frozen=True)
class ImuSample:
    t_s: float
    accel: tuple  # (x, y, z) m/s^2
    gyro: tuple   # (x, y, z) rad/s


def _read_at(fp, start, length):
    fp.seek(start)
    return fp.read(length)


def _read_offsets_table(fp, file_size):
    off = _HEADER_SIZE + 4 + 1 + 1  # 78
    first_id = _read_at(fp, file_size - (off - 1), 1)[0]
    if first_id != 0:
        return None
    tbl_size = struct.unpack("<I", _read_at(fp, file_size - (off - 1) + 1, 4))[0]
    body = _read_at(fp, file_size - off - tbl_size, tbl_size)
    offsets = {}
    pos = 0
    while pos + _OFFSETS_ENTRY <= len(body):
        rid, _fmt = body[pos], body[pos + 1]
        size, offset = struct.unpack_from("<II", body, pos + 2)
        if rid > 0:
            offsets[rid] = (offset, size)
        pos += _OFFSETS_ENTRY
    return offsets


def _pb_walk(buf):
    i, out = 0, []
    while i < len(buf):
        tag = buf[i]; i += 1
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            v = s = 0
            while True:
                b = buf[i]; i += 1
                v |= (b & 0x7F) << s; s += 7
                if not b & 0x80:
                    break
            out.append((fno, wt, v))
        elif wt == 2:
            ln = s = 0
            while True:
                b = buf[i]; i += 1
                ln |= (b & 0x7F) << s; s += 7
                if not b & 0x80:
                    break
            out.append((fno, wt, buf[i:i + ln])); i += ln
        elif wt == 5:
            out.append((fno, wt, buf[i:i + 4])); i += 4
        elif wt == 1:
            out.append((fno, wt, buf[i:i + 8])); i += 8
        else:
            break
    return out


def _read_ranges(fp, extra_start, offsets):
    acc_range = gyro_range = None
    if _METADATA_RECORD_ID in offsets:
        m_off, m_size = offsets[_METADATA_RECORD_ID]
        md = _read_at(fp, extra_start + m_off, m_size)
        for fno, wt, v in _pb_walk(md):
            if fno == 65 and wt == 2:
                for f2, w2, v2 in _pb_walk(v):
                    if f2 == 1 and w2 == 0:
                        acc_range = float(v2)
                    elif f2 == 2 and w2 == 0:
                        gyro_range = float(v2)
    return acc_range or _DEFAULT_ACC_RANGE, gyro_range or _DEFAULT_GYRO_RANGE


def read_imu(insv: Path) -> list[ImuSample]:
    """Read the IMU time series (accel m/s^2, gyro rad/s) from the .insv trailer.
    Reads only the trailer tail, never the video body. Raises ValueError if the
    Insta360 trailer / gyro record is absent or malformed."""
    file_size = insv.stat().st_size
    with open(insv, "rb") as fp:
        header = _read_at(fp, file_size - _HEADER_SIZE, _HEADER_SIZE)
        if header[-32:] != _MAGIC:
            raise ValueError(f"no Insta360 trailer magic in {insv}")
        extra_size = struct.unpack_from("<I", header, 32)[0]
        extra_start = file_size - extra_size
        offsets = _read_offsets_table(fp, file_size)
        if offsets is None:
            raise ValueError("trailer not offsets-index layout (first_id != 0)")
        if _IMU_RECORD_ID not in offsets:
            raise ValueError(f"no IMU record (id=3); have ids {sorted(offsets)}")
        try:
            acc_range, gyro_range = _read_ranges(fp, extra_start, offsets)
        except Exception:
            acc_range, gyro_range = _DEFAULT_ACC_RANGE, _DEFAULT_GYRO_RANGE
        g_off, g_size = offsets[_IMU_RECORD_ID]
        body = _read_at(fp, extra_start + g_off, g_size)
    if g_size % _ITEM != 0 or g_size == 0:
        raise ValueError(f"gyro body {g_size} not a positive multiple of {_ITEM}")
    n = g_size // _ITEM
    a = np.frombuffer(body, dtype=np.uint8).reshape(n, _ITEM)
    ts = a[:, 0:8].copy().view("<u8").ravel().astype(np.float64) / 1e6
    vals = a[:, 8:20].copy().view("<u2").reshape(n, 6).astype(np.float64) - 32768.0
    accel = vals[:, 0:3] / (32768.0 / acc_range) * _G_TO_MS2
    gyro = vals[:, 3:6] / (32768.0 / gyro_range) * _DEG_TO_RAD
    return [ImuSample(float(ts[i]),
                      (float(accel[i, 0]), float(accel[i, 1]), float(accel[i, 2])),
                      (float(gyro[i, 0]), float(gyro[i, 1]), float(gyro[i, 2])))
            for i in range(n)]
