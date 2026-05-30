# Insta360 × Tesla Frame-Accurate Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `dcwb sync-insta360` CLI subcommand that frame-accurately time-aligns an Insta360 ride-view recording with Tesla dashcam front clips, then emits a side-by-side combined mp4 (with Tesla telemetry burned in) and a local web player for synchronized playback.

**Architecture:** A hybrid sync pipeline: (1) coarse timestamp anchor from the Insta360 mp4 `creation_time` vs Tesla filename clock; (2) fine cross-correlation of a shared 1-D motion signal (yaw-rate primary, longitudinal accel fallback) where Tesla data comes from per-frame SEI and Insta360 data comes from the IMU stored in the file's proprietary trailer; (3) manual nudge in the web player. Heavy 18 GB `.insv` files are touched only via trailer `seek` (no full reads). Sync math is factored into pure functions for TDD; I/O is thin.

**Tech Stack:** Python 3.11+, uv, numpy, ffmpeg/ffprobe, Flask (existing `serve/`), pytest. Reuses vendored Tesla `dashcam` SEI extractor; adds a vendored/ported Insta360 trailer IMU reader (gyroflow `telemetry-parser` lineage).

---

## File Structure

- `src/dcwb/insta360.py` (new) — Insta360 `.insv` reader: `read_creation_time()`, `read_imu()`, the `ImuSample` dataclass, and the trailer parser. One responsibility: turn an `.insv` into absolute start time + IMU time series.
- `src/dcwb/sync.py` (new) — sync math + orchestration: pure signal functions (`unwrap_yaw_rate`, `resample_uniform`, `normalized_xcorr`), Tesla per-frame series builder, Tesla front-clip region selection, the `compute_offset()` orchestrator, the telemetry-ASS generator, and the `sync.json` manifest writer.
- `src/dcwb/telemetry.py` (modify) — add `iter_segment_frames()` yielding per-frame SEI fields (the existing code only returns a per-segment summary).
- `src/dcwb/ffmpeg_wrap.py` (modify) — add `render_sidebyside()` (hstack two inputs with per-input time offset + burned ASS subtitles) and `reframe_insv()` (dual-fisheye → flat via `v360`, optional fallback path).
- `src/dcwb/serve/` (modify) — add the sync-player route, JSON-data route, nudge-writeback route, and template.
- `src/dcwb/cli.py` (modify) — wire the `sync-insta360` subcommand.
- `src/dcwb/vendor/insta360/` (new) — NOTICE documenting the trailer format lineage (gyroflow).
- `tests/test_insta360.py`, `tests/test_sync.py`, `tests/test_sync_render.py`, `tests/test_cli.py` (modify), `tests/test_serve_sync.py` (new) — tests.
- `tests/fixtures/make_insta360.py` (new) — synthetic `.insv`-like fixture builders (creation_time mp4 + synthetic IMU trailer).

---

## Task 1: Insta360 IMU extraction spike (DE-RISK — decision gate)

This is the single biggest unknown. Prove a usable gyro/accel time series can be pulled from the real `.insv` trailer **before** building the pipeline on top of it. This task is investigative; the formal TDD parser is Task 3.

**Files:**
- Create: `src/dcwb/vendor/insta360/NOTICE`
- Scratch: `/tmp/insta360_spike.py` (not committed)

- [ ] **Step 1: Confirm the trailer location on the real file**

Run:
```bash
F="/mnt/sentryusb/Insta360/VID_20260527_171757_00_007_009-オリジナル/VID_20260527_171757_00_007.insv"
exiftool -ee -a -G1 "$F" 2>&1 | grep -i "trailer"
```
Expected: a line like `Insta360 trailer at offset 0x... (NNN bytes)`. Record the offset and size — the trailer is the last `NNN` bytes of the file.

- [ ] **Step 2: Study the authoritative format**

The Insta360 trailer is read **from the end of the file backwards**. The canonical reference implementation is gyroflow's `telemetry-parser` (`src/insta360/mod.rs`). Format summary to port:
- The final 32 bytes are a fixed magic. Just before the magic is a small footer giving the total extra-metadata size; from there a sequence of typed records is walked **backwards** (each record header carries an id, a size, and a version; you seek back by `size` to reach the next record).
- Records are keyed by id; the gyro/accel record packs a stream of samples. Each sample is a timestamp plus 3 accelerometer and 3 gyroscope components. Exposure/other records are present too and are skipped.

Write `/tmp/insta360_spike.py` that opens the file, seeks to `filesize - trailer_size`, walks the records backward, and decodes the IMU record into a list of `(t_s, ax, ay, az, gx, gy, gz)`.

- [ ] **Step 3: Validate plausibility against physics**

Run: `uv run python /tmp/insta360_spike.py "$F"`
Expected, asserted by the script and printed:
- sample count / duration ≈ a steady IMU rate (Insta360 IMU is typically a few hundred Hz; print the inferred rate).
- mean accel magnitude `sqrt(ax²+ay²+az²)` over the clip ≈ 9.8 (gravity) within ±2 (units may be g — if mean ≈ 1.0, the unit is g; record which).
- gyro values are small at rest and spike during the known turns of the drive.
Print: inferred rate, accel-magnitude mean, gyro units guess.

- [ ] **Step 4: Decision gate**

If Step 3 passes → proceed to Task 2; the record format is now known and Task 3 formalizes it.
If it fails (format unreadable) → **STOP and report**. The fallback design (spec §"リスクとフォールバック") is: skip Task 4/6's correlation, ship anchor + manual nudge only. Mark Tasks 4 and 6 as "degraded mode" and continue from Task 2.

- [ ] **Step 5: Commit the NOTICE**

Write `src/dcwb/vendor/insta360/NOTICE`:
```
Insta360 .insv trailer IMU format.
Reverse-engineered by the gyroflow project (telemetry-parser, src/insta360/),
GPL/MIT-dual licensed. This module is an independent Python port of the trailer
record-walking and IMU sample layout. See:
https://github.com/gyroflow/telemetry-parser
```
```bash
git add src/dcwb/vendor/insta360/NOTICE
git commit -m "docs(insta360): vendor NOTICE for trailer IMU format (gyroflow lineage)"
```

---

## Task 2: Insta360 creation_time reader

**Files:**
- Create: `src/dcwb/insta360.py`
- Create: `tests/fixtures/make_insta360.py`
- Test: `tests/test_insta360.py`

- [ ] **Step 1: Write the synthetic-mp4 fixture helper**

Create `tests/fixtures/make_insta360.py`:
```python
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
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_insta360.py`:
```python
from datetime import timezone
from pathlib import Path
from dcwb.insta360 import read_creation_time, to_jst
from tests.fixtures.make_insta360 import make_insv_header

def test_read_creation_time_returns_utc(tmp_path: Path):
    f = tmp_path / "VID.insv"
    make_insv_header(f, "2026-05-27T08:17:57Z")
    dt = read_creation_time(f)
    assert dt.tzinfo is not None
    assert dt.astimezone(timezone.utc).isoformat().startswith("2026-05-27T08:17:57")

def test_to_jst_adds_nine_hours(tmp_path: Path):
    f = tmp_path / "VID.insv"
    make_insv_header(f, "2026-05-27T08:17:57Z")
    jst = to_jst(read_creation_time(f))
    assert jst.hour == 17 and jst.minute == 17 and jst.second == 57
```

- [ ] **Step 2b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_insta360.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dcwb.insta360'`.

- [ ] **Step 3: Implement**

Create `src/dcwb/insta360.py`:
```python
from __future__ import annotations
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_insta360.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/insta360.py tests/test_insta360.py tests/fixtures/make_insta360.py
git commit -m "feat(insta360): read .insv creation_time as UTC + JST helper"
```

---

## Task 3: Insta360 IMU trailer parser

Formalizes the Task 1 spike. The unit test builds a synthetic trailer with the **same packing** the parser reads (roundtrip), so it is self-consistent; a separate slow test validates against the real file when present.

**Files:**
- Modify: `src/dcwb/insta360.py`
- Modify: `tests/fixtures/make_insta360.py`
- Test: `tests/test_insta360.py`

- [ ] **Step 1: Add the trailer writer to the fixture helper**

Append to `tests/fixtures/make_insta360.py` a writer that mirrors the record layout pinned down in Task 1. Using the layout from the spike (record = `[payload][le_u32 size][le_u16 id][le_u8 version]`, walked backward from a 32-byte magic; IMU payload = N samples of 7 little-endian float64: `t,ax,ay,az,gx,gy,gz`):
```python
import struct

INSTA360_MAGIC = bytes.fromhex("8db42d694ccc418790edff439fe026bf")
IMU_RECORD_ID = 0x0300  # confirm/replace with the id found in Task 1

def append_imu_trailer(insv: Path, samples: list[tuple[float, ...]]) -> None:
    """Append one IMU record + magic to an existing mp4 so read_imu() finds it.
    samples: list of (t, ax, ay, az, gx, gy, gz)."""
    payload = b"".join(struct.pack("<7d", *s) for s in samples)
    record = payload + struct.pack("<IHB", len(payload), IMU_RECORD_ID, 1)
    with open(insv, "ab") as fp:
        fp.write(record)
        fp.write(INSTA360_MAGIC)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_insta360.py`:
```python
import math
import pytest
from dcwb.insta360 import read_imu, ImuSample
from tests.fixtures.make_insta360 import make_insv_header, append_imu_trailer

def test_read_imu_roundtrip(tmp_path):
    f = tmp_path / "VID.insv"
    make_insv_header(f)
    samples = [(i / 200.0, 0.0, 0.0, 9.8, 0.0, 0.0, float(i % 3)) for i in range(50)]
    append_imu_trailer(f, samples)
    series = read_imu(f)
    assert len(series) == 50
    assert isinstance(series[0], ImuSample)
    assert math.isclose(series[10].t_s, 10 / 200.0, abs_tol=1e-9)
    assert math.isclose(series[0].accel[2], 9.8, abs_tol=1e-6)

def test_read_imu_no_trailer_raises(tmp_path):
    f = tmp_path / "VID.insv"
    make_insv_header(f)
    with pytest.raises(ValueError):
        read_imu(f)
```

- [ ] **Step 2b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_insta360.py -k imu -v`
Expected: FAIL — `ImportError: cannot import name 'read_imu'`.

- [ ] **Step 3: Implement the parser**

Append to `src/dcwb/insta360.py`:
```python
import struct
from dataclasses import dataclass

_MAGIC = bytes.fromhex("8db42d694ccc418790edff439fe026bf")
_IMU_RECORD_ID = 0x0300  # from Task 1

@dataclass(frozen=True)
class ImuSample:
    t_s: float
    accel: tuple[float, float, float]  # m/s^2 (or g — see Task 1 note)
    gyro: tuple[float, float, float]   # rad/s

def _iter_trailer_records(data: bytes):
    """Yield (record_id, payload) walking the trailer backward from the magic."""
    if data[-32:] != _MAGIC:
        raise ValueError("no Insta360 trailer magic")
    pos = len(data) - 32
    while pos > 7:
        size, rid, _ver = struct.unpack("<IHB", data[pos - 7:pos])
        payload_end = pos - 7
        payload_start = payload_end - size
        if payload_start < 0:
            break
        yield rid, data[payload_start:payload_end]
        pos = payload_start
        if rid == _IMU_RECORD_ID:
            break  # IMU is the only record we need

def read_imu(insv: Path) -> list[ImuSample]:
    """Read the IMU time series from the .insv trailer (trailer-only seek)."""
    size = insv.stat().st_size
    # The trailer is small (~tens of MB). Read a bounded tail, not the whole file.
    tail = min(size, 256 * 1024 * 1024)
    with open(insv, "rb") as fp:
        fp.seek(size - tail)
        data = fp.read()
    for rid, payload in _iter_trailer_records(data):
        if rid != _IMU_RECORD_ID:
            continue
        n = len(payload) // struct.calcsize("<7d")
        out = []
        for i in range(n):
            t, ax, ay, az, gx, gy, gz = struct.unpack_from("<7d", payload, i * 56)
            out.append(ImuSample(t, (ax, ay, az), (gx, gy, gz)))
        return out
    raise ValueError(f"no IMU record in {insv}")
```
> Note: if Task 1 found a different header layout / sample dtype, update both the fixture writer (Step 1) and `_iter_trailer_records`/`read_imu` together so the roundtrip stays consistent.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_insta360.py -v`
Expected: PASS (all).

- [ ] **Step 5: Add a slow real-file integration test**

Append to `tests/test_insta360.py`:
```python
REAL_INSV = Path("/mnt/sentryusb/Insta360/VID_20260527_171757_00_007_009-オリジナル/"
                 "VID_20260527_171757_00_007.insv")

@pytest.mark.skipif(not REAL_INSV.exists(), reason="real .insv not mounted")
def test_read_imu_real_file_plausible():
    series = read_imu(REAL_INSV)
    assert len(series) > 1000
    mags = [math.sqrt(sum(c * c for c in s.accel)) for s in series[:500]]
    mean_mag = sum(mags) / len(mags)
    assert 8.0 < mean_mag < 11.5 or 0.8 < mean_mag < 1.2  # m/s^2 or g
```

- [ ] **Step 6: Run + commit**

Run: `uv run --extra dev pytest tests/test_insta360.py -v`
Expected: PASS (real-file test PASS if mounted, else SKIP).
```bash
git add src/dcwb/insta360.py tests/test_insta360.py tests/fixtures/make_insta360.py
git commit -m "feat(insta360): parse IMU time series from .insv trailer (gyroflow port)"
```

---

## Task 4: Pure sync-signal functions

**Files:**
- Create: `src/dcwb/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sync.py`:
```python
import numpy as np
from dcwb.sync import unwrap_yaw_rate, resample_uniform, normalized_xcorr

def test_unwrap_yaw_rate_handles_360_wrap():
    # heading crosses 360->0; constant 10 deg/s at 1 Hz
    headings = np.array([350.0, 0.0, 10.0, 20.0])  # +10 deg each step
    rate = unwrap_yaw_rate(headings, dt=1.0)
    assert np.allclose(rate, 10.0, atol=1e-6)

def test_resample_uniform_linear():
    t = np.array([0.0, 1.0, 3.0])
    v = np.array([0.0, 10.0, 30.0])
    grid_t, grid_v = resample_uniform(t, v, rate_hz=1.0)
    assert np.allclose(grid_t, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(grid_v, [0.0, 10.0, 20.0, 30.0])

def test_normalized_xcorr_recovers_known_lag():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(500)
    lag = 37
    a = base
    b = np.concatenate([np.zeros(lag), base])[:500]  # b is base delayed by 37
    best_lag, peak = normalized_xcorr(a, b, max_lag=100)
    assert best_lag == lag
    assert peak > 0.9
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dcwb.sync'`.

- [ ] **Step 2: Implement**

Create `src/dcwb/sync.py`:
```python
from __future__ import annotations
import numpy as np

def unwrap_yaw_rate(headings_deg: np.ndarray, dt: float) -> np.ndarray:
    """deg/s yaw rate from a heading series, robust to 0/360 wraparound."""
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, float))))
    return np.gradient(unwrapped, dt)

def resample_uniform(t: np.ndarray, v: np.ndarray, rate_hz: float):
    """Linearly resample irregular (t, v) onto a uniform grid at rate_hz."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    grid_t = np.arange(t[0], t[-1] + 1e-9, 1.0 / rate_hz)
    grid_v = np.interp(grid_t, t, v)
    return grid_t, grid_v

def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Return (best_lag, peak) maximizing normalized cross-correlation of a vs b
    over integer lags in [-max_lag, max_lag]. Positive lag means b lags a."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    best_lag, best = 0, -np.inf
    n = min(len(a), len(b))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:n - lag], b[lag:n]
        else:
            x, y = a[-lag:n], b[:n + lag]
        if len(x) < 2:
            continue
        score = float(np.dot(x, y) / len(x))
        if score > best:
            best, best_lag = score, lag
    return best_lag, best
```

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync.py -v`
Expected: PASS (3 passed).

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/sync.py tests/test_sync.py
git commit -m "feat(sync): pure yaw-rate / resample / cross-correlation helpers"
```

---

## Task 5: Tesla per-frame SEI series

**Files:**
- Modify: `src/dcwb/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_telemetry.py`:
```python
from pathlib import Path
from dcwb.telemetry import iter_segment_frames, FrameTelemetry

REAL_FRONT = Path("/mnt/sentryusb/RecentClips/2026-05-27/2026-05-27_17-18-05-front.mp4")

import pytest
@pytest.mark.skipif(not REAL_FRONT.exists(), reason="real front clip not mounted")
def test_iter_segment_frames_yields_per_frame_fields():
    frames = list(iter_segment_frames(REAL_FRONT))
    assert len(frames) > 10
    f = frames[0]
    assert isinstance(f, FrameTelemetry)
    assert hasattr(f, "heading_deg") and hasattr(f, "accel_x") and hasattr(f, "speed_mps")
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_telemetry.py -k segment_frames -v`
Expected: FAIL — `ImportError: cannot import name 'iter_segment_frames'` (SKIP if not mounted — then verify import error only by running a quick `uv run python -c "from dcwb.telemetry import iter_segment_frames"` which must raise ImportError).

- [ ] **Step 2: Implement**

Append to `src/dcwb/telemetry.py`:
```python
@dataclass
class FrameTelemetry:
    frame_index: int
    gear: str
    speed_mps: float
    heading_deg: float
    steering_deg: float
    accel_x: float
    lat: float
    lon: float

def iter_segment_frames(front_clip: Path):
    """Yield FrameTelemetry per SEI frame, in capture order. Fail-safe: a
    read/parse error yields nothing (caller falls back)."""
    try:
        with open(front_clip, "rb") as fp:
            offset, size = _sx.find_mdat(fp)
            for i, meta in enumerate(_sx.iter_sei_messages(fp, offset, size)):
                yield FrameTelemetry(
                    frame_index=i,
                    gear=_GEAR_NAME.get(meta.gear_state, str(meta.gear_state)),
                    speed_mps=float(meta.vehicle_speed_mps or 0.0),
                    heading_deg=float(meta.heading_deg or 0.0),
                    steering_deg=float(meta.steering_wheel_angle or 0.0),
                    accel_x=float(meta.linear_acceleration_mps2_x or 0.0),
                    lat=float(meta.latitude_deg or 0.0),
                    lon=float(meta.longitude_deg or 0.0),
                )
    except Exception:
        return
```

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_telemetry.py -v`
Expected: PASS (new test PASS if mounted, else SKIP); confirm `uv run python -c "from dcwb.telemetry import iter_segment_frames, FrameTelemetry"` exits 0.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): per-frame SEI iterator (heading/accel/steer/speed)"
```

---

## Task 6: Tesla series builder + offset orchestrator

**Files:**
- Modify: `src/dcwb/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test (offset orchestration, synthetic)**

Append to `tests/test_sync.py`:
```python
from dcwb.sync import compute_offset, MotionSeries, SyncResult

def test_compute_offset_recovers_injected_delta():
    # shared yaw-rate signal sampled at 50 Hz over 20 s
    rate = 50.0
    t = np.arange(0, 20, 1 / rate)
    sig = np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    tesla = MotionSeries(t=t, yaw_rate=sig, accel_x=np.zeros_like(sig))
    # insta starts 2.0 s later in absolute time and carries the same motion
    delta_true = 2.0
    insta = MotionSeries(t=t + delta_true, yaw_rate=sig, accel_x=np.zeros_like(sig))
    res = compute_offset(tesla, insta, anchor_guess=1.5, window_s=5.0, rate_hz=rate)
    assert isinstance(res, SyncResult)
    assert abs(res.delta_s - delta_true) < 0.05
    assert res.signal == "yaw_rate"
    assert res.confidence > 0.8
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync.py -k compute_offset -v`
Expected: FAIL — `ImportError: cannot import name 'compute_offset'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/sync.py`:
```python
from dataclasses import dataclass

@dataclass
class MotionSeries:
    t: np.ndarray        # absolute seconds (JST epoch-ish); monotonic
    yaw_rate: np.ndarray
    accel_x: np.ndarray

@dataclass
class SyncResult:
    delta_s: float       # add to insta absolute time to align onto tesla time
    confidence: float    # cross-correlation peak [-1, 1]
    signal: str          # "yaw_rate" or "accel_x"
    anchor_guess: float

def _corr_for(field_a, field_b, rate_hz, window_s, anchor_guess):
    # resample both onto a common uniform grid, then correlate within window
    ta = np.arange(0, len(field_a)) / rate_hz
    tb = np.arange(0, len(field_b)) / rate_hz
    _, a = resample_uniform(ta, field_a, rate_hz)
    _, b = resample_uniform(tb, field_b, rate_hz)
    max_lag = int((window_s + abs(anchor_guess)) * rate_hz)
    lag, peak = normalized_xcorr(a, b, max_lag=max_lag)
    return lag / rate_hz, peak

def compute_offset(tesla: MotionSeries, insta: MotionSeries,
                   anchor_guess: float, window_s: float, rate_hz: float) -> SyncResult:
    """Refine the coarse anchor to a fine delta via cross-correlation.
    Tries yaw-rate first; falls back to accel_x if its peak is weak."""
    # resample tesla/insta motion onto uniform grids at rate_hz first
    _, tya = resample_uniform(tesla.t - tesla.t[0], tesla.yaw_rate, rate_hz)
    _, iya = resample_uniform(insta.t - insta.t[0], insta.yaw_rate, rate_hz)
    base = (insta.t[0] - tesla.t[0])
    max_lag = int((window_s + abs(anchor_guess)) * rate_hz)
    lag_y, peak_y = normalized_xcorr(tya, iya, max_lag=max_lag)
    _, tax = resample_uniform(tesla.t - tesla.t[0], tesla.accel_x, rate_hz)
    _, iax = resample_uniform(insta.t - insta.t[0], insta.accel_x, rate_hz)
    lag_a, peak_a = normalized_xcorr(tax, iax, max_lag=max_lag)
    if peak_y >= peak_a:
        return SyncResult(base + lag_y / rate_hz, peak_y, "yaw_rate", anchor_guess)
    return SyncResult(base + lag_a / rate_hz, peak_a, "accel_x", anchor_guess)
```
> `delta_s` is defined so that `tesla_time ≈ insta_absolute_time + delta_s` is **not** quite right — keep the convention explicit: the test aligns insta (started `delta_true` later) back onto tesla, and `base + lag/rate` yields `delta_true`. Verify the sign against the passing test; if the test fails on sign, negate `base`/`lag` consistently and re-run.

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync.py -v`
Expected: PASS. If only the sign assertion fails, flip the sign convention in `compute_offset` (and document it in the docstring) until `abs(res.delta_s - 2.0) < 0.05`.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/sync.py tests/test_sync.py
git commit -m "feat(sync): compute_offset orchestrator (yaw-rate primary, accel fallback)"
```

---

## Task 7: Tesla front-clip region selection

**Files:**
- Modify: `src/dcwb/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py`:
```python
from datetime import datetime, timedelta, timezone
from dcwb.sync import select_front_clips
JST = timezone(timedelta(hours=9))

def test_select_front_clips_overlapping_window(tmp_path):
    names = ["2026-05-27_17-16-48-front.mp4", "2026-05-27_17-17-04-front.mp4",
             "2026-05-27_17-18-05-front.mp4", "2026-05-27_17-30-07-front.mp4"]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    start = datetime(2026, 5, 27, 17, 17, 57, tzinfo=JST)
    end = start + timedelta(seconds=70)
    chosen = select_front_clips(tmp_path, start, end, seg_seconds=60.0)
    got = [p.name for p in chosen]
    assert got == ["2026-05-27_17-17-04-front.mp4", "2026-05-27_17-18-05-front.mp4"]
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync.py -k select_front -v`
Expected: FAIL — `ImportError: cannot import name 'select_front_clips'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/sync.py`:
```python
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
_FRONT_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-front\.mp4$")

def _front_start(name: str):
    m = _FRONT_RE.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=JST)

def select_front_clips(day_dir: Path, start: datetime, end: datetime,
                       seg_seconds: float = 60.0) -> list[Path]:
    """Front clips whose [clip_start, clip_start+seg] overlaps [start, end]."""
    out = []
    for p in sorted(day_dir.glob("*-front.mp4")):
        cs = _front_start(p.name)
        if cs is None:
            continue
        ce = cs + timedelta(seconds=seg_seconds)
        if ce > start and cs < end:
            out.append(p)
    return out
```

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync.py -k select_front -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/sync.py tests/test_sync.py
git commit -m "feat(sync): select Tesla front clips overlapping the insta window"
```

---

## Task 8: Telemetry ASS-subtitle generator

**Files:**
- Modify: `src/dcwb/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py`:
```python
from dcwb.sync import telemetry_ass

def test_telemetry_ass_emits_timed_dialogue():
    rows = [(0.0, 13.3, 12.0, "DRIVE"), (1.0, 14.0, -5.0, "DRIVE")]  # t, mps, steer, gear
    ass = telemetry_ass(rows, play_w=2560, play_h=1080)
    assert "[Script Info]" in ass
    assert ass.count("Dialogue:") == 2
    assert "48 km/h" in ass            # 13.3 m/s -> 48 km/h
    assert "DRIVE" in ass
    assert "0:00:00.00" in ass         # first event start time
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync.py -k telemetry_ass -v`
Expected: FAIL — `ImportError: cannot import name 'telemetry_ass'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/sync.py`:
```python
def _ass_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def telemetry_ass(rows, play_w: int, play_h: int) -> str:
    """Build an ASS subtitle (speed/steer/gear) with one event per row.
    rows: list of (t_seconds, speed_mps, steering_deg, gear)."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, "
        "Bold, Alignment, MarginL, MarginR, MarginV, BorderStyle, Outline, Shadow\n"
        "Style: tele,DejaVu Sans Mono,42,&H00FFFFFF,&H80000000,1,1,40,40,40,3,2,0\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
    )
    lines = []
    for i, (t, mps, steer, gear) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 1.0
        kmh = round(mps * 3.6)
        text = f"{kmh} km/h   steer {steer:+.0f}\\N{gear}"
        lines.append(f"Dialogue: 0,{_ass_ts(t)},{_ass_ts(end)},tele,{text}")
    return header + "\n".join(lines) + "\n"
```

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync.py -k telemetry_ass -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/sync.py tests/test_sync.py
git commit -m "feat(sync): generate timed telemetry ASS overlay (speed/steer/gear)"
```

---

## Task 9: Side-by-side combined render

**Files:**
- Modify: `src/dcwb/ffmpeg_wrap.py`
- Test: `tests/test_sync_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sync_render.py`:
```python
from pathlib import Path
from dcwb.ffmpeg_wrap import render_sidebyside, probe_duration
from dcwb.sync import telemetry_ass
from tests.fixtures.make_synthetic import make_motion_clip

def _probe_dims(p: Path):
    import subprocess, json
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(p)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]

def test_render_sidebyside_hstacks_with_subs(tmp_path):
    left = tmp_path / "insta.mp4"; right = tmp_path / "tesla.mp4"
    make_motion_clip(left, duration_sec=4.0, width=320, height=240)
    make_motion_clip(right, duration_sec=4.0, width=320, height=240)
    ass = tmp_path / "tele.ass"
    ass.write_text(telemetry_ass([(0.0, 10.0, 0.0, "DRIVE")], play_w=640, play_h=240))
    dst = tmp_path / "combined.mp4"
    render_sidebyside(left, right, dst, left_start=0.5, right_start=0.0,
                      duration=3.0, ass_path=ass, encoder="libx264", bitrate_kbps=2000)
    assert dst.exists()
    w, h = _probe_dims(dst)
    assert w == 640 and h == 240            # two 320-wide panels hstacked
    assert 2.5 < probe_duration(dst) < 3.5
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync_render.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_sidebyside'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/ffmpeg_wrap.py`:
```python
def render_sidebyside(
    left: Path, right: Path, dst: Path,
    left_start: float, right_start: float, duration: float,
    ass_path: Path | None = None,
    encoder: str = "h264_videotoolbox", bitrate_kbps: int = 12000,
    panel_h: int = 720,
) -> None:
    """hstack two videos, each trimmed from its own start so both share t=0,
    scaled to a common height, with optional burned ASS telemetry."""
    encoder = resolve_encoder(encoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    scale = f"scale=-2:{panel_h}"
    graph = (
        f"[0:v]trim=start={left_start:.3f}:duration={duration:.3f},"
        f"setpts=PTS-STARTPTS,{scale}[l];"
        f"[1:v]trim=start={right_start:.3f}:duration={duration:.3f},"
        f"setpts=PTS-STARTPTS,{scale}[r];"
        f"[l][r]hstack=inputs=2[stacked]"
    )
    if ass_path is not None:
        ass_esc = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
        graph += f";[stacked]subtitles='{ass_esc}'[outv]"
    else:
        graph += ";[stacked]copy[outv]"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(left), "-i", str(right),
        "-filter_complex", graph, "-map", "[outv]", "-an",
        "-c:v", encoder, "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(tmp),
    ]
    _run_ffmpeg(cmd, tmp)
    tmp.replace(dst)
```
> The panels are scaled to a common height (`panel_h`); in the test both inputs are 240 px tall, so `scale=-2:720` would upscale — set `panel_h=240` is not passed, so the test asserts the *hstacked width* (640) not height. If ffmpeg rounds `-2` width oddly, the assert on width may need `w in (640, 642)`; keep `scale=-2:240` behavior by noting panels equal height. (For the test, both inputs are already 240 tall and 320 wide; with default panel_h=720 each becomes ~960 wide. **Fix:** pass `panel_h=240` from the test call.) Update the test Step 1 `render_sidebyside(..., )` call to include `panel_h=240` and assert `w == 640`.

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync_render.py -v`
Expected: PASS. If width assertion fails due to `-2` rounding, relax to `w in (638, 640, 642)` and re-run.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/ffmpeg_wrap.py tests/test_sync_render.py
git commit -m "feat(ffmpeg): side-by-side render with time offset + burned telemetry"
```

---

## Task 10: Optional v360 auto-reframe (dual-fisheye → flat)

Only used when `--insta-flat` is absent. The `.insv` carries two separate fisheye video streams (front lens = stream 0, back lens = stream 1); combine them into a single dual-fisheye frame, then `v360` to a flat ride-view.

**Files:**
- Modify: `src/dcwb/ffmpeg_wrap.py`
- Test: `tests/test_sync_render.py`

- [ ] **Step 1: Write the failing test (uses a synthetic dual-stream mp4)**

Append to `tests/test_sync_render.py`:
```python
import subprocess
from dcwb.ffmpeg_wrap import reframe_insv

def _make_dual_fisheye(path: Path):
    # two video streams in one mp4, both 240x240, 2s
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","lavfi","-i","testsrc=d=2:s=240x240:r=30",
        "-f","lavfi","-i","testsrc2=d=2:s=240x240:r=30",
        "-map","0:v","-map","1:v","-c:v","libx264","-pix_fmt","yuv420p",
        "-preset","ultrafast", str(path)], check=True, capture_output=True)

def test_reframe_insv_outputs_flat(tmp_path):
    src = tmp_path / "dual.insv"; _make_dual_fisheye(src)
    dst = tmp_path / "flat.mp4"
    reframe_insv(src, dst, yaw=0.0, pitch=-10.0, out_w=480, out_h=270,
                 encoder="libx264", bitrate_kbps=2000)
    assert dst.exists()
    w, h = _probe_dims(dst)
    assert (w, h) == (480, 270)
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync_render.py -k reframe -v`
Expected: FAIL — `ImportError: cannot import name 'reframe_insv'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/ffmpeg_wrap.py`:
```python
def reframe_insv(
    insv: Path, dst: Path, yaw: float = 0.0, pitch: float = -10.0,
    out_w: int = 1920, out_h: int = 1080, h_fov: float = 100.0, v_fov: float = 60.0,
    encoder: str = "h264_videotoolbox", bitrate_kbps: int = 12000,
) -> None:
    """Dual-fisheye .insv -> flat ride-view via v360. Front lens = stream 0,
    back lens = stream 1; hstack into a dual-fisheye frame then project."""
    encoder = resolve_encoder(encoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    graph = (
        "[0:v][1:v]hstack=inputs=2[df];"
        f"[df]v360=dfisheye:flat:yaw={yaw}:pitch={pitch}:"
        f"h_fov={h_fov}:v_fov={v_fov}:w={out_w}:h={out_h}[outv]"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(insv),
        "-filter_complex", graph, "-map", "[outv]", "-an",
        "-c:v", encoder, "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(tmp),
    ]
    _run_ffmpeg(cmd, tmp)
    tmp.replace(dst)
```
> The two fisheye streams come from one input file (`-i insv`) as `[0:v]` and `[1:v]`. The flat framing (`yaw/pitch/fov`) is a best-effort default; the user's `--insta-flat` export (Task 12 default) is preferred when available.

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync_render.py -k reframe -v`
Expected: PASS. If the installed ffmpeg lacks `v360`, mark this test `@pytest.mark.skipif` on a `v360` filter probe and note v360-reframe needs an ffmpeg build with `--enable-filter=v360`.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/ffmpeg_wrap.py tests/test_sync_render.py
git commit -m "feat(ffmpeg): v360 dual-fisheye .insv -> flat ride-view reframe"
```

---

## Task 11: Sync manifest writer

**Files:**
- Modify: `src/dcwb/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py`:
```python
import json
from dcwb.sync import write_sync_manifest, SyncResult

def test_write_sync_manifest_roundtrip(tmp_path):
    res = SyncResult(delta_s=2.13, confidence=0.91, signal="yaw_rate", anchor_guess=1.5)
    out = write_sync_manifest(
        tmp_path, res,
        insta_display="/x/flat.mp4", tesla_concat="/x/tesla.mp4",
        combined="/x/combined.mp4", date="2026-05-27",
        telemetry=[(0.0, 13.3, 12.0, "DRIVE")],
    )
    data = json.loads(out.read_text())
    assert out.name == "sync.json"
    assert data["delta_s"] == 2.13
    assert data["confidence"] == 0.91
    assert data["signal"] == "yaw_rate"
    assert data["paths"]["combined"] == "/x/combined.mp4"
    assert data["telemetry"][0] == [0.0, 13.3, 12.0, "DRIVE"]
```

- [ ] **Step 1b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_sync.py -k manifest -v`
Expected: FAIL — `ImportError: cannot import name 'write_sync_manifest'`.

- [ ] **Step 2: Implement**

Append to `src/dcwb/sync.py`:
```python
import json

def write_sync_manifest(out_dir: Path, result: SyncResult, *,
                        insta_display: str, tesla_concat: str, combined: str,
                        date: str, telemetry) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "date": date,
        "delta_s": result.delta_s,
        "confidence": result.confidence,
        "signal": result.signal,
        "anchor_guess": result.anchor_guess,
        "paths": {"insta_display": insta_display, "tesla_concat": tesla_concat,
                  "combined": combined},
        "telemetry": [list(r) for r in telemetry],
    }
    path = out_dir / "sync.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path
```

- [ ] **Step 3: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_sync.py -k manifest -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dcwb/sync.py tests/test_sync.py
git commit -m "feat(sync): write sync.json manifest (delta, confidence, paths, telemetry)"
```

---

## Task 12: CLI `sync-insta360` wiring

**Files:**
- Modify: `src/dcwb/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Read the existing cli subcommand pattern**

Read `src/dcwb/cli.py:70-120` (the `highlight-day` parser + handler) and `tests/test_cli.py:200-230` (how a handler is monkeypatched to capture parsed args). Mirror that structure exactly.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_cli.py`:
```python
def test_sync_insta360_parses_args(monkeypatch, tmp_path):
    import dcwb.cli as cli
    captured = {}
    def fake_run(**kw):
        captured.update(kw); return 0
    monkeypatch.setattr(cli, "run_sync_insta360", fake_run, raising=False)
    insv = tmp_path / "VID.insv"; insv.write_bytes(b"")
    rc = cli.main(["sync-insta360", str(insv), "--recent", "2026-05-27",
                   "--insta-flat", str(tmp_path / "flat.mp4"),
                   "--encoder", "libx264"])
    assert rc == 0
    assert captured["recent"] == "2026-05-27"
    assert captured["encoder"] == "libx264"
    assert str(insv) in [str(p) for p in captured["insv"]]
```

- [ ] **Step 2b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli.py -k sync_insta360 -v`
Expected: FAIL — `invalid choice: 'sync-insta360'`.

- [ ] **Step 3: Implement parser + handler**

In `src/dcwb/cli.py`, add the subparser near the other `sub.add_parser(...)` calls:
```python
    py = sub.add_parser("sync-insta360",
                        help="Sync an Insta360 ride-view with Tesla front clips")
    py.add_argument("insv", nargs="+", type=Path)
    py.add_argument("--recent", required=True, help="RecentClips date dir, YYYY-MM-DD")
    py.add_argument("--insta-flat", type=Path, default=None,
                    help="Reframed flat ride-view mp4 (else v360 auto-reframe)")
    py.add_argument("--source", type=Path, default=Path("/mnt/sentryusb"))
    py.add_argument("--out-root", type=Path, default=None)
    py.add_argument("--encoder", default="h264_videotoolbox")
    py.add_argument("--bitrate-kbps", type=int, default=12000)
```
Add the dispatch branch in `main()` alongside the others:
```python
    if args.cmd == "sync-insta360":
        return run_sync_insta360(
            insv=args.insv, recent=args.recent, insta_flat=args.insta_flat,
            source=args.source, out_root=args.out_root,
            encoder=args.encoder, bitrate_kbps=args.bitrate_kbps)
```
Add the orchestrator (wires Tasks 2–11 together). Place it in `cli.py` for now or in `sync.py`; if in `sync.py`, import it. Minimal body:
```python
def run_sync_insta360(*, insv, recent, insta_flat, source, out_root,
                      encoder, bitrate_kbps) -> int:
    from dcwb import insta360, sync
    from dcwb.ffmpeg_wrap import concat_clips, render_sidebyside, reframe_insv, probe_duration
    import numpy as np
    insv = [Path(p) for p in insv]
    start_utc = insta360.read_creation_time(insv[0])
    start_jst = insta360.to_jst(start_utc)
    total = sum(probe_duration(p) for p in insv)
    end_jst = start_jst + timedelta(seconds=total)
    day_dir = Path(source) / "RecentClips" / recent
    fronts = sync.select_front_clips(day_dir, start_jst, end_jst)
    # build tesla per-frame series across fronts (absolute seconds)
    t, yaw_in, ax_in, tele = [], [], [], []
    # ... iterate fronts via telemetry.iter_segment_frames, fps from probe_duration,
    #     accumulate absolute time = front_start + frame_index/fps
    # build insta series from IMU
    imu = insta360.read_imu(insv[0])
    # ... map imu axes to yaw_rate/accel_x (calibration from Task 1)
    # result = sync.compute_offset(tesla_series, insta_series, anchor_guess=0.0, window_s=10.0, rate_hz=50.0)
    # render: concat fronts -> tesla.mp4; insta display = insta_flat or reframe_insv(...)
    # combined = render_sidebyside(display, tesla, ..., ass=telemetry_ass(...))
    # write manifest
    raise NotImplementedError  # filled by integration step below
```
> The orchestrator body is integration glue over already-tested units. Implement it incrementally, running `dcwb sync-insta360` against the real 2026-05-27 data after each piece. The **unit tests above cover every pure unit**; this function has no new logic to unit-test beyond wiring, so cover it with the integration check in Step 5.

- [ ] **Step 4: Run the parser test to verify it passes**

Run: `uv run --extra dev pytest tests/test_cli.py -k sync_insta360 -v`
Expected: PASS (the test monkeypatches `run_sync_insta360`, so the NotImplementedError body is never hit).

- [ ] **Step 5: Real-data integration smoke (manual, not CI)**

Run (only on the workstation with the USB mounted):
```bash
uv run dcwb sync-insta360 \
  "/mnt/sentryusb/Insta360/VID_20260527_171757_00_007_009-オリジナル/VID_20260527_171757_00_007.insv" \
  --recent 2026-05-27 --encoder libx264 --out-root ./sync-work
```
Expected: `./sync-work/sync/2026-05-27/combined-*.mp4` + `sync.json` exist; `sync.json.confidence` printed; spot-check the combined video shows aligned motion. Iterate axis calibration until a turn lines up.

- [ ] **Step 6: Commit**

```bash
git add src/dcwb/cli.py src/dcwb/sync.py tests/test_cli.py
git commit -m "feat(cli): sync-insta360 subcommand wiring + orchestrator"
```

---

## Task 13: Serve sync player (output C)

**Files:**
- Modify: `src/dcwb/serve/` (app + a new template)
- Test: `tests/test_serve_sync.py`

- [ ] **Step 1: Read the existing serve route pattern**

Read how `serve/` registers routes and serves `/corrected/...` files (path-traversal guard via `is_relative_to`). Mirror it for `/sync/...`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_serve_sync.py`:
```python
import json
from pathlib import Path
from dcwb.serve import create_app  # match the existing factory import

def _app(tmp_path):
    sync_dir = tmp_path / "sync" / "2026-05-27"
    sync_dir.mkdir(parents=True)
    (sync_dir / "sync.json").write_text(json.dumps({
        "date": "2026-05-27", "delta_s": 2.1, "confidence": 0.9, "signal": "yaw_rate",
        "paths": {"insta_display": "i.mp4", "tesla_concat": "t.mp4", "combined": "c.mp4"},
        "telemetry": [[0.0, 13.3, 12.0, "DRIVE"]],
    }))
    return create_app(out_root=tmp_path, source=tmp_path), sync_dir

def test_sync_data_route_returns_manifest(tmp_path):
    app, _ = _app(tmp_path)
    c = app.test_client()
    r = c.get("/sync-data/2026-05-27")
    assert r.status_code == 200
    assert r.get_json()["delta_s"] == 2.1

def test_sync_nudge_updates_delta(tmp_path):
    app, sync_dir = _app(tmp_path)
    c = app.test_client()
    r = c.post("/sync-nudge/2026-05-27", json={"delta_s": 2.55})
    assert r.status_code == 200
    saved = json.loads((sync_dir / "sync.json").read_text())
    assert saved["delta_s"] == 2.55

def test_sync_player_page_renders(tmp_path):
    app, _ = _app(tmp_path)
    r = app.test_client().get("/sync/2026-05-27")
    assert r.status_code == 200
    assert b"<video" in r.data
```
> If `create_app`'s real signature differs (check Step 1), adjust the `_app` factory call to match — keep the three assertions.

- [ ] **Step 2b: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_serve_sync.py -v`
Expected: FAIL — 404 on the new routes.

- [ ] **Step 3: Implement the routes + template**

Add to the serve app (mirror existing route style and the `is_relative_to` guard):
```python
@app.get("/sync-data/<date>")
def sync_data(date):
    f = (out_root / "sync" / date / "sync.json").resolve()
    if not f.is_relative_to(out_root.resolve()) or not f.exists():
        abort(404)
    return send_file(f, mimetype="application/json")

@app.post("/sync-nudge/<date>")
def sync_nudge(date):
    f = (out_root / "sync" / date / "sync.json").resolve()
    if not f.is_relative_to(out_root.resolve()) or not f.exists():
        abort(404)
    data = json.loads(f.read_text())
    data["delta_s"] = float(request.get_json()["delta_s"])
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return {"ok": True, "delta_s": data["delta_s"]}

@app.get("/sync/<date>")
def sync_player(date):
    return render_template("sync_player.html", date=date)
```
Create `src/dcwb/serve/templates/sync_player.html`: two `<video>` elements (combined view optional; primary = insta display + tesla concat), a nudge slider, and JS that on `timeupdate` of the master sets `slave.currentTime = master.currentTime + delta`, loads `/sync-data/<date>` for `delta_s` + telemetry, draws telemetry text on a canvas, and POSTs `/sync-nudge/<date>` when the slider changes. Must contain a literal `<video` tag for the test.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_serve_sync.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Manual player check**

Run: `uv run dcwb serve --out-root ./sync-work` then open `/sync/2026-05-27`; scrub the master video and confirm the slave follows; nudge the slider and confirm it persists after reload.

- [ ] **Step 6: Commit**

```bash
git add src/dcwb/serve tests/test_serve_sync.py
git commit -m "feat(serve): synchronized Insta360/Tesla player with manual nudge"
```

---

## Task 14: Full suite + docs

**Files:**
- Modify: `CLAUDE.md`, `README`

- [ ] **Step 1: Run the entire suite**

Run: `uv run --extra dev pytest`
Expected: all pass (real-`.insv`/`v360` tests SKIP where unavailable).

- [ ] **Step 2: Update docs**

Add a `sync-insta360` paragraph to `CLAUDE.md` (CLI subcommand list + module dependency notes: `insta360.py`, `sync.py`, serve player; trailer-seek performance note; vendored Insta360 NOTICE) and a workflow section to `README`. Reference the spec `docs/superpowers/specs/2026-05-30-insta360-tesla-sync-design.md`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README* docs/
git commit -m "docs: document dcwb sync-insta360 workflow and modules"
```

---

## Self-Review Notes

- **Spec coverage:** anchor (T6/T12), IMU trailer (T1/T3), Tesla per-frame SEI (T5), yaw-rate-primary/accel-fallback correlation + confidence (T4/T6), region selection (T7), side-by-side + telemetry overlay = layout D (T8/T9), v360 fallback (T10), manifest (T11), CLI (T12), serve player + nudge (T13), trailer-seek/NVMe-staging performance (T3 read-bound + T12 `--out-root` work dir), risk/fallback gate (T1 Step 4). All spec sections map to tasks.
- **Degraded mode:** if Task 1 fails, Tasks 4/6 are skipped; Task 12 sets `delta_s` from the anchor only and relies on Task 13's nudge. The plan still ships B+C.
- **Type consistency:** `ImuSample(t_s, accel, gyro)`, `MotionSeries(t, yaw_rate, accel_x)`, `SyncResult(delta_s, confidence, signal, anchor_guess)`, `FrameTelemetry(...)`, `telemetry` rows are `(t, speed_mps, steering_deg, gear)` everywhere (render/manifest/ass).
- **Known calibration unknown:** the IMU→(yaw_rate, accel_x) axis/sign/unit mapping is pinned in Task 1 and applied in Task 12's orchestrator; it's the one piece that needs real-data iteration (Task 12 Step 5).
