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

from dcwb.sync import compute_offset, MotionSeries, SyncResult

def test_compute_offset_recovers_injected_delta():
    rate = 50.0
    t = np.arange(0, 20, 1 / rate)
    sig = np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    tesla = MotionSeries(t=t, yaw_rate=sig, accel_x=np.zeros_like(sig))
    delta_true = 2.0
    insta = MotionSeries(t=t + delta_true, yaw_rate=sig, accel_x=np.zeros_like(sig))
    res = compute_offset(tesla, insta, anchor_guess=1.5, window_s=5.0, rate_hz=rate)
    assert isinstance(res, SyncResult)
    assert abs(res.delta_s - delta_true) < 0.05
    assert res.signal == "yaw_rate"
    assert res.confidence > 0.8

from datetime import datetime, timedelta, timezone
from dcwb.sync import select_front_clips
_JST = timezone(timedelta(hours=9))

def test_select_front_clips_overlapping_window(tmp_path):
    names = ["2026-05-27_17-16-48-front.mp4", "2026-05-27_17-17-04-front.mp4",
             "2026-05-27_17-18-05-front.mp4", "2026-05-27_17-30-07-front.mp4"]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    start = datetime(2026, 5, 27, 17, 17, 57, tzinfo=_JST)
    end = start + timedelta(seconds=70)
    chosen = select_front_clips(tmp_path, start, end, seg_seconds=60.0)
    got = [p.name for p in chosen]
    assert got == ["2026-05-27_17-17-04-front.mp4", "2026-05-27_17-18-05-front.mp4"]

from dcwb.sync import telemetry_ass

def test_telemetry_ass_emits_timed_dialogue():
    rows = [(0.0, 13.3, 12.0, "DRIVE"), (1.0, 14.0, -5.0, "DRIVE")]  # t, mps, steer, gear
    ass = telemetry_ass(rows, play_w=2560, play_h=1080)
    assert "[Script Info]" in ass
    assert ass.count("Dialogue:") == 2
    assert "48 km/h" in ass            # 13.3 m/s -> 48 km/h
    assert "DRIVE" in ass
    assert "0:00:00.00" in ass         # first event start time

import json as _json
from dcwb.sync import write_sync_manifest, SyncResult

def test_write_sync_manifest_roundtrip(tmp_path):
    res = SyncResult(delta_s=2.13, confidence=0.91, signal="yaw_rate", anchor_guess=1.5)
    out = write_sync_manifest(
        tmp_path, res,
        insta_display="/x/flat.mp4", tesla_concat="/x/tesla.mp4",
        combined="/x/combined.mp4", date="2026-05-27",
        telemetry=[(0.0, 13.3, 12.0, "DRIVE")],
    )
    data = _json.loads(out.read_text())
    assert out.name == "sync.json"
    assert data["delta_s"] == 2.13
    assert data["confidence"] == 0.91
    assert data["signal"] == "yaw_rate"
    assert data["paths"]["combined"] == "/x/combined.mp4"
    assert data["telemetry"][0] == [0.0, 13.3, 12.0, "DRIVE"]
