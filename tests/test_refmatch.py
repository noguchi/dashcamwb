import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
from dcwb.refmatch import reference_gain, compute_reference_gain
from dcwb.profile import Profile, CalibrationMeta
from dcwb.sync import JST


def test_reference_neutral_returns_tesla_gain():
    """If the reference is perfectly neutral (g_R = 1,1,1) the match gain
    collapses to the plain gray-world neutralisation g_T (continuity)."""
    g_t = (1.2, 1.0, 0.85)
    g = reference_gain(g_t, (1.0, 1.0, 1.0))
    assert g == pytest.approx(g_t)


def test_reference_gain_is_channelwise_ratio_green_normalised():
    """G = g_T / g_R per channel, then re-normalised so g_g == 1."""
    g_t = (1.2, 1.0, 0.8)
    g_r = (1.1, 1.0, 0.9)
    g = reference_gain(g_t, g_r)
    assert g[1] == pytest.approx(1.0)
    assert g[0] == pytest.approx(1.2 / 1.1)
    assert g[2] == pytest.approx(0.8 / 0.9)


def test_reference_gain_renormalises_when_green_not_unity():
    """Even if inputs are not green-normalised, the result is (g_g == 1)."""
    g_t = (2.4, 2.0, 1.6)   # == 2 * (1.2, 1.0, 0.8)
    g_r = (1.0, 1.0, 1.0)
    g = reference_gain(g_t, g_r)
    assert g[1] == pytest.approx(1.0)
    assert g[0] == pytest.approx(1.2)
    assert g[2] == pytest.approx(0.8)


def _neutral_profile(camera="front", gain_r=1.0, gain_b=1.0) -> Profile:
    return Profile.from_white_point(
        camera, np.array([1.0 / gain_r, 1.0, 1.0 / gain_b]) * 200.0,
        CalibrationMeta(samples_used=1, events_sampled=1, method="test",
                        calibrated_at=datetime.now(JST), samples_per_event_max=1),
    )


def _uniform(rgb) -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def _patch_orchestration(monkeypatch, *, tesla_rgb, ref_rgb, peak=0.8,
                         daytime=True, durations=120.0):
    import dcwb.refmatch as R

    def fake_extract(path, t):
        name = Path(path).name
        return _uniform(tesla_rgb) if "tesla" in name else _uniform(ref_rgb)

    monkeypatch.setattr(R, "concat_clips", lambda *a, **k: Path(a[1]).write_bytes(b"x"))
    monkeypatch.setattr(R, "reframe_insv", lambda *a, **k: Path(a[1]).write_bytes(b"x"))
    monkeypatch.setattr(R, "probe_duration", lambda p: durations)
    monkeypatch.setattr(R, "detect_visual_offset", lambda *a, **k: (0.0, peak))
    monkeypatch.setattr(R, "extract_frame", fake_extract)
    monkeypatch.setattr(R, "is_daytime", lambda when, lat, lon: daytime)


def test_compute_reference_gain_matches_neutral_reference(tmp_path, monkeypatch):
    """Identity A, a casted Tesla frame and a neutral reference frame: the match
    gain equals the gray-world neutralisation of the Tesla cast (continuity)."""
    _patch_orchestration(monkeypatch, tesla_rgb=(240, 200, 160), ref_rgb=(200, 200, 200))
    fronts = [tmp_path / "2026-05-27_13-00-00-front.mp4"]
    g, peak, n = compute_reference_gain(
        tmp_path / "ride.insv", fronts, _neutral_profile(),
        start_jst=datetime(2026, 5, 27, 13, 0, 0, tzinfo=JST),
        work_dir=tmp_path / "work", samples=4, encoder="libx264",
    )
    # SoG of a uniform (240,200,160) frame, green-normalised: (200/240, 1, 200/160)
    assert g == pytest.approx((200 / 240, 1.0, 200 / 160), rel=1e-3)
    assert peak == pytest.approx(0.8)
    assert n == 4


def test_compute_reference_gain_pulls_toward_reference_tone(tmp_path, monkeypatch):
    """A warm reference shifts the match gain away from pure neutralisation:
    G = g_T / g_R."""
    _patch_orchestration(monkeypatch, tesla_rgb=(240, 200, 160), ref_rgb=(220, 200, 180))
    fronts = [tmp_path / "2026-05-27_13-00-00-front.mp4"]
    g, _, _ = compute_reference_gain(
        tmp_path / "ride.insv", fronts, _neutral_profile(),
        start_jst=datetime(2026, 5, 27, 13, 0, 0, tzinfo=JST),
        work_dir=tmp_path / "work", samples=3,
    )
    g_t = (200 / 240, 1.0, 200 / 160)
    g_r = (200 / 220, 1.0, 200 / 180)
    expected = reference_gain(g_t, g_r)
    assert g == pytest.approx(expected, rel=1e-3)


def test_compute_reference_gain_caps_window_for_large_reference(tmp_path, monkeypatch):
    """A long reference (e.g. a 30-min .insv) must not be reframed/sampled in
    full: max_window bounds the reframe duration and the sampling window so the
    command stays feasible over a network mount."""
    import dcwb.refmatch as R
    captured = {}
    ref_times = []

    def fake_reframe(reference, dst, **k):
        captured["reframe_duration"] = k.get("duration")
        Path(dst).write_bytes(b"x")

    def fake_extract(path, t):
        if "tesla" not in Path(path).name:
            ref_times.append(t)
        return _uniform((200, 200, 200))

    monkeypatch.setattr(R, "concat_clips", lambda *a, **k: Path(a[1]).write_bytes(b"x"))
    monkeypatch.setattr(R, "reframe_insv", fake_reframe)
    monkeypatch.setattr(R, "probe_duration", lambda p: 1800.0)
    monkeypatch.setattr(R, "detect_visual_offset", lambda *a, **k: (0.0, 0.7))
    monkeypatch.setattr(R, "extract_frame", fake_extract)
    monkeypatch.setattr(R, "is_daytime", lambda when, lat, lon: True)

    fronts = [tmp_path / "2026-05-27_13-00-00-front.mp4"]
    g, peak, n = compute_reference_gain(
        tmp_path / "ride.insv", fronts, _neutral_profile(),
        start_jst=datetime(2026, 5, 27, 13, 0, 0, tzinfo=JST),
        work_dir=tmp_path / "work", samples=6, max_window=600.0,
    )
    assert captured["reframe_duration"] == 600.0
    assert ref_times and max(ref_times) <= 600.0


def test_compute_reference_gain_raises_when_no_daytime_frames(tmp_path, monkeypatch):
    _patch_orchestration(monkeypatch, tesla_rgb=(200, 200, 200),
                         ref_rgb=(200, 200, 200), daytime=False)
    fronts = [tmp_path / "2026-05-27_13-00-00-front.mp4"]
    with pytest.raises(Exception):
        compute_reference_gain(
            tmp_path / "ride.insv", fronts, _neutral_profile(),
            start_jst=datetime(2026, 5, 27, 13, 0, 0, tzinfo=JST),
            work_dir=tmp_path / "work", samples=3,
        )
