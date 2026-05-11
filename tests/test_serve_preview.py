from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
from dcwb.profile import Profile, CalibrationMeta
from dcwb.serve.index import Event
from dcwb.serve.preview import ensure_previews, PreviewResult
from tests.fixtures.make_synthetic import make_event

CAMERAS = ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater")
PIPELINE_CFG = {
    "samples_per_clip": 3, "minkowski_p": 6,
    "saturation_high": 0.97, "saturation_low": 0.03,
    "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
}


def _make_profiles(profiles_dir: Path) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        cast = (1.1, 1.0, 0.9) if cam == "front" else (1.0, 1.0, 1.0)
        p = Profile.from_white_point(
            cam, np.array(cast) * 200.0,
            CalibrationMeta(samples_used=10, events_sampled=2, method="t",
                            calibrated_at=datetime.now(timezone.utc),
                            samples_per_event_max=3),
        )
        p.to_json(profiles_dir / f"{cam}.json")


def _event_from_dir(event_dir: Path) -> Event:
    clips = sorted(event_dir.glob("*.mp4"))
    return Event(
        source="SentryClips",
        name=event_dir.name,
        path=event_dir,
        clips=clips,
        start=datetime(2026, 5, 5, 13, 49, 39),
        end=datetime(2026, 5, 5, 13, 50, 39),
    )


def test_ensure_previews_creates_files_and_meta(tmp_path):
    ev_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(ev_dir)
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)

    res = ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)

    assert isinstance(res, PreviewResult)
    cache_dir = cache_root / "previews" / ev.source / ev.name
    assert (cache_dir / "meta.json").exists()
    for cam in CAMERAS:
        assert (cache_dir / f"{cam}_before.png").exists()
        assert (cache_dir / f"{cam}_after.png").exists()
        assert res.paths[cam]["before"].name == f"{cam}_before.png"
        assert res.paths[cam]["after"].name == f"{cam}_after.png"
        assert cam in res.scene_gains
        assert len(res.scene_gains[cam]) == 3
        assert res.errors[cam] is None


def test_ensure_previews_reuses_cache_when_unchanged(tmp_path):
    ev_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(ev_dir)
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)

    ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    cache_dir = cache_root / "previews" / ev.source / ev.name
    before_mtime = (cache_dir / "front_before.png").stat().st_mtime_ns

    # second call should reuse
    ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    after_mtime = (cache_dir / "front_before.png").stat().st_mtime_ns
    assert before_mtime == after_mtime


def test_ensure_previews_regenerates_when_profile_changes(tmp_path):
    ev_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(ev_dir)
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)

    ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    cache_dir = cache_root / "previews" / ev.source / ev.name
    before_mtime = (cache_dir / "front_before.png").stat().st_mtime_ns

    # touch one profile so its mtime moves forward
    front_profile = profiles_dir / "front.json"
    import os, time
    time.sleep(0.01)
    os.utime(front_profile, None)

    ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    after_mtime = (cache_dir / "front_before.png").stat().st_mtime_ns
    assert after_mtime > before_mtime


def test_ensure_previews_passes_night_attenuation(tmp_path, monkeypatch):
    """Regression: nighttime events must drive compose_clip_matrix through
    night_attenuation. Previously preview composed without attenuation, so
    /verify and / event detail showed stronger AWB than the rendered output."""
    import dcwb.serve.preview as prev_mod

    ev_dir = tmp_path / "src" / "2026-05-05_22-00-00"
    make_event(ev_dir)
    (ev_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T22:00:00","city":"Tokyo",'
        '"street":"","est_lat":"35.68","est_lon":"139.65",'
        '"reason":"sentry_aware_object_detection","camera":"5"}'
    )
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)

    captured: list[float] = []
    real = prev_mod.compute_frame_triple

    def spy(clip, profile, awb_cfg, attenuation=1.0):
        captured.append(attenuation)
        return real(clip, profile, awb_cfg, attenuation=attenuation)

    monkeypatch.setattr(prev_mod, "compute_frame_triple", spy)
    prev_mod.ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    assert captured, "compute_frame_triple should have been called"
    assert all(a == 0.5 for a in captured), \
        f"expected night_attenuation=0.5 for all cameras, got {captured}"


def test_ensure_previews_daytime_uses_unit_attenuation(tmp_path, monkeypatch):
    """Daytime events keep attenuation=1.0 (no behaviour change for daytime)."""
    import dcwb.serve.preview as prev_mod

    ev_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(ev_dir)
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)

    captured: list[float] = []
    real = prev_mod.compute_frame_triple

    def spy(clip, profile, awb_cfg, attenuation=1.0):
        captured.append(attenuation)
        return real(clip, profile, awb_cfg, attenuation=attenuation)

    monkeypatch.setattr(prev_mod, "compute_frame_triple", spy)
    prev_mod.ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    assert captured
    assert all(a == 1.0 for a in captured), captured


def test_ensure_previews_records_clip_error_without_failing_event(tmp_path):
    ev_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(ev_dir)
    # corrupt one clip
    bad = ev_dir / "2026-05-05_13-49-39-back.mp4"
    bad.write_bytes(b"not an mp4")
    profiles_dir = tmp_path / "profiles"
    _make_profiles(profiles_dir)
    cache_root = tmp_path / "cache"
    ev = _event_from_dir(ev_dir)
    ev.clips = sorted(ev_dir.glob("*.mp4"))  # refresh after corruption

    res = ensure_previews(ev, profiles_dir, PIPELINE_CFG, cache_root)
    assert res.errors["back"] is not None
    # other cameras still produced files
    assert res.errors["front"] is None
    cache_dir = cache_root / "previews" / ev.source / ev.name
    assert (cache_dir / "front_before.png").exists()
