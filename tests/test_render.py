import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timezone
from dcwb.render import render_event, compose_clip_matrix
from dcwb.profile import Profile, CalibrationMeta
from dcwb.matrix import from_diag
from dcwb.ffmpeg_wrap import extract_frame
from tests.fixtures.make_synthetic import make_event, CAMERAS

def _write_neutral_profiles(profiles_dir: Path) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        prof = Profile.from_white_point(
            cam, np.array([200.0, 200.0, 200.0]),
            CalibrationMeta(
                samples_used=100, events_sampled=10, method="test",
                calibrated_at=datetime.now(timezone.utc),
                samples_per_event_max=3,
            ),
        )
        prof.to_json(profiles_dir / f"{cam}.json")

def _default_pipeline_cfg() -> dict:
    return {"awb": {
        "method": "shades_of_gray", "minkowski_p": 6,
        "samples_per_clip": 3, "saturation_high": 0.97, "saturation_low": 0.03,
        "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
    }}

def _mk_profile(camera: str, gain_r=1.0, gain_b=1.0) -> Profile:
    return Profile.from_white_point(
        camera=camera,
        rgb_white=np.array([1.0 / gain_r, 1.0, 1.0 / gain_b]) * 200.0,
        meta=CalibrationMeta(
            samples_used=100, events_sampled=10,
            method="test", calibrated_at=datetime.now(timezone.utc),
            samples_per_event_max=3,
        ),
    )

def test_compose_clip_matrix_combines_profile_and_scene_gain():
    profile = _mk_profile("front", gain_r=0.9, gain_b=1.1)
    final = compose_clip_matrix(profile, scene_gain=(1.0, 1.0, 1.0))
    np.testing.assert_array_almost_equal(final, profile.matrix_3x3)

def test_compose_clip_matrix_applies_fallback_when_gain_extreme():
    profile = _mk_profile("front")
    final = compose_clip_matrix(
        profile,
        scene_gain=(1.0, 1.0, 2.0),  # > gain_max=1.5 → B 全体破棄
        gain_min=0.7, gain_max=1.5,
    )
    # B 破棄なら final は profile.matrix_3x3 と一致
    np.testing.assert_array_almost_equal(final, profile.matrix_3x3)

def test_render_event_neutralises_known_cast(tmp_path):
    # source: front カメラに R=1.10 のキャスト
    source_root = tmp_path / "src"
    event_name = "2026-05-05_13-50-46"
    event_dir = source_root / "SentryClips" / event_name
    cast = (1.10, 1.00, 0.90)
    make_event(
        event_dir,
        casts={
            "front": cast,
            "back": (1.0, 1.0, 1.0),
            "left_pillar": (1.0, 1.0, 1.0),
            "right_pillar": (1.0, 1.0, 1.0),
            "left_repeater": (1.0, 1.0, 1.0),
            "right_repeater": (1.0, 1.0, 1.0),
        },
    )

    # profiles/ を準備 (キャストの逆ゲイン)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam, c in [
        ("front", cast),
        ("back", (1.0, 1.0, 1.0)),
        ("left_pillar", (1.0, 1.0, 1.0)),
        ("right_pillar", (1.0, 1.0, 1.0)),
        ("left_repeater", (1.0, 1.0, 1.0)),
        ("right_repeater", (1.0, 1.0, 1.0)),
    ]:
        # Profile.from_white_point は rgb_white から gain を逆算 → 逆キャスト
        white_pt = np.array(c) * 200.0
        prof = Profile.from_white_point(
            cam, white_pt,
            CalibrationMeta(
                samples_used=100, events_sampled=10, method="test",
                calibrated_at=datetime.now(timezone.utc),
                samples_per_event_max=3,
            ),
        )
        prof.to_json(profiles_dir / f"{cam}.json")

    out_root = tmp_path / "corrected"
    pipeline_cfg = {
        "awb": {
            "method": "shades_of_gray", "minkowski_p": 6,
            "samples_per_clip": 3, "saturation_high": 0.97, "saturation_low": 0.03,
            "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
        }
    }
    render_event(
        event_dir=event_dir,
        out_root=out_root,
        profiles_dir=profiles_dir,
        pipeline_cfg=pipeline_cfg,
        encoder="libx264",
    )
    out_event_dir = out_root / event_name
    assert out_event_dir.exists()

    # 出力 front 動画を確認 → R=G=B（補正成功）
    front_clips = sorted(out_event_dir.glob("*-front.mp4"))
    assert len(front_clips) > 0
    img = extract_frame(front_clips[0], t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    # 全チャネルが ≈180 に揃う（合成 base_gray=180）
    assert max(mean) - min(mean) < 10

    # event.json と _pipeline.json が出力されている
    assert (out_event_dir / "event.json").exists()
    assert (out_event_dir / "_pipeline.json").exists()

def test_estimate_scene_gain_handles_short_clip(tmp_path):
    """Real Tesla DashCam events end with a short trailing segment (~5s).

    With samples_per_clip=10 the last sampled timestamp lands close to EOF
    where OpenCV's seek can return no frame. estimate_scene_gain must
    sample with a small EOF guard and skip per-frame failures rather than
    raise.
    """
    from tests.fixtures.make_synthetic import make_clip
    from dcwb.render import estimate_scene_gain
    short = tmp_path / "short.mp4"
    make_clip(short, cast_rgb=(1.0, 1.0, 1.0), duration_sec=1.5)
    profile = Profile.from_white_point(
        "front", np.array([200.0, 200.0, 200.0]),
        CalibrationMeta(samples_used=10, events_sampled=2, method="t",
                        calibrated_at=datetime.now(timezone.utc),
                        samples_per_event_max=3),
    )
    g_r, g_g, g_b = estimate_scene_gain(
        short, profile, samples_per_clip=10,
        sat_high=0.97, sat_low=0.03, p=6,
    )
    assert g_g == 1.0
    assert 0.9 < g_r < 1.1
    assert 0.9 < g_b < 1.1

def test_render_event_resilient_to_corrupted_clip(tmp_path):
    """spec §7.4: a single corrupted clip must not abort the whole event.

    Other 5 cameras render normally; the failed clip is recorded as an error
    entry in _pipeline.json; event.json + _pipeline.json are written even
    though one clip blew up.
    """
    source_root = tmp_path / "src"
    event_name = "2026-05-05_13-50-46"
    event_dir = source_root / "SentryClips" / event_name
    make_event(event_dir)  # all neutral

    # corrupt the front clip
    front_clips = sorted(event_dir.glob("*-front.mp4"))
    assert len(front_clips) == 1
    corrupt_clip = front_clips[0]
    corrupt_clip.write_bytes(b"not an mp4")

    profiles_dir = tmp_path / "profiles"
    _write_neutral_profiles(profiles_dir)

    out_root = tmp_path / "corrected"
    # must NOT raise
    render_event(
        event_dir=event_dir,
        out_root=out_root,
        profiles_dir=profiles_dir,
        pipeline_cfg=_default_pipeline_cfg(),
        encoder="libx264",
    )
    out_event_dir = out_root / event_name
    # event.json + _pipeline.json written despite failure
    assert (out_event_dir / "event.json").exists()
    pj_path = out_event_dir / "_pipeline.json"
    assert pj_path.exists()
    snapshot = json.loads(pj_path.read_text())
    # find the front entry
    front_entries = [c for c in snapshot["clips"] if c["clip"].endswith("-front.mp4")]
    assert len(front_entries) == 1
    assert "error" in front_entries[0]
    # the corrupted clip must not have been written to output
    assert not (out_event_dir / corrupt_clip.name).exists()
    # other 5 cameras' rendered outputs exist
    for cam in CAMERAS:
        if cam == "front":
            continue
        rendered = sorted(out_event_dir.glob(f"*-{cam}.mp4"))
        assert len(rendered) == 1, f"expected one rendered clip for {cam}"

def test_render_event_uses_reference_gain_for_all_clips(tmp_path, monkeypatch):
    """spec §3: when awb.reference_gain is set, every clip uses that fixed gain
    as B (no per-clip estimate). The snapshot records it and the final matrix is
    diag(reference_gain) @ A."""
    import dcwb.render as render_mod

    def _boom(*a, **kw):
        raise AssertionError("estimate_scene_gain must not be called with reference_gain set")
    monkeypatch.setattr(render_mod, "estimate_scene_gain", _boom)

    source_root = tmp_path / "src"
    event_name = "2026-05-05_13-50-46"
    event_dir = source_root / "SentryClips" / event_name
    make_event(event_dir)  # all-neutral daytime event

    profiles_dir = tmp_path / "profiles"
    _write_neutral_profiles(profiles_dir)

    ref_gain = [1.2, 1.0, 0.8]
    cfg = _default_pipeline_cfg()
    cfg["awb"]["reference_gain"] = ref_gain

    out_root = tmp_path / "corrected"
    render_event(
        event_dir=event_dir, out_root=out_root, profiles_dir=profiles_dir,
        pipeline_cfg=cfg, encoder="libx264",
    )
    snap = json.loads((out_root / event_name / "_pipeline.json").read_text())
    clips = [c for c in snap["clips"] if "scene_gain" in c]
    assert len(clips) == len(CAMERAS)
    for c in clips:
        assert c["scene_gain"] == ref_gain
        prof = Profile.from_json(profiles_dir / f"{c['camera']}.json")
        expected = from_diag(*ref_gain) @ prof.matrix_3x3
        np.testing.assert_array_almost_equal(np.array(c["final_matrix"]), expected)


def test_render_event_ignores_null_reference_gain(tmp_path, monkeypatch):
    """A null/absent reference_gain keeps the legacy per-clip estimate path."""
    import dcwb.render as render_mod
    called = {"n": 0}
    real = render_mod.estimate_scene_gain

    def _spy(*a, **kw):
        called["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(render_mod, "estimate_scene_gain", _spy)

    event_dir = tmp_path / "src" / "SentryClips" / "2026-05-05_13-50-46"
    make_event(event_dir)
    profiles_dir = tmp_path / "profiles"
    _write_neutral_profiles(profiles_dir)
    cfg = _default_pipeline_cfg()
    cfg["awb"]["reference_gain"] = None  # explicit null → legacy behaviour
    render_event(
        event_dir=event_dir, out_root=tmp_path / "out", profiles_dir=profiles_dir,
        pipeline_cfg=cfg, encoder="libx264",
    )
    assert called["n"] == len(CAMERAS)


def test_render_event_night_attenuation_engaged(tmp_path):
    """spec §4.3: night/twilight events must apply night_attenuation; daytime gets 1.0.

    We render two synthetic events (one daytime, one night) and read
    _pipeline.json["attenuation"] from each.
    """
    profiles_dir = tmp_path / "profiles"
    _write_neutral_profiles(profiles_dir)
    pipeline_cfg = _default_pipeline_cfg()
    out_root = tmp_path / "corrected"

    # Day event: default event.json from make_event uses 13:49:56 (daytime)
    day_event_dir = tmp_path / "src_day" / "2026-05-05_day"
    make_event(day_event_dir)
    render_event(
        event_dir=day_event_dir,
        out_root=out_root,
        profiles_dir=profiles_dir,
        pipeline_cfg=pipeline_cfg,
        encoder="libx264",
    )
    day_snap = json.loads((out_root / day_event_dir.name / "_pipeline.json").read_text())
    assert day_snap["attenuation"] == 1.0

    # Night event: overwrite event.json with a 22:00 timestamp
    night_event_dir = tmp_path / "src_night" / "2026-05-05_night"
    make_event(night_event_dir)
    (night_event_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T22:00:00","city":"Tokyo",'
        '"street":"","est_lat":"35.68","est_lon":"139.65",'
        '"reason":"sentry_aware_object_detection","camera":"5"}'
    )
    render_event(
        event_dir=night_event_dir,
        out_root=out_root,
        profiles_dir=profiles_dir,
        pipeline_cfg=pipeline_cfg,
        encoder="libx264",
    )
    night_snap = json.loads((out_root / night_event_dir.name / "_pipeline.json").read_text())
    assert night_snap["attenuation"] == 0.5
