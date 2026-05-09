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
