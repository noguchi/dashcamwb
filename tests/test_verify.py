import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from dcwb.verify import generate_verify_report
from dcwb.profile import Profile, CalibrationMeta
from tests.fixtures.make_synthetic import make_event

def test_generate_verify_report_uses_night_attenuation(tmp_path, monkeypatch):
    """Regression: verify report must mirror render.py's night attenuation so
    the preview matches the actual rendered output for nighttime events."""
    import dcwb.verify as verify_mod
    from dcwb.serve.preview import compute_frame_triple as real

    event_dir = tmp_path / "src" / "evt-night"
    make_event(event_dir)
    (event_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T22:00:00","city":"Tokyo",'
        '"street":"","est_lat":"35.68","est_lon":"139.65",'
        '"reason":"sentry_aware_object_detection","camera":"5"}'
    )
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        Profile.from_white_point(
            cam, np.array([200.0, 200.0, 200.0]),
            CalibrationMeta(samples_used=10, events_sampled=1, method="t",
                            calibrated_at=datetime.now(timezone.utc),
                            samples_per_event_max=3),
        ).to_json(profiles_dir / f"{cam}.json")

    captured: list[float] = []

    def spy(clip, profile, awb_cfg, attenuation=1.0):
        captured.append(attenuation)
        return real(clip, profile, awb_cfg, attenuation=attenuation)

    monkeypatch.setattr(verify_mod, "compute_frame_triple", spy)
    out = tmp_path / "report.html"
    verify_mod.generate_verify_report(
        event_dir=event_dir,
        profiles_dir=profiles_dir,
        out_html=out,
        encoder="libx264",
        pipeline_cfg={"awb": {
            "samples_per_clip": 3, "minkowski_p": 6,
            "saturation_high": 0.97, "saturation_low": 0.03,
            "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
        }},
    )
    assert captured, "compute_frame_triple should have been called"
    assert all(a == 0.5 for a in captured), captured


def test_generate_verify_report_creates_html(tmp_path):
    event_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(event_dir, casts={
        "front": (1.1, 1.0, 0.9),
        "back": (1.0, 1.0, 1.0),
        "left_pillar": (1.0, 1.0, 1.0),
        "right_pillar": (1.0, 1.0, 1.0),
        "left_repeater": (1.0, 1.0, 1.0),
        "right_repeater": (1.0, 1.0, 1.0),
    })
    # profiles
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        cast = (1.1, 1.0, 0.9) if cam == "front" else (1.0, 1.0, 1.0)
        p = Profile.from_white_point(
            cam, np.array(cast) * 200.0,
            CalibrationMeta(samples_used=10, events_sampled=2, method="t",
                            calibrated_at=datetime.now(timezone.utc),
                            samples_per_event_max=3),
        )
        p.to_json(profiles_dir / f"{cam}.json")

    out = tmp_path / "report.html"
    generate_verify_report(
        event_dir=event_dir,
        profiles_dir=profiles_dir,
        out_html=out,
        encoder="libx264",
        pipeline_cfg={"awb": {
            "samples_per_clip": 3, "minkowski_p": 6,
            "saturation_high": 0.97, "saturation_low": 0.03,
            "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
        }},
    )
    assert out.exists()
    html = out.read_text()
    # 6 カメラ分が含まれている
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        assert cam in html
    # 補正前後の画像（base64）参照が3列ある想定
    assert html.count("data:image/png;base64,") >= 6  # 各カメラごとに最低1枚
