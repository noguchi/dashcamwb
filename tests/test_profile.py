import json
import numpy as np
from datetime import datetime, timezone
from dcwb.profile import Profile, CalibrationMeta

def test_profile_round_trip(tmp_path):
    p = Profile(
        camera="front",
        gain_r=0.918,
        gain_g=1.000,
        gain_b=1.067,
        matrix_3x3=np.diag([0.918, 1.0, 1.067]),
        calibration=CalibrationMeta(
            samples_used=247,
            events_sampled=89,
            method="robust_white_patch_median",
            calibrated_at=datetime(2026, 5, 9, 12, 34, 56, tzinfo=timezone.utc),
            samples_per_event_max=3,
        ),
    )
    path = tmp_path / "front.json"
    p.to_json(path)
    loaded = Profile.from_json(path)
    assert loaded.camera == "front"
    assert loaded.gain_r == 0.918
    assert loaded.calibration.samples_used == 247
    np.testing.assert_array_equal(loaded.matrix_3x3, p.matrix_3x3)

def test_from_white_point_computes_gains():
    p = Profile.from_white_point(
        camera="front",
        rgb_white=np.array([180.0, 200.0, 220.0]),
        meta=CalibrationMeta(
            samples_used=100,
            events_sampled=50,
            method="robust_white_patch_median",
            calibrated_at=datetime.now(timezone.utc),
            samples_per_event_max=3,
        ),
    )
    assert p.gain_r == 200.0 / 180.0
    assert p.gain_g == 1.0
    assert p.gain_b == 200.0 / 220.0

def test_json_format_is_human_readable(tmp_path):
    p = Profile.from_white_point(
        camera="back",
        rgb_white=np.array([200.0, 200.0, 200.0]),
        meta=CalibrationMeta(
            samples_used=10,
            events_sampled=5,
            method="robust_white_patch_median",
            calibrated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
            samples_per_event_max=3,
        ),
    )
    path = tmp_path / "back.json"
    p.to_json(path)
    raw = json.loads(path.read_text())
    assert raw["camera"] == "back"
    assert raw["gain_r"] == 1.0
    assert "calibration" in raw
