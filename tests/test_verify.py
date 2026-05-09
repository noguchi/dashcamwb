import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from dcwb.verify import generate_verify_report
from dcwb.profile import Profile, CalibrationMeta
from tests.fixtures.make_synthetic import make_event

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
    )
    assert out.exists()
    html = out.read_text()
    # 6 カメラ分が含まれている
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        assert cam in html
    # 補正前後の画像（base64）参照が3列ある想定
    assert html.count("data:image/png;base64,") >= 6  # 各カメラごとに最低1枚
