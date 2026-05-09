import json
import sys
import pytest
from pathlib import Path
from dcwb import cli

def test_cli_no_args_prints_help(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
    captured = capsys.readouterr()
    assert "calibrate" in captured.out or "calibrate" in captured.err

def test_cli_calibrate_invokes_calibrate_camera(tmp_path, monkeypatch):
    called = []
    def fake_calibrate(**kw):
        called.append(kw)
        from dcwb.profile import Profile, CalibrationMeta
        from datetime import datetime, timezone
        import numpy as np
        return Profile.from_white_point(
            kw["camera"], np.array([200.0, 200.0, 200.0]),
            CalibrationMeta(
                samples_used=10, events_sampled=5, method="test",
                calibrated_at=datetime.now(timezone.utc),
                samples_per_event_max=3,
            ),
        )
    monkeypatch.setattr("dcwb.cli.calibrate_camera", fake_calibrate)
    out_profiles = tmp_path / "profiles"
    cli.main([
        "calibrate",
        "--source", str(tmp_path),
        "--profiles-dir", str(out_profiles),
        "--max-samples-per-event", "2",
    ])
    assert len(called) == 6  # 6 cameras
    assert all((out_profiles / f"{c}.json").exists()
               for c in ("front","back","left_pillar","right_pillar","left_repeater","right_repeater"))

def test_cli_render_invokes_render_event(tmp_path, monkeypatch):
    captured = {}
    def fake_render(**kw):
        captured.update(kw)
    monkeypatch.setattr("dcwb.cli.render_event", fake_render)
    event_dir = tmp_path / "evt"
    event_dir.mkdir()
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    out_root = tmp_path / "out"
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(json.dumps({"awb": {
        "method": "shades_of_gray", "minkowski_p": 6, "samples_per_clip": 3,
        "saturation_high": 0.97, "saturation_low": 0.03,
        "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
    }}))
    cli.main([
        "render", str(event_dir),
        "--profiles-dir", str(profiles_dir),
        "--out-root", str(out_root),
        "--pipeline-config", str(pipeline_cfg),
    ])
    assert captured["event_dir"] == event_dir
    assert captured["out_root"] == out_root
