import numpy as np
from pathlib import Path
from dcwb import cli
from dcwb.ffmpeg_wrap import extract_frame
from tests.fixtures.make_synthetic import make_event, CAMERAS

def test_full_pipeline_calibrate_then_render(tmp_path, monkeypatch):
    """Generate a synthetic event with a known per-camera cast,
    run dcwb calibrate, then dcwb render, and assert the rendered
    front clip is neutral within tolerance."""
    monkeypatch.setattr("dcwb.calibrate.is_multicolor", lambda img, threshold=0.05: True)

    source_root = tmp_path / "source"
    sentry_root = source_root / "SentryClips"
    casts = {
        "front": (1.10, 1.00, 0.90),
        "back": (0.95, 1.00, 1.05),
        "left_pillar": (1.05, 1.00, 0.97),
        "right_pillar": (1.02, 1.00, 0.98),
        "left_repeater": (0.98, 1.00, 1.03),
        "right_repeater": (1.01, 1.00, 0.99),
    }
    # 同じイベントを2回生成（calibrate のサンプル数を増やすため）
    for ts in ("2026-05-05_13-50-46", "2026-05-05_14-30-00"):
        make_event(sentry_root / ts, casts=casts)

    profiles_dir = tmp_path / "profiles"
    out_root = tmp_path / "out"
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(
        '{"awb":{"method":"shades_of_gray","minkowski_p":6,'
        '"samples_per_clip":3,"saturation_high":0.97,"saturation_low":0.03,'
        '"gain_min":0.7,"gain_max":1.5,"night_attenuation":0.5}}'
    )

    # 1. Calibrate
    rc = cli.main([
        "calibrate",
        "--source", str(source_root),
        "--profiles-dir", str(profiles_dir),
        "--max-samples-per-event", "3",
    ])
    assert rc == 0
    for cam in CAMERAS:
        assert (profiles_dir / f"{cam}.json").exists()

    # 2. Render
    event_dir = sentry_root / "2026-05-05_13-50-46"
    rc = cli.main([
        "render", str(event_dir),
        "--profiles-dir", str(profiles_dir),
        "--out-root", str(out_root),
        "--pipeline-config", str(pipeline_cfg),
        "--encoder", "libx264",
        "--bitrate-kbps", "4000",
    ])
    assert rc == 0

    # 3. Assert each rendered camera output is neutral
    out_event = out_root / "2026-05-05_13-50-46"
    for cam in CAMERAS:
        clips = sorted(out_event.glob(f"*-{cam}.mp4"))
        assert len(clips) > 0, f"no rendered clip for {cam}"
        img = extract_frame(clips[0], t=1.0)
        mean = img.reshape(-1, 3).mean(axis=0)
        spread = float(max(mean) - min(mean))
        assert spread < 12, f"{cam}: channel spread {spread:.1f} too large (mean={mean})"
