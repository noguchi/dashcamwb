import json
import sys
import pytest
from pathlib import Path
from dcwb import cli
from dcwb.cli import main
from dcwb.serve.index import CAMERAS
from tests.fixtures.make_synthetic import make_clip

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

def test_serve_help_parses(capsys):
    from dcwb.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.cmd == "serve"
    assert args.port == 9000
    assert str(args.source) == "/Volumes/sentryusb"


def test_cli_prune_recent_dry_run_default(tmp_path, capsys):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"  # old → always past min-age
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
    ])
    assert rc == 0
    assert len(list(day.glob("*.mp4"))) == 6  # dry-run: nothing moved
    assert "2020-01-01_00-00-00" in capsys.readouterr().out


def test_cli_prune_recent_reports_scan_progress(tmp_path, capsys):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for ts in ("2020-01-01_00-00-00", "2020-01-01_00-01-00"):
        for cam in CAMERAS:
            make_clip(day / f"{ts}-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "[prune] scanning RecentClips: 1/2 segment(s) 50.0% candidates=1" in captured.err
    assert "[prune] scanning RecentClips: 2/2 segment(s) 100.0% candidates=2" in captured.err


def test_cli_prune_recent_apply_reports_quarantine_progress(tmp_path, capsys):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
        "--apply",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "[prune] quarantining: 6/6 file(s) 100.0% moved=6" in captured.err


def test_cli_prune_recent_apply_quarantines(tmp_path):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
        "--apply",
    ])
    assert rc == 0
    assert list(day.glob("*.mp4")) == []
    trash = usb / "@dcwb_trash" / "RecentClips" / "2020-01-01"
    assert len(list(trash.glob("*.mp4"))) == 6


def test_cli_prune_recent_restore_with_apply_errors(tmp_path):
    usb = tmp_path / "usb"
    (usb / "RecentClips").mkdir(parents=True)
    rc = main([
        "prune-recent", "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
        "--restore", "all", "--apply",
    ])
    assert rc == 1  # incompatible flags rejected, nothing mutated


def test_cli_prune_recent_restore_roundtrip(tmp_path):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    cfg = str(tmp_path / "absent.json")
    assert main(["prune-recent", "--source", str(usb), "--pipeline-config", cfg, "--apply"]) == 0
    assert list(day.glob("*.mp4")) == []
    assert main(["prune-recent", "--source", str(usb), "--pipeline-config", cfg, "--restore", "all"]) == 0
    assert len(list(day.glob("*.mp4"))) == 6


def test_cli_prune_recent_purge_standalone(tmp_path):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    cfg = str(tmp_path / "absent.json")
    # quarantine with default 14d retention → fresh, not purged
    assert main(["prune-recent", "--source", str(usb), "--pipeline-config", cfg, "--apply"]) == 0
    trash = usb / "@dcwb_trash" / "RecentClips" / "2020-01-01"
    assert len(list(trash.glob("*.mp4"))) == 6
    # standalone purge with retention 0 → all expired → deleted
    assert main(["prune-recent", "--source", str(usb), "--pipeline-config", cfg, "--purge", "--retention-days", "0"]) == 0
    assert list(trash.glob("*.mp4")) == []


def test_cli_highlight_day_invokes_highlight_day(tmp_path, monkeypatch, capsys):
    from dataclasses import dataclass

    captured = {}

    @dataclass
    class FakeResult:
        output_path: Path
        manifest_path: Path
        excerpt_paths: list[Path]
        excerpt_count: int

    def fake_highlight_day(**kw):
        captured.update(kw)
        out = tmp_path / "highlight-fast.mp4"
        manifest = tmp_path / "highlight-fast.json"
        return FakeResult(out, manifest, [], 0)

    monkeypatch.setattr("dcwb.cli.highlight_day", fake_highlight_day)

    rc = main([
        "highlight-day",
        "--source", str(tmp_path / "usb"),
        "--date", "2026-05-08",
        "--out-root", str(tmp_path / "highlights"),
        "--style", "fast",
        "--allow-no-sei",
        "--encoder", "libx264",
        "--bitrate-kbps", "1000",
    ])

    assert rc == 0
    assert captured["source_root"] == (tmp_path / "usb").resolve()
    assert captured["date"] == "2026-05-08"
    assert captured["style"] == "fast"
    assert captured["allow_no_sei"] is True
    assert captured["encoder"] == "libx264"
    assert captured["bitrate_kbps"] == 1000
    assert "no eligible clips" in capsys.readouterr().err


def test_cli_highlight_day_missing_day_returns_error(tmp_path, monkeypatch, capsys):
    def fake_highlight_day(**kw):
        raise FileNotFoundError("missing RecentClips/2026-05-08")

    monkeypatch.setattr("dcwb.cli.highlight_day", fake_highlight_day)

    rc = main(["highlight-day", "--source", str(tmp_path), "--date", "2026-05-08"])

    assert rc == 1
    assert "missing RecentClips/2026-05-08" in capsys.readouterr().err
