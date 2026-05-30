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
    from dcwb import cli

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

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    monkeypatch.setattr(cli, "VlmClient", LiveClient)
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
    from dcwb import cli

    def fake_highlight_day(**kw):
        raise FileNotFoundError("missing RecentClips/2026-05-08")

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr("dcwb.cli.highlight_day", fake_highlight_day)

    rc = main(["highlight-day", "--source", str(tmp_path), "--date", "2026-05-08"])

    assert rc == 1
    assert "missing RecentClips/2026-05-08" in capsys.readouterr().err


def test_highlight_day_hard_errors_when_vlm_unavailable(tmp_path, monkeypatch, capsys):
    from dcwb import cli
    from dcwb.vlm import VlmUnavailableError

    class DeadClient:
        def __init__(self, config): self.config = config
        def health_check(self): raise VlmUnavailableError("refused")

    called = {"n": 0}
    def fake_highlight_day(**kwargs):
        called["n"] += 1
    monkeypatch.setattr(cli, "VlmClient", DeadClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 1
    assert called["n"] == 0
    assert "VLM unavailable" in capsys.readouterr().err


def test_highlight_day_falls_back_with_allow_no_ai(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.vlm import VlmUnavailableError
    from dcwb.highlight import HighlightResult

    class DeadClient:
        def __init__(self, config): self.config = config
        def health_check(self): raise VlmUnavailableError("refused")

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", DeadClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--allow-no-ai",
        "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 0
    assert captured["vlm_client"] is None
    assert captured["selection"] == "mvp-fallback"


def test_highlight_day_passes_client_when_healthy(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--vlm-model", "custom-vlm",
        "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 0
    assert isinstance(captured["vlm_client"], LiveClient)
    assert captured["vlm_client"].config.model == "custom-vlm"


def test_highlight_day_passes_white_balance_args(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    cfg = tmp_path / "pipeline.json"
    cfg.write_text('{"awb": {"gain_min": 0.6, "night_attenuation": 0.4}}')
    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--profiles-dir", str(tmp_path / "profs"),
        "--pipeline-config", str(cfg),
    ])

    assert rc == 0
    assert captured["white_balance"] is True
    assert captured["profiles_dir"] == (tmp_path / "profs").resolve()
    assert captured["awb_cfg"]["gain_min"] == 0.6


def test_highlight_day_no_look_flag(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    cfg = tmp_path / "pipeline.json"
    cfg.write_text('{"look": {"saturation": 1.2}}')
    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--no-look", "--pipeline-config", str(cfg),
    ])

    assert rc == 0
    assert captured["apply_look"] is False
    assert captured["look_cfg"]["saturation"] == 1.2


def test_highlight_day_no_white_balance_flag(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--no-white-balance",
        "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 0
    assert captured["white_balance"] is False


def test_highlight_day_prints_progress_to_stderr(tmp_path, monkeypatch, capsys):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    def fake_highlight_day(**kwargs):
        cb = kwargs["on_progress"]
        cb("telemetry", 5, 100, "kept=3")    # throttled (not a multiple of 25, not last)
        cb("telemetry", 100, 100, "kept=60")  # final clip always reported
        cb("vlm", 2, 4, "interest=8 海沿い")  # every VLM event reported
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)

    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    err = capsys.readouterr().err
    assert rc == 0
    assert "telemetry 5/100" not in err       # throttled out
    assert "telemetry 100/100 kept=60" in err
    assert "vlm 2/4" in err
    assert "interest=8 海沿い" in err


def _setup_match_reference(tmp_path, monkeypatch, gain=((1.2, 1.0, 0.8), 0.7, 8), total=600.0):
    from datetime import datetime, timezone
    from dcwb import cli, refmatch
    import dcwb.refmatch as R
    from dcwb.profile import Profile, CalibrationMeta
    import numpy as np

    # front profile + an overlapping RecentClips front clip
    profiles = tmp_path / "profiles"; profiles.mkdir()
    Profile.from_white_point(
        "front", np.array([200.0, 200.0, 200.0]),
        CalibrationMeta(samples_used=1, events_sampled=1, method="t",
                        calibrated_at=datetime.now(timezone.utc), samples_per_event_max=1),
    ).to_json(profiles / "front.json")
    day = tmp_path / "usb" / "RecentClips" / "2026-05-27"
    day.mkdir(parents=True)
    (day / "2026-05-27_13-00-00-front.mp4").write_bytes(b"x")

    monkeypatch.setattr(cli.insta360, "read_creation_time",
                        lambda p: datetime(2026, 5, 27, 4, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr("dcwb.ffmpeg_wrap.probe_duration", lambda p: total)

    captured = {}
    def fake_compute(reference, fronts, front_profile, **kw):
        captured["reference"] = reference
        captured["fronts"] = fronts
        captured.update(kw)
        return gain
    monkeypatch.setattr(cli, "compute_reference_gain", fake_compute)
    return profiles, day, captured


def test_match_reference_prints_gain_without_write(tmp_path, monkeypatch, capsys):
    profiles, day, captured = _setup_match_reference(tmp_path, monkeypatch)
    ref = tmp_path / "ride.insv"; ref.write_bytes(b"x")
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"awb": {"gain_min": 0.7}}))

    rc = main(["match-reference", str(ref), "--recent", "2026-05-27",
               "--source", str(tmp_path / "usb"), "--profiles-dir", str(profiles),
               "--pipeline-config", str(cfg), "--samples", "12", "--encoder", "libx264"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["reference_gain"] == [1.2, 1.0, 0.8]
    assert captured["samples"] == 12
    assert captured["fronts"][0].name == "2026-05-27_13-00-00-front.mp4"
    # without --write the config is untouched
    assert json.loads(cfg.read_text()) == {"awb": {"gain_min": 0.7}}


def test_match_reference_write_updates_pipeline_preserving_keys(tmp_path, monkeypatch, capsys):
    profiles, day, captured = _setup_match_reference(tmp_path, monkeypatch)
    ref = tmp_path / "ride.insv"; ref.write_bytes(b"x")
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"awb": {"gain_min": 0.7, "night_attenuation": 0.5},
                               "look": {"saturation": 1.1}}))

    rc = main(["match-reference", str(ref), "--recent", "2026-05-27",
               "--source", str(tmp_path / "usb"), "--profiles-dir", str(profiles),
               "--pipeline-config", str(cfg), "--write"])
    assert rc == 0
    written = json.loads(cfg.read_text())
    assert written["awb"]["reference_gain"] == [1.2, 1.0, 0.8]
    assert written["awb"]["gain_min"] == 0.7
    assert written["awb"]["night_attenuation"] == 0.5
    assert written["look"] == {"saturation": 1.1}


def test_match_reference_caps_front_window(tmp_path, monkeypatch):
    """--max-window bounds the anchor window so a long reference does not pull in
    (and concat) the whole drive's front clips."""
    profiles, day, captured = _setup_match_reference(tmp_path, monkeypatch, total=1800.0)
    (day / "2026-05-27_13-15-00-front.mp4").write_bytes(b"x")  # outside the 600s window
    ref = tmp_path / "ride.insv"; ref.write_bytes(b"x")
    rc = main(["match-reference", str(ref), "--recent", "2026-05-27",
               "--source", str(tmp_path / "usb"), "--profiles-dir", str(profiles),
               "--pipeline-config", str(tmp_path / "absent.json"), "--max-window", "600"])
    assert rc == 0
    assert [p.name for p in captured["fronts"]] == ["2026-05-27_13-00-00-front.mp4"]
    assert captured["max_window"] == 600.0


def test_match_reference_errors_when_no_front_clips(tmp_path, monkeypatch, capsys):
    profiles, day, captured = _setup_match_reference(tmp_path, monkeypatch)
    # remove the only front clip → no overlap
    (day / "2026-05-27_13-00-00-front.mp4").unlink()
    ref = tmp_path / "ride.insv"; ref.write_bytes(b"x")
    rc = main(["match-reference", str(ref), "--recent", "2026-05-27",
               "--source", str(tmp_path / "usb"), "--profiles-dir", str(profiles),
               "--pipeline-config", str(tmp_path / "absent.json")])
    assert rc == 1
    assert "front" in capsys.readouterr().err.lower()


def test_sync_insta360_parses_args(monkeypatch, tmp_path):
    from dcwb import cli
    captured = {}
    def fake_run(**kw):
        captured.update(kw); return 0
    monkeypatch.setattr(cli, "run_sync_insta360", fake_run, raising=False)
    insv = tmp_path / "VID.insv"; insv.write_bytes(b"")
    rc = cli.main(["sync-insta360", str(insv), "--recent", "2026-05-27",
                   "--insta-flat", str(tmp_path / "flat.mp4"),
                   "--encoder", "libx264"])
    assert rc == 0
    assert captured["recent"] == "2026-05-27"
    assert captured["encoder"] == "libx264"
    assert str(insv) in [str(p) for p in captured["insv"]]
    assert captured["reference_gain"] is None  # default: no colour match


def test_sync_insta360_forwards_reference_gain_flag(monkeypatch, tmp_path):
    from dcwb import cli
    captured = {}
    monkeypatch.setattr(cli, "run_sync_insta360",
                        lambda **kw: (captured.update(kw) or 0), raising=False)
    insv = tmp_path / "VID.insv"; insv.write_bytes(b"")
    rc = cli.main(["sync-insta360", str(insv), "--recent", "2026-05-27",
                   "--reference-gain", "1.02", "1.0", "0.94", "--encoder", "libx264"])
    assert rc == 0
    assert captured["reference_gain"] == (1.02, 1.0, 0.94)


def test_sync_insta360_reads_reference_gain_from_pipeline(monkeypatch, tmp_path):
    from dcwb import cli
    captured = {}
    monkeypatch.setattr(cli, "run_sync_insta360",
                        lambda **kw: (captured.update(kw) or 0), raising=False)
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"awb": {"reference_gain": [1.01, 1.0, 0.95]}}))
    insv = tmp_path / "VID.insv"; insv.write_bytes(b"")
    rc = cli.main(["sync-insta360", str(insv), "--recent", "2026-05-27",
                   "--pipeline-config", str(cfg), "--encoder", "libx264"])
    assert rc == 0
    assert captured["reference_gain"] == (1.01, 1.0, 0.95)
