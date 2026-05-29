from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pytest

from dcwb.calibrate import JST
from dcwb.serve.index import CAMERAS
from tests.fixtures.make_synthetic import make_clip, make_motion_clip


def test_segments_for_day_groups_by_timestamp(tmp_path):
    from dcwb.prune import _segments_for_day
    day = tmp_path / "2026-05-08"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        (day / f"2026-05-08_00-00-00-{cam}.mp4").write_bytes(b"")
        (day / f"2026-05-08_00-01-00-{cam}.mp4").write_bytes(b"")
    segs = _segments_for_day(day)
    assert len(segs) == 2
    assert {s.ts_str for s in segs} == {"2026-05-08_00-00-00", "2026-05-08_00-01-00"}
    assert all(len(s.clips) == 6 for s in segs)
    assert segs[0].ts.tzinfo == JST
    # sorted ascending by time
    assert segs[0].ts_str == "2026-05-08_00-00-00"


def test_compute_motion_score_static_below_threshold(tmp_path):
    from dcwb.prune import compute_motion_score
    clip = tmp_path / "2026-05-08_00-00-00-front.mp4"
    make_clip(clip, (1.0, 1.0, 1.0), duration_sec=1.0)
    assert compute_motion_score(clip, 8) < 2.0


def test_compute_motion_score_motion_above_threshold(tmp_path):
    from dcwb.prune import compute_motion_score
    clip = tmp_path / "2026-05-08_00-00-00-front.mp4"
    make_motion_clip(clip, duration_sec=1.0)
    assert compute_motion_score(clip, 8) > 2.0


def test_compute_motion_score_missing_clip_returns_inf(tmp_path):
    from dcwb.prune import compute_motion_score
    assert compute_motion_score(tmp_path / "nope.mp4", 8) == float("inf")


def test_segment_motion_score_uses_analyzed_camera(tmp_path):
    from dcwb.prune import _segments_for_day, segment_motion_score, DEFAULT_PRUNE_CFG
    day = tmp_path / "2026-05-08"
    day.mkdir(parents=True)
    make_clip(day / "2026-05-08_00-00-00-front.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    make_clip(day / "2026-05-08_00-00-00-back.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    seg = _segments_for_day(day)[0]
    assert segment_motion_score(seg, DEFAULT_PRUNE_CFG) < 2.0


def test_segment_motion_score_ignores_non_analyzed_camera(tmp_path):
    from dcwb.prune import _segments_for_day, segment_motion_score, DEFAULT_PRUNE_CFG
    day = tmp_path / "2026-05-08"
    day.mkdir(parents=True)
    # front is static (low motion); back has heavy motion but is NOT analyzed
    make_clip(day / "2026-05-08_00-00-00-front.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    make_motion_clip(day / "2026-05-08_00-00-00-back.mp4", duration_sec=1.0)
    seg = _segments_for_day(day)[0]
    # cameras_analyzed == ["front"], so the motion on `back` must be ignored
    assert segment_motion_score(seg, DEFAULT_PRUNE_CFG) < 2.0


def _make_static_segment(day: Path, ts: str) -> None:
    day.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        make_clip(day / f"{ts}-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)


def test_find_candidates_selects_static_segments(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)  # well past min-age
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1
    assert cands[0].segment.ts_str == "2026-05-08_00-00-00"
    assert cands[0].score < DEFAULT_PRUNE_CFG["motion_threshold"]


def test_min_age_guard_skips_recent_segments(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 8, 1, 0, tzinfo=JST)  # only 1h later (< 48h)
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_overlap_guard_skips_sentry_window(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    sev = tmp_path / "SentryClips" / "2026-05-08_00-00-00"
    sev.mkdir(parents=True)
    for cam in CAMERAS:
        (sev / f"2026-05-08_00-00-00-{cam}.mp4").write_bytes(b"")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_min_age_boundary_segment_at_cutoff_included(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    # now is exactly min_age_hours (48h) after the segment → ts == cutoff → included
    now = datetime(2026, 5, 10, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1


def test_overlap_boundary_segment_at_event_end_protected(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-01-00")
    # SentryClips event at 00:00:00 → Event.end == 00:01:00 (max-ts + 1 min)
    sev = tmp_path / "SentryClips" / "2026-05-08_00-00-00"
    sev.mkdir(parents=True)
    for cam in CAMERAS:
        (sev / f"2026-05-08_00-00-00-{cam}.mp4").write_bytes(b"")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    # segment ts == event end → inclusive overlap → protected
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_quarantine_moves_files_and_writes_manifest(tmp_path):
    from dcwb.prune import find_candidates, quarantine, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    rows = quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, now)
    # originals moved out
    assert list(day.glob("*.mp4")) == []
    # landed in trash, structure preserved
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert len(list(trash.glob("*.mp4"))) == 6
    # manifest: 6 rows, one per file, all quarantined, shared segment_id
    assert len(rows) == 6
    assert all(r["status"] == "quarantined" for r in rows)
    assert all(r["segment_id"] == "2026-05-08_00-00-00" for r in rows)
    manifest = tmp_path / "@dcwb_trash" / "manifest.jsonl"
    assert manifest.exists()
    assert len([ln for ln in manifest.read_text().splitlines() if ln.strip()]) == 6


def test_quarantine_leaves_sentryclips_untouched(tmp_path):
    from dcwb.prune import find_candidates, quarantine, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    # A *separate-time* Sentry event so it does not trigger the overlap guard
    sev = tmp_path / "SentryClips" / "2026-05-09_12-00-00"
    sev.mkdir(parents=True)
    for cam in CAMERAS:
        (sev / f"2026-05-09_12-00-00-{cam}.mp4").write_bytes(b"x")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, now)
    assert len(list(sev.glob("*.mp4"))) == 6  # Sentry files all present


def test_quarantine_skips_when_already_in_trash(tmp_path):
    from dcwb.prune import find_candidates, quarantine, _load_manifest, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now), DEFAULT_PRUNE_CFG, now)
    # re-create sources at the same paths (simulate re-run / Tesla name reuse)
    _make_static_segment(day, "2026-05-08_00-00-00")
    rows2 = quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now), DEFAULT_PRUNE_CFG, now)
    assert rows2 == []                                 # all dests existed → skipped
    assert len(list(day.glob("*.mp4"))) == 6           # sources untouched (not lost)
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert len(list(trash.glob("*.mp4"))) == 6         # trash not duplicated/overwritten
    assert len(_load_manifest(tmp_path / "@dcwb_trash")) == 6


def test_quarantine_accumulates_manifest_across_calls(tmp_path):
    from dcwb.prune import find_candidates, quarantine, _load_manifest, DEFAULT_PRUNE_CFG
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    day1 = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day1, "2026-05-08_00-00-00")
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now), DEFAULT_PRUNE_CFG, now)
    day2 = tmp_path / "RecentClips" / "2026-05-09"
    _make_static_segment(day2, "2026-05-09_00-00-00")
    rows2 = quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now), DEFAULT_PRUNE_CFG, now)
    assert len(rows2) == 6                              # only new rows returned
    assert len(_load_manifest(tmp_path / "@dcwb_trash")) == 12  # accumulated existing+new


def test_purge_deletes_expired_quarantined(tmp_path):
    from dcwb.prune import find_candidates, quarantine, purge, _load_manifest, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, t0)
    n = purge(tmp_path, DEFAULT_PRUNE_CFG, now=t0 + timedelta(days=15))  # > 14d retention
    assert n == 6
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert list(trash.glob("*.mp4")) == []
    rows = _load_manifest(tmp_path / "@dcwb_trash")
    assert all(r["status"] == "purged" for r in rows)


def test_purge_keeps_fresh_quarantined(tmp_path):
    from dcwb.prune import find_candidates, quarantine, purge, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, t0)
    n = purge(tmp_path, DEFAULT_PRUNE_CFG, now=t0 + timedelta(days=1))  # < 14d
    assert n == 0
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert len(list(trash.glob("*.mp4"))) == 6


def test_purge_already_missing_trash_file_still_purges_manifest(tmp_path):
    from dcwb.prune import find_candidates, quarantine, purge, _load_manifest, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0), DEFAULT_PRUNE_CFG, t0)
    # simulate external cleanup: delete trash files before purge runs
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    for f in trash.glob("*.mp4"):
        f.unlink()
    n = purge(tmp_path, DEFAULT_PRUNE_CFG, now=t0 + timedelta(days=15))
    assert n == 6  # still flips manifest status even though files were already gone
    rows = _load_manifest(tmp_path / "@dcwb_trash")
    assert all(r["status"] == "purged" for r in rows)


def test_purge_boundary_exactly_retention_days_purges(tmp_path):
    from dcwb.prune import find_candidates, quarantine, purge, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0), DEFAULT_PRUNE_CFG, t0)
    # exactly retention_days later → quarantined_at == cutoff → <= is inclusive → purged
    n = purge(tmp_path, DEFAULT_PRUNE_CFG, now=t0 + timedelta(days=14))
    assert n == 6


def test_restore_moves_files_back(tmp_path):
    from dcwb.prune import find_candidates, quarantine, restore, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, t0)
    n = restore(tmp_path, DEFAULT_PRUNE_CFG, "2026-05-08_00-00-00")
    assert n == 6
    assert len(list(day.glob("*.mp4"))) == 6
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert list(trash.glob("*.mp4")) == []


def test_restore_skips_on_collision(tmp_path):
    from dcwb.prune import find_candidates, quarantine, restore, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, t0)
    # recreate one original so restore must skip it
    (day / "2026-05-08_00-00-00-front.mp4").write_bytes(b"collision")
    n = restore(tmp_path, DEFAULT_PRUNE_CFG, "all")
    assert n == 5  # 6 minus the colliding front
    assert (day / "2026-05-08_00-00-00-front.mp4").read_bytes() == b"collision"


def test_restore_skips_when_trash_file_missing(tmp_path):
    from dcwb.prune import find_candidates, quarantine, restore, _load_manifest, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0), DEFAULT_PRUNE_CFG, t0)
    # externally remove the trash files before restore runs
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    for f in trash.glob("*.mp4"):
        f.unlink()
    n = restore(tmp_path, DEFAULT_PRUNE_CFG, "all")  # must not raise
    assert n == 0
    rows = _load_manifest(tmp_path / "@dcwb_trash")
    assert all(r["status"] == "quarantined" for r in rows)  # left intact for a later retry


def test_restore_unknown_segment_id_returns_zero(tmp_path):
    from dcwb.prune import find_candidates, quarantine, restore, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0), DEFAULT_PRUNE_CFG, t0)
    assert restore(tmp_path, DEFAULT_PRUNE_CFG, "no-such-id") == 0


def test_format_report_lists_candidates(tmp_path):
    from dcwb.prune import find_candidates, format_report, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    report = format_report(cands)
    assert "2026-05-08_00-00-00" in report
    assert "1 segment(s)" in report


def test_format_report_empty():
    from dcwb.prune import format_report
    assert "no low-motion" in format_report([])


def test_overlap_guard_protects_segment_whose_tail_touches_event_start(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-04-00")  # spans 00:04:00–00:05:00
    # Sentry event starts at 00:05:00 → the segment's tail overlaps the event's first moment
    sev = tmp_path / "SentryClips" / "2026-05-08_00-05-00"
    sev.mkdir(parents=True)
    for cam in CAMERAS:
        (sev / f"2026-05-08_00-05-00-{cam}.mp4").write_bytes(b"")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_overlap_guard_allows_segment_with_gap_before_event(tmp_path):
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-03-00")  # spans 00:03:00–00:04:00
    # event [00:05:00, 00:06:00] — a clear 1-min gap, must NOT over-protect
    sev = tmp_path / "SentryClips" / "2026-05-08_00-05-00"
    sev.mkdir(parents=True)
    for cam in CAMERAS:
        (sev / f"2026-05-08_00-05-00-{cam}.mp4").write_bytes(b"")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    assert len(find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)) == 1


def test_telemetry_drove_protects_static_segment(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")  # low pixel motion
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 100, {"DRIVE": 100}, True, 12.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_telemetry_parked_sei_is_candidate(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 100, {"PARK": 100}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1
    assert cands[0].reason == "parked-sei"
    assert cands[0].gear_counts == {"PARK": 100}


def test_no_sei_falls_back_to_motion(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")  # low motion -> candidate
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(False, 0, {}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1
    assert cands[0].reason == "low-motion"


def test_use_telemetry_false_ignores_gear(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    def _boom(f):
        raise AssertionError("telemetry must not be read when use_telemetry is false")
    monkeypatch.setattr(prune, "read_segment_telemetry", _boom)
    cfg = {**DEFAULT_PRUNE_CFG, "use_telemetry": False}
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, cfg, now)
    assert len(cands) == 1
    assert cands[0].reason == "low-motion"
