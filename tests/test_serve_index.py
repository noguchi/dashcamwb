from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pytest
from dcwb.serve.index import Event, scan_sources, RECENT_GAP_MINUTES

CAMERAS = ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater")


def _touch_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # content does not matter for index tests


def _make_event_dir(parent: Path, name: str, *, with_meta: bool = True) -> Path:
    ev = parent / name
    ev.mkdir(parents=True)
    ts = name
    for cam in CAMERAS:
        _touch_clip(ev / f"{ts}-{cam}.mp4")
    if with_meta:
        (ev / "event.json").write_text("{}")
        (ev / "thumb.png").write_bytes(b"")
    return ev


def _make_recent_clip_set(day_dir: Path, ts: str) -> None:
    for cam in CAMERAS:
        _touch_clip(day_dir / f"{ts}-{cam}.mp4")


def test_scan_sources_empty_when_root_missing(tmp_path):
    result = scan_sources(tmp_path / "does-not-exist")
    assert result == {"SentryClips": [], "SavedClips": [], "RecentClips": []}


def test_scan_sources_sentry_and_saved(tmp_path):
    _make_event_dir(tmp_path / "SentryClips", "2026-04-18_19-58-48")
    _make_event_dir(tmp_path / "SentryClips", "2026-04-20_05-42-16")
    _make_event_dir(tmp_path / "SavedClips", "2026-04-18_15-32-56")

    result = scan_sources(tmp_path)

    assert len(result["SentryClips"]) == 2
    assert len(result["SavedClips"]) == 1
    # newest first
    assert result["SentryClips"][0].name == "2026-04-20_05-42-16"
    ev = result["SentryClips"][0]
    assert ev.source == "SentryClips"
    assert len(ev.clips) == 6
    assert ev.thumb is not None and ev.thumb.name == "thumb.png"


def test_recentclips_grouping_by_gap(tmp_path):
    day = tmp_path / "RecentClips" / "2026-05-08"
    # group A: 00:00, 00:01, 00:02 (1-min spacing → same pseudo-event)
    _make_recent_clip_set(day, "2026-05-08_00-00-00")
    _make_recent_clip_set(day, "2026-05-08_00-01-00")
    _make_recent_clip_set(day, "2026-05-08_00-02-00")
    # gap > 10 min
    _make_recent_clip_set(day, "2026-05-08_00-15-00")  # group B, single clip
    # gap exactly 10 min from 00:15 → 00:25 should still split (>= boundary)
    _make_recent_clip_set(day, "2026-05-08_00-25-00")  # group C
    _make_recent_clip_set(day, "2026-05-08_00-26-00")  # still group C (1-min gap)

    result = scan_sources(tmp_path)

    events = result["RecentClips"]
    assert len(events) == 3
    # newest first
    assert events[0].name == "2026-05-08_0025"
    assert events[1].name == "2026-05-08_0015"
    assert events[2].name == "2026-05-08_0000"
    # group A has 3 timestamps × 6 cams = 18 clips
    assert len(events[2].clips) == 18
    assert len(events[1].clips) == 6
    assert len(events[0].clips) == 12
    # all events report source/path correctly
    for ev in events:
        assert ev.source == "RecentClips"
        assert ev.path == day
        assert ev.thumb is None


def test_recentclips_gap_threshold_constant(tmp_path):
    assert RECENT_GAP_MINUTES == 10
