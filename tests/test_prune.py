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
