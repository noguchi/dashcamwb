from __future__ import annotations

from pathlib import Path

import pytest

from dcwb.telemetry import SegmentTelemetry
from tests.fixtures.make_synthetic import make_clip


def _front_clip(day: Path, ts: str) -> Path:
    day.mkdir(parents=True, exist_ok=True)
    clip = day / f"{ts}-front.mp4"
    make_clip(clip, (1.0, 1.0, 1.0), duration_sec=1.0)
    return clip


def test_discover_day_front_clips_only_returns_requested_date_front_camera(tmp_path):
    from dcwb.highlight import discover_day_front_clips
    day = tmp_path / "RecentClips" / "2026-05-08"
    other_day = tmp_path / "RecentClips" / "2026-05-09"
    front = _front_clip(day, "2026-05-08_00-00-00")
    make_clip(day / "2026-05-08_00-00-00-back.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    _front_clip(other_day, "2026-05-09_00-00-00")

    clips = discover_day_front_clips(tmp_path, "2026-05-08")

    assert clips == [front]


def test_discover_day_front_clips_missing_day_errors(tmp_path):
    from dcwb.highlight import discover_day_front_clips

    with pytest.raises(FileNotFoundError, match="RecentClips/2026-05-08"):
        discover_day_front_clips(tmp_path, "2026-05-08")


def test_build_candidates_skips_no_sei_by_default(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import build_candidates
    clip = _front_clip(tmp_path / "RecentClips" / "2026-05-08", "2026-05-08_00-00-00")
    monkeypatch.setattr(
        highlight,
        "read_segment_telemetry",
        lambda p: SegmentTelemetry(False, 0, {}, False, 0.0),
    )

    candidates = build_candidates([clip], allow_no_sei=False)

    assert candidates == []


def test_build_candidates_includes_driving_sei_clip(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import build_candidates
    clip = _front_clip(tmp_path / "RecentClips" / "2026-05-08", "2026-05-08_00-00-00")
    monkeypatch.setattr(
        highlight,
        "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 4.0, 10),
    )

    candidates = build_candidates([clip], allow_no_sei=False)

    assert len(candidates) == 1
    assert candidates[0].clip == clip
    assert candidates[0].telemetry.avg_speed_mps == 8.0
