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


def test_extract_visual_features_distinguishes_motion_from_static(tmp_path):
    from dcwb.highlight import extract_visual_features
    from tests.fixtures.make_synthetic import make_motion_clip
    static = tmp_path / "static.mp4"
    motion = tmp_path / "motion.mp4"
    make_clip(static, (1.0, 1.0, 1.0), duration_sec=1.0)
    make_motion_clip(motion, duration_sec=1.0)

    static_features = extract_visual_features(static, duration_sec=1.0)
    motion_features = extract_visual_features(motion, duration_sec=1.0)

    assert motion_features.visual_change > static_features.visual_change
    assert static_features.mean_luma > 0.0


def test_score_candidate_prefers_moving_bright_changing_clip(tmp_path):
    from dcwb.highlight import (
        HighlightCandidate,
        VisualFeatures,
        score_candidate,
    )
    clip = tmp_path / "2026-05-08_00-00-00-front.mp4"
    clip.write_bytes(b"not used")
    moving = HighlightCandidate(
        clip=clip,
        ts_str="2026-05-08_00-00-00",
        duration_sec=60.0,
        telemetry=SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 22.0, 16.0, 8.0, 10),
    )
    stopped = HighlightCandidate(
        clip=clip,
        ts_str="2026-05-08_00-01-00",
        duration_sec=60.0,
        telemetry=SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 0.5, 0.2, 0.1, 10),
    )

    moving_score = score_candidate(moving, VisualFeatures(mean_luma=145.0, visual_change=24.0))
    stopped_score = score_candidate(stopped, VisualFeatures(mean_luma=20.0, visual_change=0.2))

    assert moving_score.total > stopped_score.total
    assert moving_score.components["speed"] > stopped_score.components["speed"]
    assert moving_score.components["visual_change"] > stopped_score.components["visual_change"]
    assert stopped_score.components["penalty"] < 0.0
    assert stopped_score.components["still_penalty"] < 0.0
    assert stopped_score.components["dark_penalty"] < 0.0
    assert stopped_score.components["low_confidence_penalty"] == 0.0
    assert stopped_score.components["penalty"] == (
        stopped_score.components["still_penalty"]
        + stopped_score.components["dark_penalty"]
        + stopped_score.components["low_confidence_penalty"]
    )


def test_score_candidate_treats_non_finite_values_as_zero(tmp_path):
    import math
    from dcwb.highlight import HighlightCandidate, VisualFeatures, score_candidate
    clip = tmp_path / "2026-05-08_00-00-00-front.mp4"
    clip.write_bytes(b"not used")
    candidate = HighlightCandidate(
        clip=clip,
        ts_str="2026-05-08_00-00-00",
        duration_sec=60.0,
        telemetry=SegmentTelemetry(True, 10, {"DRIVE": 10}, True, math.nan, math.nan, math.inf, 10),
        low_confidence=True,
    )

    score = score_candidate(candidate, VisualFeatures(mean_luma=math.nan, visual_change=math.inf))

    assert score.components["speed"] == 0.0
    assert score.components["speed_delta"] == 0.0
    assert score.components["visual_change"] == 0.0
    assert score.components["brightness"] == 0.0
    assert score.components["low_confidence_penalty"] < 0.0
    assert 0.0 <= score.total <= 1.0


def _scored_candidate(tmp_path, ts: str, score: float, duration: float = 60.0):
    from dcwb.highlight import (
        CandidateScore,
        HighlightCandidate,
        VisualFeatures,
    )
    clip = tmp_path / f"{ts}-front.mp4"
    clip.write_bytes(b"not used")
    candidate = HighlightCandidate(
        clip=clip,
        ts_str=ts,
        duration_sec=duration,
        telemetry=SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 4.0, 10),
    )
    return CandidateScore(
        candidate=candidate,
        visual=VisualFeatures(mean_luma=145.0, visual_change=10.0),
        total=score,
        components={"speed": score, "speed_delta": 0.0, "visual_change": 0.0, "brightness": 0.0, "penalty": 0.0},
    )


def test_plan_excerpts_fast_uses_shorter_windows_than_cruise(tmp_path):
    from dcwb.highlight import plan_excerpts
    scores = [
        _scored_candidate(tmp_path, "2026-05-08_00-00-00", 0.9),
        _scored_candidate(tmp_path, "2026-05-08_00-01-00", 0.8),
    ]

    fast = plan_excerpts(scores, "fast")
    cruise = plan_excerpts(scores, "cruise")

    assert fast
    assert cruise
    assert max(e.duration_sec for e in fast) <= 15.0
    assert min(e.duration_sec for e in cruise) >= 30.0


def test_plan_excerpts_preserves_chronological_order(tmp_path):
    from dcwb.highlight import plan_excerpts
    scores = [
        _scored_candidate(tmp_path, "2026-05-08_00-02-00", 0.9),
        _scored_candidate(tmp_path, "2026-05-08_00-00-00", 0.8),
        _scored_candidate(tmp_path, "2026-05-08_00-01-00", 0.7),
    ]

    excerpts = plan_excerpts(scores, "fast", target_duration_sec=24)

    assert [e.ts_str for e in excerpts] == sorted(e.ts_str for e in excerpts)
