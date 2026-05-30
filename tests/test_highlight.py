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


def test_build_candidates_reports_progress_per_clip(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import build_candidates
    day = tmp_path / "RecentClips" / "2026-05-08"
    c1 = _front_clip(day, "2026-05-08_00-00-00")
    c2 = _front_clip(day, "2026-05-08_00-01-00")
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 4.0, 10),
    )

    events = []
    build_candidates(
        [c1, c2], allow_no_sei=False,
        on_progress=lambda done, total, kept: events.append((done, total, kept)),
    )

    assert events == [(1, 2, 1), (2, 2, 2)]


def test_describe_candidates_reports_progress_per_clip(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import HighlightCandidate, describe_candidates
    from dcwb.vlm import ClipDescription
    from tests.fixtures.make_synthetic import make_motion_clip

    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    a = day / "2026-05-08_00-00-00-front.mp4"
    b = day / "2026-05-08_00-01-00-front.mp4"
    make_motion_clip(a, duration_sec=1.0)
    make_motion_clip(b, duration_sec=1.0)
    cands = [
        HighlightCandidate(a, "2026-05-08_00-00-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
        HighlightCandidate(b, "2026-05-08_00-01-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
    ]
    descs = {a.name: ClipDescription(8, ["海沿い"], "海", "flowing"), b.name: ClipDescription(5, [], "街", "flowing")}
    client = _FakeVlmClient({}, _ai_config())
    def sample_stub(clip, duration, cfg):
        client._next_name = clip.name
        client.by_name = descs
        return ["uri"]
    monkeypatch.setattr(highlight, "_sample_frames_b64", sample_stub)

    events = []
    describe_candidates(
        cands, client, source_root=tmp_path, cache_path=tmp_path / "c.json", use_cache=False,
        on_progress=lambda done, total, note: events.append((done, total, note)),
    )

    assert [(d, t) for d, t, _ in events] == [(1, 2), (2, 2)]


def test_highlight_day_emits_phase_tagged_progress(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from dcwb.vlm import ClipDescription
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    clip = day / "2026-05-08_00-00-00-front.mp4"
    make_motion_clip(clip, duration_sec=2.0)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    class FakeClient:
        config = _ai_config()
        def health_check(self): return None
        def describe(self, frames_b64): return ClipDescription(8, ["海沿い"], "海", "flowing")
    monkeypatch.setattr(highlight, "_sample_frames_b64", lambda c, d, cfg: ["uri"])

    phases = []
    highlight_day(
        source_root=source, date="2026-05-08", out_root=tmp_path / "h",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=FakeClient(),
        on_progress=lambda phase, done, total, note: phases.append(phase),
    )

    assert "telemetry" in phases
    assert "vlm" in phases


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


def test_plan_excerpts_never_exceeds_source_duration(tmp_path):
    from dcwb.highlight import plan_excerpts
    scores = [_scored_candidate(tmp_path, "2026-05-08_00-00-00", 0.9, duration=5.0)]

    excerpts = plan_excerpts(scores, "fast")

    assert len(excerpts) == 1
    assert excerpts[0].duration_sec == 5.0
    assert excerpts[0].start_sec == 0.0


def test_highlight_day_writes_video_and_manifest(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    for idx in range(2):
        make_motion_clip(day / f"2026-05-08_00-0{idx}-00-front.mp4", duration_sec=2.0)
    monkeypatch.setattr(
        highlight,
        "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    result = highlight_day(
        source_root=source,
        date="2026-05-08",
        out_root=tmp_path / "highlights",
        style="fast",
        allow_no_sei=False,
        encoder="libx264",
        bitrate_kbps=1000,
        target_duration_sec=1.0,
    )

    assert result.output_path.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["date"] == "2026-05-08"
    assert manifest["style"] == "fast"
    assert manifest["clips"]
    assert manifest["clips"][0]["scores"]


def test_highlight_day_no_eligible_clips_writes_empty_manifest(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    make_clip(day / "2026-05-08_00-00-00-front.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    monkeypatch.setattr(
        highlight,
        "read_segment_telemetry",
        lambda p: SegmentTelemetry(False, 0, {}, False, 0.0),
    )

    result = highlight_day(
        source_root=source,
        date="2026-05-08",
        out_root=tmp_path / "highlights",
        style="fast",
        allow_no_sei=False,
        encoder="libx264",
        bitrate_kbps=1000,
    )

    assert result.excerpt_count == 0
    assert not result.output_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clips"] == []
    assert manifest["skips"][0]["reason"] == "no-sei"


class _FakeVlmClient:
    """Returns canned ClipDescriptions keyed by clip filename; counts calls."""

    def __init__(self, by_name, config):
        self.by_name = by_name
        self.config = config
        self.calls = 0
        self._next_name = ""

    def health_check(self):
        return None

    def describe(self, frames_b64):
        self.calls += 1
        return self.by_name[self._next_name]


def _ai_config(**kwargs):
    from dcwb.vlm import VlmConfig
    return VlmConfig.from_dict({"frames_per_clip": 2, "frame_max_edge": 64, **kwargs})


def test_describe_candidates_scores_by_interest_and_caches(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import HighlightCandidate, describe_candidates
    from dcwb.vlm import ClipDescription

    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    from tests.fixtures.make_synthetic import make_motion_clip
    clip_hi = day / "2026-05-08_00-00-00-front.mp4"
    clip_lo = day / "2026-05-08_00-01-00-front.mp4"
    make_motion_clip(clip_hi, duration_sec=1.0)
    make_motion_clip(clip_lo, duration_sec=1.0)
    cands = [
        HighlightCandidate(clip_hi, "2026-05-08_00-00-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
        HighlightCandidate(clip_lo, "2026-05-08_00-01-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
    ]
    descs = {
        clip_hi.name: ClipDescription(9, ["海沿い"], "海沿いを流す", "flowing"),
        clip_lo.name: ClipDescription(2, ["渋滞"], "渋滞", "stop_and_go"),
    }

    client = _FakeVlmClient({}, _ai_config())
    real_sample = highlight._sample_frames_b64
    def tracking_sample(clip, duration, cfg):
        client._next_name = clip.name
        client.by_name = descs
        return real_sample(clip, duration, cfg)
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)

    cache_path = tmp_path / "vlm-cache.json"
    skips = []
    result = describe_candidates(cands, client, source_root=tmp_path, cache_path=cache_path, use_cache=True, skips=skips)

    assert result.calls == 2
    assert [s.candidate.clip.name for s in sorted(result.scores, key=lambda s: s.total, reverse=True)][0] == clip_hi.name
    assert cache_path.exists()

    client2 = _FakeVlmClient(descs, _ai_config())
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)
    result2 = describe_candidates(cands, client2, source_root=tmp_path, cache_path=cache_path, use_cache=True, skips=[])
    assert client2.calls == 0
    assert result2.cache_hits == 2


def test_describe_candidates_records_skips(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import HighlightCandidate, describe_candidates
    from dcwb.vlm import ClipDescription

    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    from tests.fixtures.make_synthetic import make_motion_clip
    bad = day / "2026-05-08_00-00-00-front.mp4"
    low = day / "2026-05-08_00-01-00-front.mp4"
    make_motion_clip(bad, duration_sec=1.0)
    make_motion_clip(low, duration_sec=1.0)
    cands = [
        HighlightCandidate(bad, "2026-05-08_00-00-00", 1.0, SegmentTelemetry(True, 1, {"DRIVE": 1}, True, 1.0, 1.0, 0.0, 1)),
        HighlightCandidate(low, "2026-05-08_00-01-00", 1.0, SegmentTelemetry(True, 1, {"DRIVE": 1}, True, 1.0, 1.0, 0.0, 1)),
    ]
    descs = {
        bad.name: ClipDescription(None, [], "", "", parse_failed=True),
        low.name: ClipDescription(0, [], "暗い", "stopped"),
    }
    client = _FakeVlmClient({}, _ai_config(interest_min=1))
    # stub sampling to return a dummy frame list and route describe() to this clip
    def sample_stub(clip, duration, cfg):
        client._next_name = clip.name
        client.by_name = descs
        return ["uri"]
    monkeypatch.setattr(highlight, "_sample_frames_b64", sample_stub)

    skips = []
    result = describe_candidates(cands, client, source_root=tmp_path, cache_path=tmp_path / "c.json", use_cache=False, skips=skips)

    assert result.scores == []
    assert not (tmp_path / "c.json").exists()
    reasons = {s["reason"] for s in skips}
    assert "vlm-parse-failed" in reasons
    assert "interest-below-min" in reasons


def test_highlight_day_ai_path_writes_manifest_with_ai_block(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from dcwb.vlm import ClipDescription
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    names = []
    for idx in range(2):
        clip = day / f"2026-05-08_00-0{idx}-00-front.mp4"
        make_motion_clip(clip, duration_sec=2.0)
        names.append(clip.name)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    descs = {
        names[0]: ClipDescription(9, ["海沿い", "夕焼け"], "夕暮れの海岸を流す", "flowing"),
        names[1]: ClipDescription(7, ["市街地"], "街中を抜ける", "flowing"),
    }

    class FakeClient:
        config = _ai_config()
        def __init__(self):
            self.calls = 0
            self._name = ""
        def health_check(self): return None
        def describe(self, frames_b64):
            self.calls += 1
            return descs[self._name]

    client = FakeClient()
    real_sample = highlight._sample_frames_b64
    def tracking_sample(clip, duration, cfg):
        client._name = clip.name
        return real_sample(clip, duration, cfg)
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)

    result = highlight_day(
        source_root=source, date="2026-05-08", out_root=tmp_path / "highlights",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=client, use_cache=True,
    )

    assert result.output_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["ai_model"] == "qwen2.5-vl-7b-instruct"
    assert manifest["prompt_version"] == "2"
    assert manifest["vlm_calls"] == 2
    clip0 = manifest["clips"][0]
    assert clip0["selection"] == "ai"
    assert clip0["ai"]["interest"] in (7, 9)
    assert clip0["ai"]["scene_tags"]
    assert "scores" not in clip0
    assert "visual" not in clip0
    assert (tmp_path / "highlights" / "2026-05-08" / "vlm-cache.json").exists()


def test_wb_attenuation_day_is_identity_night_is_attenuated():
    from dcwb.highlight import _wb_attenuation
    # ~13:00 JST late May in Tokyo is daytime; ~02:00 is night.
    assert _wb_attenuation("2026-05-27_13-00-00", night_attenuation=0.5) == 1.0
    assert _wb_attenuation("2026-05-27_02-00-00", night_attenuation=0.5) == 0.5
    # unparseable timestamp falls back to no attenuation (identity)
    assert _wb_attenuation("not-a-timestamp", night_attenuation=0.5) == 1.0


def _write_front_profile(profiles_dir, gain_r=1.2, gain_b=0.8):
    import json
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "front.json").write_text(json.dumps({
        "camera": "front", "gain_r": gain_r, "gain_g": 1.0, "gain_b": gain_b,
        "matrix_3x3": [[gain_r, 0, 0], [0, 1.0, 0], [0, 0, gain_b]],
        "calibration": {
            "samples_used": 1, "events_sampled": 1, "method": "test",
            "calibrated_at": "2026-05-01T00:00:00+09:00", "samples_per_event_max": 1,
        },
    }))


def test_highlight_day_applies_white_balance_and_records_manifest(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-27"
    day.mkdir(parents=True)
    make_motion_clip(day / "2026-05-27_13-00-00-front.mp4", duration_sec=2.0)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )
    profiles_dir = tmp_path / "profiles"
    _write_front_profile(profiles_dir)

    result = highlight_day(
        source_root=source, date="2026-05-27", out_root=tmp_path / "h",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=None,
        profiles_dir=profiles_dir, white_balance=True,
    )

    assert result.output_path.exists()
    wb = json.loads(result.manifest_path.read_text())["clips"][0]["white_balance"]
    assert wb["applied"] is True
    assert len(wb["scene_gain"]) == 3
    assert len(wb["final_matrix"]) == 3


def test_highlight_day_white_balance_off_records_not_applied(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-27"
    day.mkdir(parents=True)
    make_motion_clip(day / "2026-05-27_13-00-00-front.mp4", duration_sec=2.0)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )
    profiles_dir = tmp_path / "profiles"
    _write_front_profile(profiles_dir)

    result = highlight_day(
        source_root=source, date="2026-05-27", out_root=tmp_path / "h",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=None,
        profiles_dir=profiles_dir, white_balance=False,
    )

    assert json.loads(result.manifest_path.read_text())["clips"][0]["white_balance"]["applied"] is False


def test_highlight_day_mvp_path_marks_selection(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    make_motion_clip(day / "2026-05-08_00-00-00-front.mp4", duration_sec=2.0)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    result = highlight_day(
        source_root=source, date="2026-05-08", out_root=tmp_path / "highlights",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=None, selection="mvp-fallback",
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clips"][0]["selection"] == "mvp-fallback"
    assert manifest["clips"][0]["scores"]
