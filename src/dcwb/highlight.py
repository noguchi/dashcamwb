from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from dcwb.ffmpeg_wrap import extract_frames, probe_duration
from dcwb.telemetry import SegmentTelemetry, read_segment_telemetry


@dataclass(frozen=True)
class HighlightCandidate:
    clip: Path
    ts_str: str
    duration_sec: float
    telemetry: SegmentTelemetry
    low_confidence: bool = False


@dataclass(frozen=True)
class VisualFeatures:
    mean_luma: float
    visual_change: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: HighlightCandidate
    visual: VisualFeatures
    total: float
    components: dict[str, float]


def _clamp01(value: float) -> float:
    value = _finite_or_zero(value)
    return max(0.0, min(1.0, value))


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def extract_visual_features(clip: Path, duration_sec: float, samples: int = 8) -> VisualFeatures:
    if duration_sec <= 0 or samples < 2:
        return VisualFeatures(mean_luma=0.0, visual_change=0.0)
    times = [duration_sec * (i + 0.5) / samples for i in range(samples)]
    frames = extract_frames(clip, times)
    if not frames:
        return VisualFeatures(mean_luma=0.0, visual_change=0.0)
    grays = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 64))
        for frame in frames
    ]
    mean_luma = float(np.mean([g.mean() for g in grays]))
    diffs = [
        float(np.abs(grays[i + 1].astype(np.int16) - grays[i].astype(np.int16)).mean())
        for i in range(len(grays) - 1)
    ]
    visual_change = float(np.mean(diffs)) if diffs else 0.0
    return VisualFeatures(mean_luma=mean_luma, visual_change=visual_change)


def score_candidate(candidate: HighlightCandidate, visual: VisualFeatures) -> CandidateScore:
    tel = candidate.telemetry
    avg_speed_mps = _finite_or_zero(tel.avg_speed_mps)
    speed_delta_mps = _finite_or_zero(tel.speed_delta_mps)
    mean_luma = _finite_or_zero(visual.mean_luma)
    visual_change_raw = _finite_or_zero(visual.visual_change)
    speed = _clamp01(avg_speed_mps / 22.0)
    speed_delta = _clamp01(speed_delta_mps / 8.0)
    visual_change = _clamp01(visual_change_raw / 20.0)
    brightness = _clamp01(1.0 - abs(mean_luma - 145.0) / 145.0)
    still_penalty = -0.25 if avg_speed_mps < 1.0 and visual_change_raw < 1.0 else 0.0
    dark_penalty = -0.25 * _clamp01((45.0 - mean_luma) / 45.0)
    low_confidence_penalty = -0.20 if candidate.low_confidence else 0.0
    penalty = still_penalty + dark_penalty + low_confidence_penalty
    total = _clamp01(
        0.35 * speed
        + 0.25 * speed_delta
        + 0.25 * visual_change
        + 0.15 * brightness
        + penalty
    )
    return CandidateScore(
        candidate=candidate,
        visual=visual,
        total=total,
        components={
            "speed": speed,
            "speed_delta": speed_delta,
            "visual_change": visual_change,
            "brightness": brightness,
            "still_penalty": still_penalty,
            "dark_penalty": dark_penalty,
            "low_confidence_penalty": low_confidence_penalty,
            "penalty": penalty,
        },
    )


def _timestamp_from_front_clip(clip: Path) -> str:
    suffix = "-front.mp4"
    if not clip.name.endswith(suffix):
        raise ValueError(f"not a front clip: {clip.name}")
    return clip.name[:-len(suffix)]


def discover_day_front_clips(source_root: Path, date: str) -> list[Path]:
    day_dir = source_root / "RecentClips" / date
    if not day_dir.exists():
        raise FileNotFoundError(f"missing RecentClips/{date}: {day_dir}")
    return sorted(day_dir.glob(f"{date}_*-front.mp4"))


def build_candidates(
    clips: list[Path],
    allow_no_sei: bool,
    skips: list[dict] | None = None,
) -> list[HighlightCandidate]:
    def record_skip(clip: Path, reason: str) -> None:
        if skips is not None:
            skips.append({"source_clip": clip.as_posix(), "reason": reason})

    candidates: list[HighlightCandidate] = []
    for clip in clips:
        try:
            duration = probe_duration(clip)
        except Exception:
            record_skip(clip, "unreadable")
            continue
        if duration <= 0:
            record_skip(clip, "non-positive-duration")
            continue
        telemetry = read_segment_telemetry(clip)
        if telemetry.has_sei:
            if not telemetry.drove:
                record_skip(clip, "not-driving")
                continue
            low_confidence = False
        else:
            if not allow_no_sei:
                record_skip(clip, "no-sei")
                continue
            low_confidence = True
        candidates.append(
            HighlightCandidate(
                clip=clip,
                ts_str=_timestamp_from_front_clip(clip),
                duration_sec=duration,
                telemetry=telemetry,
                low_confidence=low_confidence,
            )
        )
    return candidates
