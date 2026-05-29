from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dcwb.ffmpeg_wrap import probe_duration
from dcwb.telemetry import SegmentTelemetry, read_segment_telemetry


@dataclass(frozen=True)
class HighlightCandidate:
    clip: Path
    ts_str: str
    duration_sec: float
    telemetry: SegmentTelemetry
    low_confidence: bool = False


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
