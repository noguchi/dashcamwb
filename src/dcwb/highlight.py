from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np

from dcwb.calibrate import JST
from dcwb.ffmpeg_wrap import concat_clips, cut_clip, extract_frames, probe_duration
from dcwb.telemetry import SegmentTelemetry, read_segment_telemetry
from dcwb.vlm import PROMPT_VERSION, ClipDescription, VlmConfig, encode_frame


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


@dataclass(frozen=True)
class Excerpt:
    source: CandidateScore | AiScore
    ts_str: str
    start_sec: float
    duration_sec: float


@dataclass(frozen=True)
class StyleConfig:
    name: str
    excerpt_sec: float
    min_sec: float
    max_sec: float
    target_sec: float


STYLE_CONFIGS = {
    "fast": StyleConfig("fast", excerpt_sec=12.0, min_sec=8.0, max_sec=15.0, target_sec=180.0),
    "cruise": StyleConfig("cruise", excerpt_sec=45.0, min_sec=30.0, max_sec=60.0, target_sec=360.0),
}


@dataclass(frozen=True)
class AiScore:
    candidate: HighlightCandidate
    total: float
    description: ClipDescription
    cached: bool


@dataclass(frozen=True)
class AiScoring:
    scores: list[AiScore]
    calls: int
    cache_hits: int


def _frame_fractions(n: int) -> list[float]:
    if n <= 1:
        return [0.5]
    return [0.1 + 0.8 * i / (n - 1) for i in range(n)]


def _sample_frames_b64(clip: Path, duration_sec: float, cfg: VlmConfig) -> list[str]:
    if duration_sec <= 0:
        return []
    n = max(1, cfg.frames_per_clip)
    times = [duration_sec * f for f in _frame_fractions(n)]
    frames = extract_frames(clip, times)
    return [encode_frame(f, cfg.frame_max_edge) for f in frames]


def _cache_key(clip: Path) -> str:
    return f"{clip.name}:{clip.stat().st_mtime_ns}"


def load_vlm_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_vlm_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


_STOPPED_PENALTY = 0.3


def _interest_to_total(desc: ClipDescription) -> float:
    base = _clamp01((desc.interest or 0) / 10.0)
    if desc.drive_quality == "stopped":
        base = _clamp01(base - _STOPPED_PENALTY)
    return base


def describe_candidates(
    candidates: list[HighlightCandidate],
    vlm_client,
    source_root: Path,
    cache_path: Path,
    use_cache: bool = True,
    skips: list[dict] | None = None,
    on_progress=None,
) -> AiScoring:
    cfg = vlm_client.config
    cache = load_vlm_cache(cache_path) if use_cache else {}
    calls = 0
    cache_hits = 0
    scores: list[AiScore] = []
    total = len(candidates)

    def record_skip(clip: Path, reason: str) -> None:
        if skips is None:
            return
        try:
            src = clip.relative_to(source_root).as_posix()
        except ValueError:
            src = clip.as_posix()
        skips.append({"source_clip": src, "reason": reason})

    def report(done: int, note: str) -> None:
        if on_progress:
            on_progress(done, total, note)

    for done, cand in enumerate(candidates, start=1):
        key = _cache_key(cand.clip)
        entry = cache.get(key) if use_cache else None
        if entry and entry.get("model") == cfg.model and entry.get("prompt_version") == PROMPT_VERSION:
            desc = ClipDescription(
                interest=entry.get("interest"),
                scene_tags=list(entry.get("scene_tags") or []),
                caption=entry.get("caption", ""),
                drive_quality=entry.get("drive_quality", ""),
            )
            cache_hits += 1
            cached = True
        else:
            frames = _sample_frames_b64(cand.clip, cand.duration_sec, cfg)
            if not frames:
                record_skip(cand.clip, "no-frames")
                report(done, "skip:no-frames")
                continue
            desc = vlm_client.describe(frames)
            calls += 1
            cached = False
            if not desc.parse_failed and desc.interest is not None:
                cache[key] = {
                    "interest": desc.interest,
                    "scene_tags": desc.scene_tags,
                    "caption": desc.caption,
                    "drive_quality": desc.drive_quality,
                    "model": cfg.model,
                    "prompt_version": PROMPT_VERSION,
                }
        if desc.parse_failed or desc.interest is None:
            record_skip(cand.clip, "vlm-parse-failed")
            report(done, "skip:vlm-parse-failed")
            continue
        if desc.interest < cfg.interest_min:
            record_skip(cand.clip, "interest-below-min")
            report(done, f"skip:interest-below-min (interest={desc.interest})")
            continue
        scores.append(AiScore(candidate=cand, total=_interest_to_total(desc), description=desc, cached=cached))
        tag = ",".join(desc.scene_tags[:2])
        suffix = " (cache)" if cached else ""
        report(done, f"interest={desc.interest} {tag}{suffix}".rstrip())

    if use_cache:
        save_vlm_cache(cache_path, cache)
    return AiScoring(scores=scores, calls=calls, cache_hits=cache_hits)


def _excerpt_duration(candidate_duration: float, cfg: StyleConfig) -> float:
    candidate_duration = _finite_or_zero(candidate_duration)
    if candidate_duration <= 0:
        return 0.0
    style_duration = max(cfg.min_sec, min(cfg.max_sec, cfg.excerpt_sec))
    return min(candidate_duration, style_duration)


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


def plan_excerpts(
    scores: list[CandidateScore | AiScore],
    style: str,
    target_duration_sec: float | None = None,
) -> list[Excerpt]:
    if style not in STYLE_CONFIGS:
        raise ValueError(f"unknown highlight style: {style}")
    cfg = STYLE_CONFIGS[style]
    target = target_duration_sec if target_duration_sec is not None else cfg.target_sec
    selected: list[tuple[CandidateScore, float]] = []
    total = 0.0
    for scored in sorted(scores, key=lambda s: s.total, reverse=True):
        if scored.total <= 0:
            continue
        duration = _excerpt_duration(scored.candidate.duration_sec, cfg)
        if duration <= 0:
            continue
        if total >= target:
            break
        selected.append((scored, duration))
        total += duration
    excerpts: list[Excerpt] = []
    for scored, duration in sorted(selected, key=lambda s: s[0].candidate.ts_str):
        start = max(0.0, (scored.candidate.duration_sec - duration) / 2.0)
        excerpts.append(
            Excerpt(
                source=scored,
                ts_str=scored.candidate.ts_str,
                start_sec=start,
                duration_sec=duration,
            )
        )
    return excerpts


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
    on_progress=None,
) -> list[HighlightCandidate]:
    def record_skip(clip: Path, reason: str) -> None:
        if skips is not None:
            skips.append({"source_clip": clip.as_posix(), "reason": reason})

    total = len(clips)
    candidates: list[HighlightCandidate] = []
    for done, clip in enumerate(clips, start=1):
        try:
            duration = probe_duration(clip)
        except Exception:
            record_skip(clip, "unreadable")
            if on_progress:
                on_progress(done, total, len(candidates))
            continue
        if duration <= 0:
            record_skip(clip, "non-positive-duration")
            if on_progress:
                on_progress(done, total, len(candidates))
            continue
        telemetry = read_segment_telemetry(clip)
        if telemetry.has_sei:
            if not telemetry.drove:
                record_skip(clip, "not-driving")
                if on_progress:
                    on_progress(done, total, len(candidates))
                continue
            low_confidence = False
        else:
            if not allow_no_sei:
                record_skip(clip, "no-sei")
                if on_progress:
                    on_progress(done, total, len(candidates))
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
        if on_progress:
            on_progress(done, total, len(candidates))
    return candidates


@dataclass(frozen=True)
class HighlightResult:
    output_path: Path
    manifest_path: Path
    excerpt_paths: list[Path]
    excerpt_count: int


def score_candidates(candidates: list[HighlightCandidate]) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for candidate in candidates:
        visual = extract_visual_features(candidate.clip, candidate.duration_sec)
        scores.append(score_candidate(candidate, visual))
    return scores


def _manifest_clip(excerpt: Excerpt, source_root: Path, rendered_path: Path, selection: str) -> dict:
    scored = excerpt.source
    candidate = scored.candidate
    tel = candidate.telemetry
    return {
        "source_clip": candidate.clip.relative_to(source_root).as_posix(),
        "rendered_clip": rendered_path.name,
        "start_sec": round(excerpt.start_sec, 3),
        "duration_sec": round(excerpt.duration_sec, 3),
        "selection": selection,
        "score": round(scored.total, 4),
        "scores": {key: round(value, 4) for key, value in scored.components.items()},
        "visual": {
            "mean_luma": round(scored.visual.mean_luma, 4),
            "visual_change": round(scored.visual.visual_change, 4),
        },
        "telemetry": {
            "has_sei": tel.has_sei,
            "gear_counts": tel.gear_counts,
            "max_speed_mps": round(tel.max_speed_mps, 4),
            "avg_speed_mps": round(tel.avg_speed_mps, 4),
            "speed_delta_mps": round(tel.speed_delta_mps, 4),
        },
        "low_confidence": candidate.low_confidence,
    }


def _manifest_clip_ai(excerpt: Excerpt, source_root: Path, rendered_path: Path, model: str) -> dict:
    scored = excerpt.source  # AiScore
    candidate = scored.candidate
    tel = candidate.telemetry
    desc = scored.description
    return {
        "source_clip": candidate.clip.relative_to(source_root).as_posix(),
        "rendered_clip": rendered_path.name,
        "start_sec": round(excerpt.start_sec, 3),
        "duration_sec": round(excerpt.duration_sec, 3),
        "selection": "ai",
        "score": round(scored.total, 4),
        "ai": {
            "interest": desc.interest,
            "scene_tags": desc.scene_tags,
            "caption": desc.caption,
            "drive_quality": desc.drive_quality,
            "model": model,
            "cached": scored.cached,
        },
        "telemetry": {
            "has_sei": tel.has_sei,
            "gear_counts": tel.gear_counts,
            "max_speed_mps": round(tel.max_speed_mps, 4),
            "avg_speed_mps": round(tel.avg_speed_mps, 4),
            "speed_delta_mps": round(tel.speed_delta_mps, 4),
        },
        "low_confidence": candidate.low_confidence,
    }


def highlight_day(
    source_root: Path,
    date: str,
    out_root: Path,
    style: str = "fast",
    allow_no_sei: bool = False,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    target_duration_sec: float | None = None,
    vlm_client=None,
    use_cache: bool = True,
    selection: str = "mvp",
    on_progress=None,
) -> HighlightResult:
    def phase_progress(phase: str):
        if on_progress is None:
            return None
        return lambda done, total, note: on_progress(phase, done, total, note)

    clips = discover_day_front_clips(source_root, date)
    skips: list[dict] = []
    candidates = build_candidates(
        clips, allow_no_sei=allow_no_sei, skips=skips,
        on_progress=phase_progress("telemetry"),
    )
    day_out = out_root / date
    clip_out = day_out / "clips"
    day_out.mkdir(parents=True, exist_ok=True)
    manifest_path = day_out / f"highlight-{style}.json"
    output_path = day_out / f"highlight-{style}.mp4"

    use_ai = vlm_client is not None
    ai_meta = {}
    if use_ai:
        scoring = describe_candidates(
            candidates, vlm_client, source_root,
            cache_path=day_out / "vlm-cache.json", use_cache=use_cache, skips=skips,
            on_progress=phase_progress("vlm"),
        )
        scores = scoring.scores
        cfg = vlm_client.config
        ai_meta = {
            "ai_endpoint": cfg.endpoint,
            "ai_model": cfg.model,
            "prompt_version": PROMPT_VERSION,
            "vlm_calls": scoring.calls,
            "vlm_cache_hits": scoring.cache_hits,
        }
    else:
        scores = score_candidates(candidates)

    if not scores:
        manifest = {
            "date": date,
            "style": style,
            "source": str(source_root),
            "created_at": datetime.now(JST).isoformat(),
            "target_duration_sec": target_duration_sec or STYLE_CONFIGS[style].target_sec,
            "output": output_path.name,
            "clips": [],
            "skips": skips or [{"reason": "no eligible driving front clips"}],
            **ai_meta,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return HighlightResult(output_path, manifest_path, [], 0)

    excerpts = plan_excerpts(scores, style, target_duration_sec=target_duration_sec)
    rendered: list[Path] = []
    manifest_clips: list[dict] = []
    for idx, excerpt in enumerate(excerpts, start=1):
        rendered_path = clip_out / f"{idx:03d}-{excerpt.ts_str}.mp4"
        cut_clip(
            excerpt.source.candidate.clip, rendered_path,
            excerpt.start_sec, excerpt.duration_sec,
            encoder=encoder, bitrate_kbps=bitrate_kbps,
        )
        rendered.append(rendered_path)
        if use_ai:
            manifest_clips.append(_manifest_clip_ai(excerpt, source_root, rendered_path, cfg.model))
        else:
            manifest_clips.append(_manifest_clip(excerpt, source_root, rendered_path, selection))
    if rendered:
        concat_clips(rendered, output_path, encoder=encoder, bitrate_kbps=bitrate_kbps)
    manifest = {
        "date": date,
        "style": style,
        "source": str(source_root),
        "created_at": datetime.now(JST).isoformat(),
        "target_duration_sec": target_duration_sec or STYLE_CONFIGS[style].target_sec,
        "output": output_path.name,
        "clips": manifest_clips,
        "skips": skips,
        **ai_meta,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return HighlightResult(output_path, manifest_path, rendered, len(rendered))
