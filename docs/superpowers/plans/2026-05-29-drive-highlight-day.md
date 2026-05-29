# Drive Highlight Day Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dcwb highlight-day` to create front-camera daily drive highlight videos from `RecentClips/<YYYY-MM-DD>` in `fast` and `cruise` styles.

**Architecture:** Implement a new focused `src/dcwb/highlight.py` module for discovery, telemetry eligibility, visual features, scoring, excerpt planning, rendering, and manifest writing. Extend telemetry with speed summary fields needed for highlight scoring while keeping existing prune behavior compatible. Add small ffmpeg helpers for excerpt cutting and concat, and keep CLI wiring thin in `src/dcwb/cli.py`.

**Tech Stack:** Python 3.11+, pytest, OpenCV via existing `cv2`, ffmpeg/ffprobe, Tesla SEI telemetry via existing vendored extractor, existing synthetic mp4 fixtures.

Spec: `docs/superpowers/specs/2026-05-29-drive-highlight-day-design.md`

---

## File Structure

- Modify `src/dcwb/telemetry.py` — add average speed, speed delta, and speed sample count to `SegmentTelemetry`.
- Modify `tests/test_telemetry.py` — verify the new speed summary fields.
- Create `src/dcwb/highlight.py` — discovery, feature extraction, scoring, excerpt selection, manifest, and orchestration.
- Create `tests/test_highlight.py` — unit and integration-style tests for the highlight module.
- Modify `src/dcwb/ffmpeg_wrap.py` — add `cut_clip` and `concat_clips` helpers.
- Modify `tests/test_ffmpeg_wrap.py` — verify excerpt cutting and concat produce playable mp4s.
- Modify `src/dcwb/cli.py` — add `highlight-day` parser and command.
- Modify `tests/test_cli.py` — CLI wiring and error tests.
- Modify `README.md` and `CLAUDE.md` — document the new workflow and development notes.

---

## Task 1: Extend Telemetry Speed Summaries

**Files:**
- Modify: `src/dcwb/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing telemetry speed test**

Append this test to `tests/test_telemetry.py`:

```python
def test_speed_summary_fields(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x00\x00"
    nals = _sei_nal(1, 0, 2.0) + _sei_nal(1, 1, 8.0) + _sei_nal(1, 2, 5.0)
    mdat = struct.pack(">I", 8 + len(nals)) + b"mdat" + nals
    clip = _write(tmp_path, "speed-summary.mp4", ftyp + mdat)

    tel = read_segment_telemetry(clip)

    assert tel.speed_sample_count == 3
    assert tel.avg_speed_mps == pytest.approx(5.0)
    assert tel.speed_delta_mps == pytest.approx(6.0)
    assert tel.max_speed_mps == pytest.approx(8.0)
```

Also add this import at the top of `tests/test_telemetry.py`:

```python
import pytest
```

- [ ] **Step 2: Run the new telemetry test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_telemetry.py::test_speed_summary_fields -v
```

Expected: FAIL with `AttributeError` for `speed_sample_count` or `avg_speed_mps`.

- [ ] **Step 3: Implement speed summary fields**

Modify `src/dcwb/telemetry.py` so `SegmentTelemetry` has defaulted fields at the end:

```python
@dataclass
class SegmentTelemetry:
    has_sei: bool
    frame_count: int
    gear_counts: dict[str, int]
    drove: bool
    max_speed_mps: float
    avg_speed_mps: float = 0.0
    speed_delta_mps: float = 0.0
    speed_sample_count: int = 0
```

Replace the body of `read_segment_telemetry` with this implementation:

```python
def read_segment_telemetry(front_clip: Path) -> SegmentTelemetry:
    """Summarize Tesla SEI gear/speed from one front clip.

    Fail-safe: any read/parse error (or no SEI) returns has_sei=False so callers
    fall back to the pixel motion path rather than crashing.
    """
    counts: dict[str, int] = {}
    frames = 0
    max_speed = 0.0
    speed_sum = 0.0
    speed_count = 0
    min_speed: float | None = None
    try:
        with open(front_clip, "rb") as fp:
            offset, size = _sx.find_mdat(fp)
            for meta in _sx.iter_sei_messages(fp, offset, size):
                frames += 1
                name = _GEAR_NAME.get(meta.gear_state, str(meta.gear_state))
                counts[name] = counts.get(name, 0) + 1
                speed = float(meta.vehicle_speed_mps or 0.0)
                speed_sum += speed
                speed_count += 1
                max_speed = max(max_speed, speed)
                min_speed = speed if min_speed is None else min(min_speed, speed)
    except Exception:
        return SegmentTelemetry(False, 0, {}, False, 0.0)
    drove = counts.get("DRIVE", 0) > 0 or counts.get("REVERSE", 0) > 0
    avg_speed = speed_sum / speed_count if speed_count else 0.0
    speed_delta = max_speed - (min_speed if min_speed is not None else max_speed)
    return SegmentTelemetry(frames > 0, frames, counts, drove, max_speed,
                            avg_speed, speed_delta, speed_count)
```

- [ ] **Step 4: Run telemetry tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_telemetry.py -v
```

Expected: PASS for all telemetry tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): expose speed summaries"
```

---

## Task 2: Add Highlight Discovery and Eligibility

**Files:**
- Create: `src/dcwb/highlight.py`
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write failing discovery and eligibility tests**

Create `tests/test_highlight.py` with:

```python
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
```

- [ ] **Step 2: Run the discovery tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py::test_discover_day_front_clips_only_returns_requested_date_front_camera tests/test_highlight.py::test_discover_day_front_clips_missing_day_errors tests/test_highlight.py::test_build_candidates_skips_no_sei_by_default tests/test_highlight.py::test_build_candidates_includes_driving_sei_clip -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dcwb.highlight'`.

- [ ] **Step 3: Implement discovery and candidate eligibility**

Create `src/dcwb/highlight.py` with:

```python
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
```

- [ ] **Step 4: Run highlight discovery tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py -v
```

Expected: PASS for the four tests created in this task.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): discover daily front drive clips"
```

---

## Task 3: Add Visual Feature Extraction and Scoring

**Files:**
- Modify: `src/dcwb/highlight.py`
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write failing feature and scoring tests**

Append to `tests/test_highlight.py`:

```python
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
```

- [ ] **Step 2: Run feature tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py::test_extract_visual_features_distinguishes_motion_from_static tests/test_highlight.py::test_score_candidate_prefers_moving_bright_changing_clip -v
```

Expected: FAIL with missing `extract_visual_features`, `VisualFeatures`, or `score_candidate`.

- [ ] **Step 3: Implement visual features and scoring**

Add imports near the top of `src/dcwb/highlight.py`:

```python
import cv2
import numpy as np

from dcwb.ffmpeg_wrap import extract_frames, probe_duration
```

Add these dataclasses and helpers:

```python
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
    return max(0.0, min(1.0, value))


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
    speed = _clamp01(tel.avg_speed_mps / 22.0)
    speed_delta = _clamp01(tel.speed_delta_mps / 8.0)
    visual_change = _clamp01(visual.visual_change / 20.0)
    brightness = _clamp01(1.0 - abs(visual.mean_luma - 145.0) / 145.0)
    still_penalty = -0.25 if tel.avg_speed_mps < 1.0 and visual.visual_change < 1.0 else 0.0
    dark_penalty = -0.25 * _clamp01((45.0 - visual.mean_luma) / 45.0)
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
            "penalty": penalty,
        },
    )
```

- [ ] **Step 4: Run highlight tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py -v
```

Expected: PASS for all current highlight tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): score drive highlight candidates"
```

---

## Task 4: Select Fast and Cruise Excerpts

**Files:**
- Modify: `src/dcwb/highlight.py`
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write failing excerpt planning tests**

Append to `tests/test_highlight.py`:

```python
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
```

- [ ] **Step 2: Run planning tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py::test_plan_excerpts_fast_uses_shorter_windows_than_cruise tests/test_highlight.py::test_plan_excerpts_preserves_chronological_order -v
```

Expected: FAIL with missing `plan_excerpts`.

- [ ] **Step 3: Implement excerpt planning**

Add to `src/dcwb/highlight.py`:

```python
@dataclass(frozen=True)
class Excerpt:
    source: CandidateScore
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


def plan_excerpts(
    scores: list[CandidateScore],
    style: str,
    target_duration_sec: float | None = None,
) -> list[Excerpt]:
    if style not in STYLE_CONFIGS:
        raise ValueError(f"unknown highlight style: {style}")
    cfg = STYLE_CONFIGS[style]
    target = target_duration_sec if target_duration_sec is not None else cfg.target_sec
    selected: list[CandidateScore] = []
    total = 0.0
    for scored in sorted(scores, key=lambda s: s.total, reverse=True):
        if scored.total <= 0:
            continue
        duration = min(cfg.max_sec, max(cfg.min_sec, min(cfg.excerpt_sec, scored.candidate.duration_sec)))
        if total >= target:
            break
        selected.append(scored)
        total += duration
    excerpts: list[Excerpt] = []
    for scored in sorted(selected, key=lambda s: s.candidate.ts_str):
        duration = min(cfg.max_sec, max(cfg.min_sec, min(cfg.excerpt_sec, scored.candidate.duration_sec)))
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
```

- [ ] **Step 4: Run highlight tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py -v
```

Expected: PASS for all current highlight tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): plan fast and cruise excerpts"
```

---

## Task 5: Add ffmpeg Excerpt Cutting and Concatenation

**Files:**
- Modify: `src/dcwb/ffmpeg_wrap.py`
- Test: `tests/test_ffmpeg_wrap.py`

- [ ] **Step 1: Write failing ffmpeg helper tests**

Append to `tests/test_ffmpeg_wrap.py`:

```python
def test_cut_clip_writes_playable_excerpt(tmp_path):
    from dcwb.ffmpeg_wrap import cut_clip, probe_duration
    from tests.fixtures.make_synthetic import make_motion_clip
    src = tmp_path / "src.mp4"
    dst = tmp_path / "cut.mp4"
    make_motion_clip(src, duration_sec=2.0)

    cut_clip(src, dst, start_sec=0.25, duration_sec=0.75, encoder="libx264", bitrate_kbps=1000)

    assert dst.exists()
    assert 0.4 <= probe_duration(dst) <= 1.2


def test_concat_clips_writes_playable_video(tmp_path):
    from dcwb.ffmpeg_wrap import concat_clips, probe_duration
    from tests.fixtures.make_synthetic import make_motion_clip
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    out = tmp_path / "joined.mp4"
    make_motion_clip(first, duration_sec=1.0)
    make_motion_clip(second, duration_sec=1.0)

    concat_clips([first, second], out, encoder="libx264", bitrate_kbps=1000)

    assert out.exists()
    assert probe_duration(out) >= 1.5
```

- [ ] **Step 2: Run ffmpeg helper tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_ffmpeg_wrap.py::test_cut_clip_writes_playable_excerpt tests/test_ffmpeg_wrap.py::test_concat_clips_writes_playable_video -v
```

Expected: FAIL with missing `cut_clip` or `concat_clips`.

- [ ] **Step 3: Implement ffmpeg helpers**

Add imports to `src/dcwb/ffmpeg_wrap.py`:

```python
import tempfile
```

Add these functions after `render_with_matrix`:

```python
def _run_ffmpeg(cmd: list[str], tmp: Path) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed: {e.stderr.decode('utf-8', errors='replace')[:500]}"
        ) from e


def cut_clip(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration_sec:.3f}",
        "-an",
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]
    _run_ffmpeg(cmd, tmp)
    tmp.replace(dst)


def concat_clips(
    clips: list[Path],
    dst: Path,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
) -> None:
    if not clips:
        raise ValueError("concat_clips requires at least one clip")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fp:
        list_path = Path(fp.name)
        for clip in clips:
            fp.write(f"file '{clip.resolve().as_posix()}'\n")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-an",
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]
    try:
        _run_ffmpeg(cmd, tmp)
        tmp.replace(dst)
    finally:
        list_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run ffmpeg helper tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_ffmpeg_wrap.py::test_cut_clip_writes_playable_excerpt tests/test_ffmpeg_wrap.py::test_concat_clips_writes_playable_video -v
```

Expected: PASS for both helper tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/ffmpeg_wrap.py tests/test_ffmpeg_wrap.py
git commit -m "feat(ffmpeg): add clip cutting and concat helpers"
```

---

## Task 6: Orchestrate Highlight Rendering and Manifest Writing

**Files:**
- Modify: `src/dcwb/highlight.py`
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write failing orchestration test**

Append to `tests/test_highlight.py`:

```python
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
```

- [ ] **Step 2: Run orchestration test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py::test_highlight_day_writes_video_and_manifest -v
```

Expected: FAIL with missing `highlight_day`.

- [ ] **Step 3: Implement orchestration and manifest writing**

Replace the existing `dcwb.ffmpeg_wrap` import in `src/dcwb/highlight.py` with:

```python
import json
from datetime import datetime

from dcwb.calibrate import JST
from dcwb.ffmpeg_wrap import concat_clips, cut_clip, extract_frames, probe_duration
```

Add result type and helpers:

```python
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


def _manifest_clip(excerpt: Excerpt, source_root: Path, rendered_path: Path) -> dict:
    scored = excerpt.source
    candidate = scored.candidate
    tel = candidate.telemetry
    return {
        "source_clip": candidate.clip.relative_to(source_root).as_posix(),
        "rendered_clip": rendered_path.name,
        "start_sec": round(excerpt.start_sec, 3),
        "duration_sec": round(excerpt.duration_sec, 3),
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


def highlight_day(
    source_root: Path,
    date: str,
    out_root: Path,
    style: str = "fast",
    allow_no_sei: bool = False,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    target_duration_sec: float | None = None,
) -> HighlightResult:
    clips = discover_day_front_clips(source_root, date)
    skips: list[dict] = []
    candidates = build_candidates(clips, allow_no_sei=allow_no_sei, skips=skips)
    day_out = out_root / date
    clip_out = day_out / "clips"
    day_out.mkdir(parents=True, exist_ok=True)
    manifest_path = day_out / f"highlight-{style}.json"
    output_path = day_out / f"highlight-{style}.mp4"
    if not candidates:
        manifest = {
            "date": date,
            "style": style,
            "source": str(source_root),
            "created_at": datetime.now(JST).isoformat(),
            "target_duration_sec": target_duration_sec or STYLE_CONFIGS[style].target_sec,
            "output": output_path.name,
            "clips": [],
            "skips": skips or [{"reason": "no eligible driving front clips"}],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return HighlightResult(output_path, manifest_path, [], 0)
    scores = score_candidates(candidates)
    excerpts = plan_excerpts(scores, style, target_duration_sec=target_duration_sec)
    rendered: list[Path] = []
    manifest_clips: list[dict] = []
    for idx, excerpt in enumerate(excerpts, start=1):
        rendered_path = clip_out / f"{idx:03d}-{excerpt.ts_str}.mp4"
        cut_clip(
            excerpt.source.candidate.clip,
            rendered_path,
            excerpt.start_sec,
            excerpt.duration_sec,
            encoder=encoder,
            bitrate_kbps=bitrate_kbps,
        )
        rendered.append(rendered_path)
        manifest_clips.append(_manifest_clip(excerpt, source_root, rendered_path))
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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return HighlightResult(output_path, manifest_path, rendered, len(rendered))
```

- [ ] **Step 4: Run highlight tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py -v
```

Expected: PASS for all highlight tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): render daily front highlights"
```

---

## Task 7: Add CLI Command

**Files:**
- Modify: `src/dcwb/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
def test_cli_highlight_day_invokes_highlight_day(tmp_path, monkeypatch, capsys):
    from dataclasses import dataclass

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
    def fake_highlight_day(**kw):
        raise FileNotFoundError("missing RecentClips/2026-05-08")

    monkeypatch.setattr("dcwb.cli.highlight_day", fake_highlight_day)

    rc = main(["highlight-day", "--source", str(tmp_path), "--date", "2026-05-08"])

    assert rc == 1
    assert "missing RecentClips/2026-05-08" in capsys.readouterr().err
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_cli.py::test_cli_highlight_day_invokes_highlight_day tests/test_cli.py::test_cli_highlight_day_missing_day_returns_error -v
```

Expected: FAIL because `highlight-day` parser or `dcwb.cli.highlight_day` does not exist.

- [ ] **Step 3: Wire the CLI**

Add import near the top of `src/dcwb/cli.py`:

```python
from dcwb.highlight import highlight_day
```

In `_build_parser`, add after `prune-recent` parser setup:

```python
    ph = sub.add_parser("highlight-day", help="Create a daily front-camera drive highlight")
    ph.add_argument("--source", type=Path, default=Path("/Volumes/sentryusb"))
    ph.add_argument("--date", required=True, help="RecentClips date directory, e.g. 2026-05-08")
    ph.add_argument("--out-root", type=Path, default=Path("highlights"))
    ph.add_argument("--style", choices=("fast", "cruise"), default="fast")
    ph.add_argument("--allow-no-sei", action="store_true")
    ph.add_argument("--encoder", default="h264_videotoolbox")
    ph.add_argument("--bitrate-kbps", type=int, default=12000)
```

Add command function before `main`:

```python
def _cmd_highlight_day(args) -> int:
    try:
        result = highlight_day(
            source_root=args.source.resolve(),
            date=args.date,
            out_root=args.out_root.resolve(),
            style=args.style,
            allow_no_sei=args.allow_no_sei,
            encoder=args.encoder,
            bitrate_kbps=args.bitrate_kbps,
        )
    except FileNotFoundError as e:
        print(f"[highlight] {e}", file=sys.stderr)
        return 1
    if result.excerpt_count == 0:
        print("[highlight] no eligible clips; wrote manifest only", file=sys.stderr)
        print(f"[highlight] manifest {result.manifest_path}", file=sys.stderr)
        return 0
    print(f"[highlight] wrote {result.output_path}", file=sys.stderr)
    print(f"[highlight] manifest {result.manifest_path}", file=sys.stderr)
    return 0
```

Add to the dispatch table:

```python
        "highlight-day": _cmd_highlight_day,
```

- [ ] **Step 4: Run CLI tests and verify green**

Run:

```bash
uv run --extra dev pytest tests/test_cli.py::test_cli_highlight_day_invokes_highlight_day tests/test_cli.py::test_cli_highlight_day_missing_day_returns_error -v
```

Expected: PASS for both CLI tests.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/cli.py tests/test_cli.py
git commit -m "feat(cli): add highlight-day command"
```

---

## Task 8: Document Highlight Workflow

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README**

Add this section after the RecentClips prune section in `README.md`:

```markdown
## Daily drive highlights

`RecentClips/<date>` から front カメラのみのハイライト動画を作れます。危険検出ではなく、見返して楽しいドライブ記録向けです。

```bash
# テンポ重視: 短い切り出しを多めにつなぐ
uv run dcwb highlight-day --source /Volumes/sentryusb --date 2026-05-08 --style fast

# ドライブ感重視: 長めの区間を少なめにつなぐ
uv run dcwb highlight-day --source /Volumes/sentryusb --date 2026-05-08 --style cruise
```

出力は `highlights/<date>/highlight-<style>.mp4` と `highlight-<style>.json` です。初期版は Tesla SEI テレメトリで DRIVE/REVERSE が確認できた front クリップだけを対象にし、速度・速度変化・画面変化・明るさからスコアを付けます。SEI が無いクリップも含めたい場合は `--allow-no-sei` を指定できますが、manifest では低信頼として記録されます。
```

- [ ] **Step 2: Update CLAUDE.md**

Add this architecture note near the existing CLI subcommand list:

```markdown
`highlight-day` は `RecentClips/<date>` の `front` カメラだけからドライブ記録向けハイライトを作る。MVP は AI なしで、SEI テレメトリの DRIVE/REVERSE、平均速度、速度変化、OpenCV の明るさ・画面変化量をスコア化し、`fast` と `cruise` の2スタイルで excerpt を選ぶ。出力 manifest には採用理由とスコア内訳を必ず残す。
```

- [ ] **Step 3: Run docs sanity checks**

Run:

```bash
rg -n "highlight-day|Daily drive highlights" README.md CLAUDE.md
git diff --check
```

Expected: `rg` prints the added sections and `git diff --check` exits 0.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document daily drive highlights"
```

---

## Task 9: Full Verification

**Files:**
- No source edits unless verification reveals a defect.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --extra dev pytest tests/test_highlight.py tests/test_telemetry.py tests/test_ffmpeg_wrap.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run:

```bash
uv run --extra dev pytest
```

Expected: PASS.

- [ ] **Step 3: Run a real dry command against an empty temp tree**

Run:

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/RecentClips"
uv run dcwb highlight-day --source "$tmp" --date 2026-05-08 --encoder libx264
```

Expected: exits 1 with a clear `[highlight] missing RecentClips/2026-05-08` message.

- [ ] **Step 4: Check repository state**

Run:

```bash
git status --short --branch
```

Expected: no unstaged or staged changes. `main` may be ahead of `origin/main`.
