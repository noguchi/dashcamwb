# RecentClips Low-Motion Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `dcwb prune-recent` CLI that detects low-motion `RecentClips` segments, quarantines them to a self-managed trash with an N-day recovery window, and purges expired trash — never touching `SentryClips`/`SavedClips`.

**Architecture:** A new `prune.py` module groups RecentClips into per-timestamp segments (6 cameras each), scores each segment's motion from a few sampled frames of the `front` camera, applies safety guards (min-age + overlap with Sentry/Saved events), and moves low-motion segments into `@dcwb_trash/` on the same QNAP share (instant rename) with a `manifest.jsonl` driving purge and restore. The CLI defaults to dry-run; `--apply` quarantines, `--purge` deletes expired, `--restore` reverses.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), NumPy, ffmpeg/ffprobe, argparse, pytest. Reuses `dcwb.ffmpeg_wrap`, `dcwb.calibrate.JST`, `dcwb.serve.index.scan_sources`.

---

## File Structure

- **Create `src/dcwb/prune.py`** — all prune logic: `Segment`/`Candidate` dataclasses, segment building, motion scoring, guards, candidate selection, quarantine/purge/restore, manifest I/O, report formatting.
- **Modify `src/dcwb/ffmpeg_wrap.py`** — add `extract_frames(path, times)` (one VideoCapture open, multiple seeks).
- **Modify `src/dcwb/cli.py`** — add the `prune-recent` subcommand + `_cmd_prune`.
- **Modify `pipeline.json`** — add a `prune` config section.
- **Modify `tests/fixtures/make_synthetic.py`** — add `make_motion_clip` (animated `testsrc2`) for high-motion fixtures.
- **Create `tests/test_prune.py`** — unit tests for scoring, segments, guards, quarantine, purge, restore.
- **Modify `tests/test_ffmpeg_wrap.py`** — tests for `extract_frames`.
- **Modify `tests/test_cli.py`** — tests for dry-run / `--apply`.
- **Modify `CLAUDE.md` and `README.md`** — document the new subcommand.

**Module contract for `prune.py` (defined in Task 2–7, referenced throughout):**

```python
DEFAULT_PRUNE_CFG = {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash",
}

@dataclass
class Segment:
    day_dir: Path
    ts: datetime          # JST-aware
    ts_str: str           # "YYYY-MM-DD_HH-MM-SS"
    clips: list[Path]     # all camera mp4s sharing ts

@dataclass
class Candidate:
    segment: Segment
    score: float

def compute_motion_score(clip: Path, frames_sampled: int) -> float
def segment_motion_score(segment: Segment, cfg: dict) -> float
def _segments_for_day(day_dir: Path) -> list[Segment]
def _overlap_intervals(usb_root: Path) -> list[tuple[datetime, datetime]]
def find_candidates(usb_root: Path, cfg: dict, now: datetime) -> list[Candidate]
def format_report(candidates: list[Candidate]) -> str
def quarantine(usb_root: Path, candidates: list[Candidate], cfg: dict, now: datetime) -> list[dict]
def purge(usb_root: Path, cfg: dict, now: datetime) -> int
def restore(usb_root: Path, cfg: dict, segment_id: str) -> int
```

**Conventions:** segment times and Sentry/Saved event times are compared as JST-aware datetimes (`dcwb.calibrate.JST`). `index.Event.start/end` are naive → convert via `.replace(tzinfo=JST)`. `quarantined_at` is stored UTC ISO. A motion score of `float("inf")` (decode failure / no analyzable camera) is never below threshold, so such segments are safely never quarantined.

---

## Task 1: `extract_frames` helper + motion fixture

**Files:**
- Modify: `src/dcwb/ffmpeg_wrap.py`
- Modify: `tests/fixtures/make_synthetic.py`
- Test: `tests/test_ffmpeg_wrap.py`

- [ ] **Step 1: Add the motion fixture**

In `tests/fixtures/make_synthetic.py`, append:

```python
def make_motion_clip(
    out_path: Path,
    duration_sec: float = 3.0,
    width: int = 320,
    height: int = 240,
) -> None:
    """Generate an mp4 with strong frame-to-frame motion (animated testsrc2)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=s={width}x{height}:d={duration_sec}:r=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_ffmpeg_wrap.py` (add imports `import numpy as np` and `from tests.fixtures.make_synthetic import make_clip, make_motion_clip` if not present):

```python
def test_extract_frames_returns_requested_count(tmp_path):
    from dcwb.ffmpeg_wrap import extract_frames
    clip = tmp_path / "m.mp4"
    make_motion_clip(clip, duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0, 1.4])
    assert len(frames) == 4
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames)


def test_extract_frames_static_clip_frames_near_identical(tmp_path):
    from dcwb.ffmpeg_wrap import extract_frames
    clip = tmp_path / "s.mp4"
    make_clip(clip, (1.0, 1.0, 1.0), duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0])
    assert len(frames) == 3
    d = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
    assert d < 2.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ffmpeg_wrap.py::test_extract_frames_returns_requested_count -v`
Expected: FAIL with `ImportError: cannot import name 'extract_frames'`.

- [ ] **Step 4: Implement `extract_frames`**

In `src/dcwb/ffmpeg_wrap.py`, add after `extract_frame`:

```python
def extract_frames(path: Path, times: list[float]) -> list[np.ndarray]:
    """Decode frames at the given timestamps (seconds) in one capture session.

    Returns RGB uint8 (H, W, 3) frames. Frames that fail to decode are skipped,
    so the result may be shorter than `times`.
    """
    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ffmpeg_wrap.py -v`
Expected: PASS (new tests green, existing unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/dcwb/ffmpeg_wrap.py tests/fixtures/make_synthetic.py tests/test_ffmpeg_wrap.py
git commit -m "feat(prune): add extract_frames helper and motion fixture"
```

---

## Task 2: Segment building (`_segments_for_day`)

**Files:**
- Create: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prune.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prune.py::test_segments_for_day_groups_by_timestamp -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dcwb.prune'`.

- [ ] **Step 3: Create `prune.py` with config, dataclasses, and segment building**

Create `src/dcwb/prune.py`:

```python
from __future__ import annotations
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2
import numpy as np

from dcwb.calibrate import JST
from dcwb.ffmpeg_wrap import probe_duration, extract_frames
from dcwb.serve.index import scan_sources, _CAM_SUFFIX_RE

DEFAULT_PRUNE_CFG = {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash",
}


@dataclass
class Segment:
    day_dir: Path
    ts: datetime
    ts_str: str
    clips: list[Path]


@dataclass
class Candidate:
    segment: Segment
    score: float


def _segments_for_day(day_dir: Path) -> list[Segment]:
    """Group a RecentClips day-dir's clips into per-timestamp segments."""
    groups: dict[str, list[Path]] = {}
    for clip in day_dir.glob("*.mp4"):
        m = _CAM_SUFFIX_RE.match(clip.name)
        if not m:
            continue
        groups.setdefault(m.group("ts"), []).append(clip)
    segs: list[Segment] = []
    for ts_str, clips in groups.items():
        ts = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=JST)
        segs.append(Segment(day_dir=day_dir, ts=ts, ts_str=ts_str, clips=sorted(clips)))
    segs.sort(key=lambda s: s.ts)
    return segs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_prune.py::test_segments_for_day_groups_by_timestamp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): per-timestamp segment grouping for RecentClips"
```

---

## Task 3: Motion scoring

**Files:**
- Modify: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prune.py::test_compute_motion_score_static_below_threshold -v`
Expected: FAIL with `ImportError: cannot import name 'compute_motion_score'`.

- [ ] **Step 3: Implement scoring in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def compute_motion_score(clip: Path, frames_sampled: int) -> float:
    """Max mean-abs luma diff between consecutive sampled frames (0-255 scale).

    Returns inf when the clip cannot be analyzed, so it is never treated as
    low-motion (fail-safe: never quarantine what we can't read).
    """
    try:
        duration = probe_duration(clip)
    except Exception:
        return float("inf")
    if frames_sampled < 2 or duration <= 0:
        return float("inf")
    times = [duration * (i + 0.5) / frames_sampled for i in range(frames_sampled)]
    frames = extract_frames(clip, times)
    if len(frames) < 2:
        return float("inf")
    grays = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64)) for f in frames]
    diffs = [
        float(np.abs(grays[i + 1].astype(np.int16) - grays[i].astype(np.int16)).mean())
        for i in range(len(grays) - 1)
    ]
    return max(diffs) if diffs else float("inf")


def segment_motion_score(segment: Segment, cfg: dict) -> float:
    """Max motion score across the configured analyzed cameras for a segment."""
    scores: list[float] = []
    for cam in cfg["cameras_analyzed"]:
        clip = next((c for c in segment.clips if c.name.endswith(f"-{cam}.mp4")), None)
        if clip is None:
            continue
        scores.append(compute_motion_score(clip, cfg["frames_sampled"]))
    if not scores:
        return float("inf")
    return max(scores)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS (all prune tests so far green).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): frame-diff motion scoring per segment"
```

---

## Task 4: Guards + candidate selection

**Files:**
- Modify: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prune.py::test_find_candidates_selects_static_segments -v`
Expected: FAIL with `ImportError: cannot import name 'find_candidates'`.

- [ ] **Step 3: Implement guards + selection in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def _overlap_intervals(usb_root: Path) -> list[tuple[datetime, datetime]]:
    """JST-aware [start, end] ranges of all SentryClips/SavedClips events."""
    sources = scan_sources(usb_root)
    intervals: list[tuple[datetime, datetime]] = []
    for src in ("SentryClips", "SavedClips"):
        for ev in sources[src]:
            intervals.append((ev.start.replace(tzinfo=JST), ev.end.replace(tzinfo=JST)))
    return intervals


def _in_intervals(ts: datetime, intervals: list[tuple[datetime, datetime]]) -> bool:
    return any(start <= ts <= end for start, end in intervals)


def find_candidates(usb_root: Path, cfg: dict, now: datetime) -> list[Candidate]:
    """Low-motion RecentClips segments that pass the min-age and overlap guards."""
    intervals = _overlap_intervals(usb_root)
    cutoff = now - timedelta(hours=cfg["min_age_hours"])
    recent_root = usb_root / "RecentClips"
    out: list[Candidate] = []
    if not recent_root.exists():
        return out
    for day_dir in sorted(recent_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for seg in _segments_for_day(day_dir):
            if seg.ts > cutoff:            # too new — protect the live buffer
                continue
            if _in_intervals(seg.ts, intervals):  # overlaps a flagged event
                continue
            score = segment_motion_score(seg, cfg)
            if score < cfg["motion_threshold"]:
                out.append(Candidate(segment=seg, score=score))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): min-age and Sentry/Saved overlap guards"
```

---

## Task 5: Quarantine + manifest

**Files:**
- Modify: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prune.py::test_quarantine_moves_files_and_writes_manifest -v`
Expected: FAIL with `ImportError: cannot import name 'quarantine'`.

- [ ] **Step 3: Implement manifest I/O + quarantine in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def _manifest_path(trash_root: Path) -> Path:
    return trash_root / "manifest.jsonl"


def _load_manifest(trash_root: Path) -> list[dict]:
    p = _manifest_path(trash_root)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _write_manifest(trash_root: Path, rows: list[dict]) -> None:
    trash_root.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r) for r in rows)
    _manifest_path(trash_root).write_text(body + ("\n" if rows else ""))


def quarantine(usb_root: Path, candidates: list[Candidate], cfg: dict, now: datetime) -> list[dict]:
    """Move each candidate segment's files into the trash and append manifest rows."""
    trash_root = usb_root / cfg["trash_dir"]
    rows = _load_manifest(trash_root)
    new_rows: list[dict] = []
    for cand in candidates:
        seg = cand.segment
        for clip in seg.clips:
            rel = clip.relative_to(usb_root)
            dest = trash_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(clip), str(dest))
            new_rows.append({
                "id": uuid.uuid4().hex,
                "segment_id": seg.ts_str,
                "original_path": rel.as_posix(),
                "trash_path": dest.relative_to(usb_root).as_posix(),
                "segment_time": seg.ts.isoformat(),
                "quarantined_at": now.astimezone(timezone.utc).isoformat(),
                "motion_score": round(cand.score, 4),
                "status": "quarantined",
            })
    _write_manifest(trash_root, rows + new_rows)
    return new_rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): quarantine segments into @dcwb_trash with manifest"
```

---

## Task 6: Purge by age

**Files:**
- Modify: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
def test_purge_deletes_expired_quarantined(tmp_path):
    from dcwb.prune import find_candidates, quarantine, purge, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    t0 = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, t0)
    quarantine(tmp_path, cands, DEFAULT_PRUNE_CFG, t0)
    n = purge(tmp_path, DEFAULT_PRUNE_CFG, now=t0 + timedelta(days=15))  # > 14d retention
    assert n == 6
    trash = tmp_path / "@dcwb_trash" / "RecentClips" / "2026-05-08"
    assert list(trash.glob("*.mp4")) == []
    from dcwb.prune import _load_manifest
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prune.py::test_purge_deletes_expired_quarantined -v`
Expected: FAIL with `ImportError: cannot import name 'purge'`.

- [ ] **Step 3: Implement purge in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def purge(usb_root: Path, cfg: dict, now: datetime) -> int:
    """Delete trash files whose quarantine is older than retention_days."""
    trash_root = usb_root / cfg["trash_dir"]
    rows = _load_manifest(trash_root)
    cutoff = now - timedelta(days=cfg["retention_days"])
    purged = 0
    for row in rows:
        if row["status"] != "quarantined":
            continue
        if datetime.fromisoformat(row["quarantined_at"]) <= cutoff:
            tp = usb_root / row["trash_path"]
            if tp.exists():
                tp.unlink()
            row["status"] = "purged"
            purged += 1
    _write_manifest(trash_root, rows)
    return purged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): purge expired trash past retention window"
```

---

## Task 7: Restore

**Files:**
- Modify: `src/dcwb/prune.py`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prune.py::test_restore_moves_files_back -v`
Expected: FAIL with `ImportError: cannot import name 'restore'`.

- [ ] **Step 3: Implement restore in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def restore(usb_root: Path, cfg: dict, segment_id: str) -> int:
    """Move quarantined files back to their original paths.

    `segment_id` selects one segment by its timestamp string, or "all".
    Collisions (original already exists) are skipped with a warning.
    """
    trash_root = usb_root / cfg["trash_dir"]
    rows = _load_manifest(trash_root)
    restored = 0
    for row in rows:
        if row["status"] != "quarantined":
            continue
        if segment_id != "all" and row["segment_id"] != segment_id:
            continue
        orig = usb_root / row["original_path"]
        tp = usb_root / row["trash_path"]
        if orig.exists():
            print(f"[prune] restore skip (exists): {row['original_path']}", file=sys.stderr)
            continue
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tp), str(orig))
        row["status"] = "restored"
        restored += 1
    _write_manifest(trash_root, rows)
    return restored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): restore quarantined segments with collision skip"
```

---

## Task 8: Report + CLI subcommand + config

**Files:**
- Modify: `src/dcwb/prune.py` (add `format_report`)
- Modify: `src/dcwb/cli.py`
- Modify: `pipeline.json`
- Test: `tests/test_prune.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing report test**

Append to `tests/test_prune.py`:

```python
def test_format_report_lists_candidates(tmp_path):
    from dcwb.prune import find_candidates, format_report, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    report = format_report(cands)
    assert "2026-05-08_00-00-00" in report
    assert "1 segment" in report


def test_format_report_empty():
    from dcwb.prune import format_report
    assert "no low-motion" in format_report([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prune.py::test_format_report_empty -v`
Expected: FAIL with `ImportError: cannot import name 'format_report'`.

- [ ] **Step 3: Implement `format_report` in `prune.py`**

Append to `src/dcwb/prune.py`:

```python
def format_report(candidates: list[Candidate]) -> str:
    if not candidates:
        return "[prune] no low-motion candidates found."
    by_day: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_day.setdefault(c.segment.day_dir.name, []).append(c)
    lines = ["[prune] low-motion candidates (dry-run unless --apply):"]
    total_files = 0
    for day in sorted(by_day):
        cands = sorted(by_day[day], key=lambda x: x.segment.ts)
        lines.append(f"  {day}: {len(cands)} segment(s)")
        for c in cands:
            n = len(c.segment.clips)
            total_files += n
            lines.append(f"    {c.segment.ts_str}  score={c.score:.2f}  files={n}")
    lines.append(f"[prune] total: {len(candidates)} segment(s), {total_files} file(s)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run report tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prune.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing CLI tests**

Append to `tests/test_cli.py` (add `from dcwb.cli import main` and `from tests.fixtures.make_synthetic import make_clip` and `from dcwb.serve.index import CAMERAS` if not present):

```python
def test_cli_prune_recent_dry_run_default(tmp_path, capsys):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"  # old → always past min-age
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
    ])
    assert rc == 0
    assert len(list(day.glob("*.mp4"))) == 6  # dry-run: nothing moved
    assert "2020-01-01_00-00-00" in capsys.readouterr().out


def test_cli_prune_recent_apply_quarantines(tmp_path):
    usb = tmp_path / "usb"
    day = usb / "RecentClips" / "2020-01-01"
    day.mkdir(parents=True)
    for cam in CAMERAS:
        make_clip(day / f"2020-01-01_00-00-00-{cam}.mp4", (1.0, 1.0, 1.0), duration_sec=1.0)
    rc = main([
        "prune-recent",
        "--source", str(usb),
        "--pipeline-config", str(tmp_path / "absent.json"),
        "--apply",
    ])
    assert rc == 0
    assert list(day.glob("*.mp4")) == []
    trash = usb / "@dcwb_trash" / "RecentClips" / "2020-01-01"
    assert len(list(trash.glob("*.mp4"))) == 6
```

- [ ] **Step 6: Run CLI tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py::test_cli_prune_recent_dry_run_default -v`
Expected: FAIL with argparse error `invalid choice: 'prune-recent'`.

- [ ] **Step 7: Add the subcommand to `cli.py`**

In `src/dcwb/cli.py`, add this import near the top imports:

```python
from dcwb import prune as prune_mod
from dcwb.calibrate import JST
from datetime import datetime
```

In `_build_parser`, add before `return p`:

```python
    pp = sub.add_parser("prune-recent", help="Quarantine low-motion RecentClips (dry-run by default)")
    pp.add_argument("--source", type=Path, default=Path("/Volumes/sentryusb"))
    pp.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    pp.add_argument("--apply", action="store_true", help="Quarantine candidates (then purge expired)")
    pp.add_argument("--purge", action="store_true", help="Delete trash past the retention window")
    pp.add_argument("--restore", metavar="SEGMENT_ID|all", help="Restore quarantined segment(s)")
    pp.add_argument("--retention-days", type=int, default=None, help="Override retention_days")
```

Add the `_cmd_prune` function (place it after `_cmd_render_all`):

```python
def _cmd_prune(args) -> int:
    usb_root = args.source.resolve()
    full = json.loads(args.pipeline_config.read_text()) if args.pipeline_config.exists() else {}
    cfg = {**prune_mod.DEFAULT_PRUNE_CFG, **full.get("prune", {})}
    if args.retention_days is not None:
        cfg["retention_days"] = args.retention_days
    now = datetime.now(JST)

    if args.restore:
        n = prune_mod.restore(usb_root, cfg, args.restore)
        print(f"[prune] restored {n} file(s)", file=sys.stderr)
        return 0

    candidates = prune_mod.find_candidates(usb_root, cfg, now)
    print(prune_mod.format_report(candidates))

    if args.apply:
        rows = prune_mod.quarantine(usb_root, candidates, cfg, now)
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] quarantined {len(rows)} file(s); purged {purged} expired", file=sys.stderr)
    elif args.purge:
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] purged {purged} expired file(s)", file=sys.stderr)
    return 0
```

In `main`, add `"prune-recent": _cmd_prune,` to the dispatch dict.

- [ ] **Step 8: Add the `prune` section to `pipeline.json`**

Edit `pipeline.json` to:

```json
{
  "awb": {
    "method": "shades_of_gray",
    "minkowski_p": 6,
    "samples_per_clip": 10,
    "saturation_high": 0.97,
    "saturation_low": 0.03,
    "gain_min": 0.7,
    "gain_max": 1.5,
    "night_attenuation": 0.5
  },
  "prune": {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash"
  }
}
```

- [ ] **Step 9: Run the full suite to verify everything passes**

Run: `.venv/bin/pytest`
Expected: PASS (all existing + new tests green).

- [ ] **Step 10: Commit**

```bash
git add src/dcwb/prune.py src/dcwb/cli.py pipeline.json tests/test_prune.py tests/test_cli.py
git commit -m "feat(prune): add dcwb prune-recent CLI (dry-run/apply/purge/restore)"
```

---

## Task 9: Document the subcommand

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md subcommand list**

In `CLAUDE.md`, find the line listing CLI subcommands:

```
`calibrate` / `render <event_dir>` / `verify <event_dir>` / `render-all --source <dir>` / `serve`。
```

Replace with:

```
`calibrate` / `render <event_dir>` / `verify <event_dir>` / `render-all --source <dir>` / `serve` / `prune-recent`。
```

Then add a short paragraph under the architecture section describing prune (one or two sentences): that `prune-recent` quarantines low-motion RecentClips segments (per-timestamp, scored from the `front` camera) into `@dcwb_trash/` with a manifest, default dry-run, `--apply`/`--purge`/`--restore`, guarded by `min_age_hours` and overlap with SentryClips/SavedClips event windows; config lives in `pipeline.json`'s `prune` section.

- [ ] **Step 2: Update README.md**

Add a section to `README.md` documenting the workflow:

```markdown
## RecentClips の整理（低モーション・クリップの隔離）

動きの少ない RecentClips を自動で隔離し、保持期間経過後に削除します。

```bash
# まずドライラン（何も消さない、候補をレポート表示）
dcwb prune-recent --source /Volumes/sentryusb

# 問題なければ隔離実行（@dcwb_trash へ移動、期限切れは同時に消去）
dcwb prune-recent --source /Volumes/sentryusb --apply

# 誤って隔離したセグメントを元に戻す
dcwb prune-recent --source /Volumes/sentryusb --restore 2026-05-08_00-00-00
# すべて戻す場合
dcwb prune-recent --source /Volumes/sentryusb --restore all
```

- SentryClips / SavedClips には一切触れません。
- 直近 48 時間のクリップ、および SentryClips/SavedClips のイベント時間に重なるクリップは保護されます。
- 隔離されたファイルは `@dcwb_trash/` に 14 日間保持され、`--apply` または `--purge` 実行時に期限切れ分が削除されます。
- 閾値・保持期間は `pipeline.json` の `prune` セクションで調整できます。
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document dcwb prune-recent subcommand"
```

---

## Tuning note (post-implementation)

`motion_threshold = 2.0` (mean-abs luma diff, 0–255 scale) is an initial guess. After implementation, run `dcwb prune-recent --source /Volumes/sentryusb` (dry-run) against real footage and inspect the printed `score=` values: confirm clear garage/static segments fall well below the threshold and any genuinely-moving clips sit above it. Adjust `pipeline.json` `prune.motion_threshold` accordingly before the first `--apply`.

---

## Self-Review

**Spec coverage:**
- New CLI `prune-recent` (dry-run/apply/purge/restore) → Task 8. ✅
- Motion judgement per timestamp segment, front camera, frame-diff → Tasks 2–4. ✅
- Quarantine whole 6-camera segment → Tasks 4–5 (candidates carry the full segment; quarantine moves all `seg.clips`). ✅
- min-age guard (48h) → Task 4. ✅
- overlap guard (Sentry/Saved windows) → Task 4. ✅
- Trash `@dcwb_trash` + `manifest.jsonl`, purge by retention, restore → Tasks 5–7. ✅
- `pipeline.json` `prune` section → Task 8. ✅
- `ffmpeg_wrap` single-pass multi-frame extraction → Task 1. ✅
- SentryClips/SavedClips untouched (test) → Task 5. ✅
- Tests for scoring/segments/guards/quarantine/purge/restore → Tasks 1–8. ✅

**Deviation from spec (intentional, scope-reducing):** The spec mentioned grouping the dry-run report by `index._group_recent_day` pseudo-events. The plan groups by day-dir + per-segment lines instead — same information, simpler, no dependency on the 10-minute grouping logic. Acceptable display-level simplification.

**Placeholder scan:** No TBD/TODO; every code step contains complete code. ✅

**Type consistency:** `Segment`/`Candidate` fields, `cfg` keys (`motion_threshold`, `frames_sampled`, `cameras_analyzed`, `min_age_hours`, `retention_days`, `trash_dir`), and function signatures match across Tasks 2–8. Manifest row keys (`id`, `segment_id`, `original_path`, `trash_path`, `segment_time`, `quarantined_at`, `motion_score`, `status`) are written in Task 5 and read identically in Tasks 6–7. ✅
