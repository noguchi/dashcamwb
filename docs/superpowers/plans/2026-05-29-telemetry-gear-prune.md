# Telemetry (gear) based RecentClips pruning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `prune-recent` use Tesla's embedded SEI `gear_state` as the primary "was the car driving" signal, falling back to the existing pixel motion score only when SEI is absent.

**Architecture:** Vendor the official `teslamotors/dashcam` SEI extractor under `src/dcwb/vendor/tesla_dashcam/`, wrap it in a small `dcwb/telemetry.py` that turns one front clip into a gear summary, and branch in `prune.find_candidates`: SEI+drove → protect, SEI+all-park → candidate, no-SEI → motion fallback. A `use_telemetry` config flag (default true) restores pure-motion behaviour.

**Tech Stack:** Python 3.11+, `protobuf` (runtime), `grpcio-tools` (dev, to compile the proto), pytest, ffmpeg (existing synthetic-clip fixtures).

Spec: `docs/superpowers/specs/2026-05-29-telemetry-gear-prune-design.md`

---

## File Structure

- Create `src/dcwb/vendor/__init__.py` — empty package marker.
- Create `src/dcwb/vendor/tesla_dashcam/__init__.py` — empty package marker.
- Create `src/dcwb/vendor/tesla_dashcam/dashcam.proto` — official schema (fetched).
- Create `src/dcwb/vendor/tesla_dashcam/dashcam_pb2.py` — generated, committed.
- Create `src/dcwb/vendor/tesla_dashcam/sei_extractor.py` — official extractor (fetched, one import line patched).
- Create `src/dcwb/vendor/tesla_dashcam/NOTICE` — attribution.
- Create `src/dcwb/telemetry.py` — `SegmentTelemetry` + `read_segment_telemetry`.
- Create `tests/test_telemetry.py` — synthetic-SEI-mp4 tests for the parser.
- Modify `pyproject.toml` — add `protobuf` dep, `grpcio-tools` dev dep, package-data for the proto.
- Modify `src/dcwb/prune.py` — `Candidate.reason`/`gear_counts`, `_classify`, `find_candidates`, `quarantine` manifest, `format_report`, `DEFAULT_PRUNE_CFG`.
- Modify `tests/test_prune.py` — classification branch tests.
- Modify `pipeline.json` — add `use_telemetry: true`.
- Modify `README.md`, `CLAUDE.md` — document the telemetry signal.

---

## Task 1: Vendor the official SEI extractor and wire up protobuf

**Files:**
- Create: `src/dcwb/vendor/__init__.py`, `src/dcwb/vendor/tesla_dashcam/__init__.py`
- Create: `src/dcwb/vendor/tesla_dashcam/dashcam.proto`
- Create: `src/dcwb/vendor/tesla_dashcam/sei_extractor.py`
- Create: `src/dcwb/vendor/tesla_dashcam/dashcam_pb2.py` (generated)
- Create: `src/dcwb/vendor/tesla_dashcam/NOTICE`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the package markers and fetch the official files**

```bash
mkdir -p src/dcwb/vendor/tesla_dashcam
touch src/dcwb/vendor/__init__.py src/dcwb/vendor/tesla_dashcam/__init__.py
gh api repos/teslamotors/dashcam/contents/dashcam.proto --jq '.content' | base64 -d > src/dcwb/vendor/tesla_dashcam/dashcam.proto
gh api repos/teslamotors/dashcam/contents/sei_extractor.py --jq '.content' | base64 -d > src/dcwb/vendor/tesla_dashcam/sei_extractor.py
```

- [ ] **Step 2: Patch the extractor's bare import to a package-relative import**

The official `sei_extractor.py` has `import dashcam_pb2`, which won't resolve inside the package. Make it relative:

```bash
sed -i 's/^import dashcam_pb2$/from . import dashcam_pb2/' src/dcwb/vendor/tesla_dashcam/sei_extractor.py
grep -n "dashcam_pb2" src/dcwb/vendor/tesla_dashcam/sei_extractor.py
```
Expected: the import line now reads `from . import dashcam_pb2`.

- [ ] **Step 3: Write the NOTICE file**

Create `src/dcwb/vendor/tesla_dashcam/NOTICE`:

```
Vendored from https://github.com/teslamotors/dashcam
Files: dashcam.proto, sei_extractor.py (the `import dashcam_pb2` line was changed
to `from . import dashcam_pb2` for packaging). dashcam_pb2.py is generated from
dashcam.proto via grpcio-tools' protoc.

These files implement parsing of the SEI (Supplemental Enhancement Information)
telemetry that Tesla embeds in dashcam mp4 video streams (firmware 2025.44.25+,
HW3+). Reuse confirmed acceptable by the dcwb project owner (2026-05-29).
```

- [ ] **Step 4: Add dependencies to `pyproject.toml`**

In `[project].dependencies` add `protobuf`:

```toml
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.9",
  "astral>=3.2",
  "jinja2>=3.1",
  "flask>=3.0",
  "protobuf>=5,<7",
]
```

In `[project.optional-dependencies].dev` add `grpcio-tools`:

```toml
dev = [
  "pytest>=8.0",
  "pytest-mock>=3.12",
  "grpcio-tools>=1.60",
]
```

In `[tool.setuptools.package-data]` add the proto so it ships with the package:

```toml
dcwb = [
  "templates/*.j2",
  "serve/templates/*.j2",
  "serve/static/*",
  "vendor/tesla_dashcam/*.proto",
]
```

- [ ] **Step 5: Sync and generate the committed `dashcam_pb2.py`**

```bash
uv sync --extra dev
uv run python -m grpc_tools.protoc -I src/dcwb/vendor/tesla_dashcam --python_out=src/dcwb/vendor/tesla_dashcam src/dcwb/vendor/tesla_dashcam/dashcam.proto
ls -la src/dcwb/vendor/tesla_dashcam/dashcam_pb2.py
```
Expected: `dashcam_pb2.py` exists.

- [ ] **Step 6: Verify the vendored package imports and runs**

```bash
uv run python -c "from dcwb.vendor.tesla_dashcam import sei_extractor, dashcam_pb2; m = dashcam_pb2.SeiMetadata(version=1, gear_state=1); print('gear', m.gear_state, '| funcs', hasattr(sei_extractor, 'find_mdat'), hasattr(sei_extractor, 'iter_sei_messages'))"
```
Expected: `gear 1 | funcs True True` (confirms proto runtime compatibility + extractor API).

- [ ] **Step 7: Commit**

```bash
git add src/dcwb/vendor pyproject.toml uv.lock
git commit -m "feat(telemetry): vendor official Tesla SEI extractor + protobuf dep"
```

---

## Task 2: `telemetry.py` — read gear summary from a clip (TDD)

**Files:**
- Create: `src/dcwb/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_telemetry.py`. The helpers build a minimal but real SEI mp4
(`ftyp` + `mdat` of AVCC-framed NAL units) so the actual parser runs end-to-end.

```python
from __future__ import annotations
import struct
from dcwb.vendor.tesla_dashcam import dashcam_pb2


def _sei_nal(gear: int, frame_seq: int) -> bytes:
    """One AVCC-framed Tesla SEI NAL: len-prefix + 0x06 0x05 size 'BBBi' proto 0x80."""
    proto = dashcam_pb2.SeiMetadata(version=1, gear_state=gear, frame_seq_no=frame_seq).SerializeToString()
    body = b"\x06\x05" + bytes([len(proto) + 4]) + b"\x42\x42\x42\x69" + proto + b"\x80"
    return struct.pack(">I", len(body)) + body


def _build_mp4(gears, with_sei=True) -> bytes:
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x00\x00"
    if with_sei:
        nals = b"".join(_sei_nal(g, i) for i, g in enumerate(gears))
    else:
        vid = b"\x65\x88\x84\x00"  # NAL type 5 (IDR slice), not SEI -> ignored
        nals = struct.pack(">I", len(vid)) + vid
    mdat = struct.pack(">I", 8 + len(nals)) + b"mdat" + nals
    return ftyp + mdat


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_all_park_clip(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    clip = _write(tmp_path, "park.mp4", _build_mp4([0, 0, 0]))
    tel = read_segment_telemetry(clip)
    assert tel.has_sei is True
    assert tel.frame_count == 3
    assert tel.gear_counts == {"PARK": 3}
    assert tel.drove is False


def test_clip_with_drive_frames(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    clip = _write(tmp_path, "drive.mp4", _build_mp4([0, 1, 1]))
    tel = read_segment_telemetry(clip)
    assert tel.has_sei is True
    assert tel.gear_counts == {"PARK": 1, "DRIVE": 2}
    assert tel.drove is True


def test_reverse_counts_as_drove(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    clip = _write(tmp_path, "rev.mp4", _build_mp4([0, 2]))
    tel = read_segment_telemetry(clip)
    assert tel.drove is True
    assert tel.gear_counts == {"PARK": 1, "REVERSE": 1}


def test_clip_without_sei(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    clip = _write(tmp_path, "nosei.mp4", _build_mp4([], with_sei=False))
    tel = read_segment_telemetry(clip)
    assert tel.has_sei is False
    assert tel.frame_count == 0
    assert tel.drove is False


def test_missing_file_returns_no_sei(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    tel = read_segment_telemetry(tmp_path / "nope.mp4")
    assert tel.has_sei is False
    assert tel.drove is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dcwb.telemetry'`.

- [ ] **Step 3: Implement `telemetry.py`**

Create `src/dcwb/telemetry.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from dcwb.vendor.tesla_dashcam import sei_extractor as _sx

# dashcam.proto: enum Gear { PARK=0; DRIVE=1; REVERSE=2; NEUTRAL=3; }
_GEAR_NAME = {0: "PARK", 1: "DRIVE", 2: "REVERSE", 3: "NEUTRAL"}


@dataclass
class SegmentTelemetry:
    has_sei: bool
    frame_count: int
    gear_counts: dict[str, int]
    drove: bool
    max_speed_mps: float


def read_segment_telemetry(front_clip: Path) -> SegmentTelemetry:
    """Summarize Tesla SEI gear/speed from one front clip.

    Fail-safe: any read/parse error (or no SEI) returns has_sei=False so callers
    fall back to the pixel motion path rather than crashing.
    """
    counts: dict[str, int] = {}
    frames = 0
    max_speed = 0.0
    try:
        with open(front_clip, "rb") as fp:
            offset, size = _sx.find_mdat(fp)
            for meta in _sx.iter_sei_messages(fp, offset, size):
                frames += 1
                name = _GEAR_NAME.get(meta.gear_state, str(meta.gear_state))
                counts[name] = counts.get(name, 0) + 1
                if meta.vehicle_speed_mps and meta.vehicle_speed_mps > max_speed:
                    max_speed = meta.vehicle_speed_mps
    except Exception:
        return SegmentTelemetry(False, 0, {}, False, 0.0)
    drove = counts.get("DRIVE", 0) > 0 or counts.get("REVERSE", 0) > 0
    return SegmentTelemetry(frames > 0, frames, counts, drove, max_speed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_telemetry.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): read_segment_telemetry gear summary from a clip"
```

---

## Task 3: `prune.find_candidates` — gear-primary classification (TDD)

**Files:**
- Modify: `src/dcwb/prune.py` (imports, `DEFAULT_PRUNE_CFG`, `Candidate`, new `_classify`, `find_candidates`)
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
def test_telemetry_drove_protects_static_segment(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")  # low pixel motion
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 100, {"DRIVE": 100}, True, 12.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    assert find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now) == []


def test_telemetry_parked_sei_is_candidate(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 100, {"PARK": 100}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1
    assert cands[0].reason == "parked-sei"
    assert cands[0].gear_counts == {"PARK": 100}


def test_no_sei_falls_back_to_motion(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")  # low motion -> candidate
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(False, 0, {}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now)
    assert len(cands) == 1
    assert cands[0].reason == "low-motion"


def test_use_telemetry_false_ignores_gear(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, DEFAULT_PRUNE_CFG
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    def _boom(f):
        raise AssertionError("telemetry must not be read when use_telemetry is false")
    monkeypatch.setattr(prune, "read_segment_telemetry", _boom)
    cfg = {**DEFAULT_PRUNE_CFG, "use_telemetry": False}
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    cands = find_candidates(tmp_path, cfg, now)
    assert len(cands) == 1
    assert cands[0].reason == "low-motion"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_prune.py -k "telemetry or no_sei" -v`
Expected: FAIL (`read_segment_telemetry` not importable from `dcwb.prune`; `Candidate` has no `reason`).

- [ ] **Step 3: Add the import and config default in `prune.py`**

After the existing `from dcwb.serve.index import scan_sources, _CAM_SUFFIX_RE` line, add:

```python
from dcwb.telemetry import read_segment_telemetry
```

In `DEFAULT_PRUNE_CFG`, add the flag (keep all existing keys):

```python
DEFAULT_PRUNE_CFG = {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash",
    "use_telemetry": True,
}
```

- [ ] **Step 4: Extend the `Candidate` dataclass**

Replace the existing `Candidate` definition:

```python
@dataclass
class Candidate:
    segment: Segment
    score: float
    reason: str = "low-motion"
    gear_counts: dict | None = None
```

- [ ] **Step 5: Add `_classify` and rewrite the `find_candidates` inner loop**

Add this helper above `find_candidates`:

```python
def _classify(seg: Segment, cfg: dict) -> Candidate | None:
    """Gear-primary classification. None = protect (keep)."""
    if cfg.get("use_telemetry", True):
        front = next((c for c in seg.clips if c.name.endswith("-front.mp4")), None)
        if front is not None:
            tel = read_segment_telemetry(front)
            if tel.has_sei:
                if tel.drove:
                    return None  # real drive -> protect
                return Candidate(segment=seg, score=0.0, reason="parked-sei",
                                 gear_counts=tel.gear_counts)
            # SEI absent -> ambiguous -> fall through to motion
    score = segment_motion_score(seg, cfg)
    if score < cfg["motion_threshold"]:
        return Candidate(segment=seg, score=score, reason="low-motion")
    return None
```

Replace the body of the segment loop in `find_candidates` (the `score = ...` / `if score < ...` block) with:

```python
        for seg in _segments_for_day(day_dir):
            if seg.ts > cutoff:            # too new — protect the live buffer
                continue
            if _overlaps(seg.ts, seg.ts + SEGMENT_SPAN, intervals):  # overlaps a flagged event
                continue
            cand = _classify(seg, cfg)
            if cand is not None:
                out.append(cand)
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_prune.py -k "telemetry or no_sei" -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the full prune test file (regression — existing tests use no-SEI synthetic clips)**

Run: `uv run --extra dev pytest tests/test_prune.py -v`
Expected: PASS (all existing + new). Existing tests use `make_clip` mp4s which contain no Tesla SEI, so `_classify` takes the motion fallback and behaves as before.

- [ ] **Step 8: Commit**

```bash
git add src/dcwb/prune.py tests/test_prune.py
git commit -m "feat(prune): gear-primary classification with motion fallback"
```

---

## Task 4: Persist `reason` and surface it (manifest, report, config) (TDD)

**Files:**
- Modify: `src/dcwb/prune.py` (`quarantine`, `format_report`)
- Modify: `pipeline.json`
- Test: `tests/test_prune.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prune.py`:

```python
def test_quarantine_records_reason_and_gear(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, quarantine, _load_manifest, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 10, {"PARK": 10}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    rows = quarantine(tmp_path, find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now), DEFAULT_PRUNE_CFG, now)
    assert rows and all(r["reason"] == "parked-sei" for r in rows)
    assert all(r["gear_counts"] == {"PARK": 10} for r in rows)
    assert all(r["reason"] == "parked-sei" for r in _load_manifest(tmp_path / "@dcwb_trash"))


def test_format_report_shows_reason(tmp_path, monkeypatch):
    from dcwb import prune
    from dcwb.prune import find_candidates, format_report, DEFAULT_PRUNE_CFG
    from dcwb.telemetry import SegmentTelemetry
    day = tmp_path / "RecentClips" / "2026-05-08"
    _make_static_segment(day, "2026-05-08_00-00-00")
    monkeypatch.setattr(prune, "read_segment_telemetry",
                        lambda f: SegmentTelemetry(True, 10, {"PARK": 10}, False, 0.0))
    now = datetime(2026, 5, 20, 0, 0, tzinfo=JST)
    report = format_report(find_candidates(tmp_path, DEFAULT_PRUNE_CFG, now))
    assert "parked-sei" in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_prune.py -k "records_reason or shows_reason" -v`
Expected: FAIL (`KeyError: 'reason'` in manifest row; `parked-sei` absent from report).

- [ ] **Step 3: Add `reason`/`gear_counts` to the manifest row in `quarantine`**

In `quarantine`, the `new_rows.append({...})` dict gains two keys (keep all existing keys):

```python
            new_rows.append({
                "id": uuid.uuid4().hex,
                "segment_id": seg.ts_str,
                "original_path": rel.as_posix(),
                "trash_path": dest.relative_to(usb_root).as_posix(),
                "segment_time": seg.ts.isoformat(),
                "quarantined_at": now.astimezone(timezone.utc).isoformat(),
                "motion_score": round(cand.score, 4),
                "reason": cand.reason,
                "gear_counts": cand.gear_counts,
                "status": "quarantined",
            })
```

- [ ] **Step 4: Show `reason` in `format_report`**

Replace the per-candidate line in `format_report`:

```python
            lines.append(f"    {c.segment.ts_str}  reason={c.reason}  score={c.score:.2f}  files={n}")
```

- [ ] **Step 5: Add `use_telemetry` to `pipeline.json`**

In `pipeline.json`, the `prune` section gains the flag (keep all existing keys):

```json
  "prune": {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash",
    "use_telemetry": true
  }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_prune.py -k "records_reason or shows_reason" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/dcwb/prune.py pipeline.json tests/test_prune.py
git commit -m "feat(prune): record reason/gear in manifest, report, and config"
```

---

## Task 5: Documentation and full-suite verification

**Files:**
- Modify: `README.md`, `CLAUDE.md`
- (no new tests)

- [ ] **Step 1: Update the prune section in `README.md`**

In the `## RecentClips の整理` section, after the bullet list, add:

```markdown
- **走行判定**: 既定（`use_telemetry: true`）では、Tesla が各 mp4 に埋め込む SEI テレメトリの `gear_state` を読み、DRIVE/REVERSE を含むセグメントは「走行」として保護します。SEI が無いセグメント（駐車中や旧 firmware）は従来どおり front カメラのモーションスコアで判定します。`pipeline.json` の `prune.use_telemetry` を `false` にすると純モーション動作に戻ります。
```

- [ ] **Step 2: Update the prune architecture note in `CLAUDE.md`**

In the `### prune-recent` section, append:

```markdown
判定は既定で **gear 主・モーション補助**: `telemetry.read_segment_telemetry` が front クリップの埋め込み SEI（Tesla 公式 `dashcam.proto`、vendored `src/dcwb/vendor/tesla_dashcam/`）から `gear_state` を読み、DRIVE/REVERSE を含めば走行として保護、全 PARK なら候補（reason=`parked-sei`）。SEI 無しは従来のモーションスコアにフォールバック（reason=`low-motion`）。`prune.use_telemetry=false` で純モーションに戻る。SEI は firmware 2025.44.25+/HW3+ かつ駐車中は欠落しうるため、フォールバックは必須。
```

- [ ] **Step 3: Run the entire test suite**

Run: `uv run --extra dev pytest`
Expected: PASS (all tests, including the original 102 plus the new telemetry/prune tests).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document gear-based prune telemetry signal"
```

---

## Self-review notes

- **Spec coverage:** vendoring (T1), `telemetry.py` (T2), gear-primary 3-branch classification + guards preserved + `use_telemetry` (T3), manifest `reason`/`gear_counts` + report + config + pure-motion regression (T3/T4), docs (T5), synthetic-SEI parser tests + mocked classification tests (T2/T3/T4). Error-handling fail-safe is in `read_segment_telemetry` (T2) and exercised by `test_missing_file_returns_no_sei`.
- **x264 SEI note:** real/synthetic H.264 clips may contain an x264 `user_data_unregistered` SEI, but its UUID does not start with the `42 42 42 69` marker, so `extract_proto_payload` returns `None` and it is not mis-parsed as Tesla telemetry — `has_sei` stays `False` for non-Tesla clips (relied on by the existing prune regression tests).
- **No placeholders / type consistency:** `SegmentTelemetry` fields and `read_segment_telemetry`/`_classify`/`Candidate` signatures are identical across all tasks.
