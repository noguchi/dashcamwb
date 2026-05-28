from __future__ import annotations
import json
import os
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
    if now.tzinfo is None:
        now = now.replace(tzinfo=JST)
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
    path = _manifest_path(trash_root)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(body + ("\n" if rows else ""))
    os.replace(tmp, path)


def quarantine(usb_root: Path, candidates: list[Candidate], cfg: dict, now: datetime) -> list[dict]:
    """Move each candidate segment's files into the trash and append manifest rows.

    Moves happen first, then the manifest is written once. Intended for a manual
    single-user CLI: if the process is killed mid-run, already-moved files may lack
    manifest rows and must be reconciled against the trash dir by hand. Files whose
    trash destination already exists are skipped (never overwritten).
    """
    trash_root = usb_root / cfg["trash_dir"]
    rows = _load_manifest(trash_root)
    new_rows: list[dict] = []
    for cand in candidates:
        seg = cand.segment
        for clip in seg.clips:
            rel = clip.relative_to(usb_root)
            dest = trash_root / rel
            if dest.exists():
                print(f"[prune] skip (already in trash): {rel.as_posix()}", file=sys.stderr)
                continue
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


def purge(usb_root: Path, cfg: dict, now: datetime) -> int:
    """Delete trash files whose quarantine is older than retention_days."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=JST)
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
        if not tp.exists():
            print(f"[prune] restore skip (trash missing): {row['trash_path']}", file=sys.stderr)
            continue
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tp), str(orig))
        row["status"] = "restored"
        restored += 1
    _write_manifest(trash_root, rows)
    return restored
