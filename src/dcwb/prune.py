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
