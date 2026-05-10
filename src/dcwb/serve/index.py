from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

RECENT_GAP_MINUTES = 10

CAMERAS = (
    "front", "back",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
)

_CAM_SUFFIX_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(?P<cam>" +
    "|".join(CAMERAS) + r")\.mp4$"
)


@dataclass
class Event:
    source: str
    name: str
    path: Path
    clips: list[Path]
    start: datetime
    end: datetime
    thumb: Path | None = None
    meta: Path | None = None  # event.json if present


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")


def _scan_event_dir(source: str, event_dir: Path) -> Event | None:
    """For SentryClips / SavedClips: one subdir = one Event."""
    clips: list[Path] = []
    timestamps: list[datetime] = []
    for clip in sorted(event_dir.glob("*.mp4")):
        m = _CAM_SUFFIX_RE.match(clip.name)
        if not m:
            continue
        clips.append(clip)
        timestamps.append(_parse_ts(m.group("ts")))
    if not clips:
        return None
    start = min(timestamps)
    end = max(timestamps) + timedelta(minutes=1)
    thumb = event_dir / "thumb.png"
    meta = event_dir / "event.json"
    return Event(
        source=source,
        name=event_dir.name,
        path=event_dir,
        clips=clips,
        start=start,
        end=end,
        thumb=thumb if thumb.exists() else None,
        meta=meta if meta.exists() else None,
    )


def _group_recent_day(day_dir: Path) -> list[Event]:
    """Group flat day-dir clips into pseudo-events by RECENT_GAP_MINUTES."""
    # collect (timestamp, clip) sorted asc
    pairs: list[tuple[datetime, Path]] = []
    for clip in day_dir.glob("*.mp4"):
        m = _CAM_SUFFIX_RE.match(clip.name)
        if not m:
            continue
        pairs.append((_parse_ts(m.group("ts")), clip))
    pairs.sort(key=lambda p: (p[0], p[1].name))
    if not pairs:
        return []

    threshold = timedelta(minutes=RECENT_GAP_MINUTES)
    groups: list[list[tuple[datetime, Path]]] = [[pairs[0]]]
    for ts, clip in pairs[1:]:
        prev_ts = groups[-1][-1][0]
        if ts - prev_ts >= threshold:
            groups.append([])
        groups[-1].append((ts, clip))

    out: list[Event] = []
    for grp in groups:
        ts_list = [t for t, _ in grp]
        clip_list = [c for _, c in grp]
        start = min(ts_list)
        end = max(ts_list) + timedelta(minutes=1)
        name = start.strftime("%Y-%m-%d_%H%M")
        out.append(Event(
            source="RecentClips",
            name=name,
            path=day_dir,
            clips=clip_list,
            start=start,
            end=end,
            thumb=None,
            meta=None,
        ))
    return out


def scan_sources(usb_root: Path) -> dict[str, list[Event]]:
    """Scan SentryClips / SavedClips / RecentClips. Empty dict-of-lists if missing."""
    result: dict[str, list[Event]] = {
        "SentryClips": [],
        "SavedClips": [],
        "RecentClips": [],
    }
    if not usb_root.exists():
        return result

    for source in ("SentryClips", "SavedClips"):
        src_root = usb_root / source
        if not src_root.exists():
            continue
        events: list[Event] = []
        for child in src_root.iterdir():
            if not child.is_dir():
                continue
            ev = _scan_event_dir(source, child)
            if ev is not None:
                events.append(ev)
        events.sort(key=lambda e: e.start, reverse=True)
        result[source] = events

    recent_root = usb_root / "RecentClips"
    if recent_root.exists():
        recent_events: list[Event] = []
        for day_dir in recent_root.iterdir():
            if not day_dir.is_dir():
                continue
            recent_events.extend(_group_recent_day(day_dir))
        recent_events.sort(key=lambda e: e.start, reverse=True)
        result["RecentClips"] = recent_events

    return result
