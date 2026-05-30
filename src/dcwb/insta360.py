from __future__ import annotations
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))

def read_creation_time(insv: Path) -> datetime:
    """Return the mp4 header creation_time as a tz-aware UTC datetime."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "json", str(insv)],
        check=True, capture_output=True, text=True,
    ).stdout
    tag = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
    if not tag:
        raise ValueError(f"no creation_time in {insv}")
    return datetime.fromisoformat(tag.replace("Z", "+00:00")).astimezone(timezone.utc)

def to_jst(dt: datetime) -> datetime:
    return dt.astimezone(JST)
