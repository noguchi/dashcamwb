from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from dcwb.vendor.tesla_dashcam import sei_extractor as _sx

# dashcam.proto: enum Gear { PARK=0; DRIVE=1; REVERSE=2; NEUTRAL=3; }
# proto enum names are GEAR_*; we strip the prefix for compact output
_GEAR_NAME = {0: "PARK", 1: "DRIVE", 2: "REVERSE", 3: "NEUTRAL"}


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
