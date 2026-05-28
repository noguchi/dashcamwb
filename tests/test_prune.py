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
