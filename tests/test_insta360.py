from datetime import timezone
from pathlib import Path
from dcwb.insta360 import read_creation_time, to_jst
from tests.fixtures.make_insta360 import make_insv_header

def test_read_creation_time_returns_utc(tmp_path: Path):
    f = tmp_path / "VID.insv"
    make_insv_header(f, "2026-05-27T08:17:57Z")
    dt = read_creation_time(f)
    assert dt.tzinfo is not None
    assert dt.astimezone(timezone.utc).isoformat().startswith("2026-05-27T08:17:57")

def test_to_jst_adds_nine_hours(tmp_path: Path):
    f = tmp_path / "VID.insv"
    make_insv_header(f, "2026-05-27T08:17:57Z")
    jst = to_jst(read_creation_time(f))
    assert jst.hour == 17 and jst.minute == 17 and jst.second == 57
