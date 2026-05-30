import math
from datetime import timezone
from pathlib import Path
import pytest
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


# ── Task 3: read_imu tests ───────────────────────────────────────────────────

from dcwb.insta360 import read_imu, ImuSample
from tests.fixtures.make_insta360 import append_imu_trailer

def test_read_imu_roundtrip(tmp_path):
    f = tmp_path / "VID.insv"
    make_insv_header(f)
    # 50 samples at 1000 Hz: az = 1 g (gravity), gyro_z cycles 0/1/2 deg/s
    samples = [(i / 1000.0, 0.0, 0.0, 1.0, 0.0, 0.0, float(i % 3)) for i in range(50)]
    append_imu_trailer(f, samples)
    series = read_imu(f)
    assert len(series) == 50
    assert isinstance(series[0], ImuSample)
    assert math.isclose(series[10].t_s, 10 / 1000.0, abs_tol=1e-6)
    assert math.isclose(series[0].accel[2], 9.80665, abs_tol=0.02)   # 1 g -> m/s^2
    assert math.isclose(series[2].gyro[2], 2.0 * math.pi / 180, abs_tol=2e-3)  # 2 deg/s

def test_read_imu_no_trailer_raises(tmp_path):
    f = tmp_path / "VID.insv"
    make_insv_header(f)
    with pytest.raises(ValueError):
        read_imu(f)

REAL_INSV = Path("/mnt/sentryusb/Insta360/VID_20260527_171757_00_007_009-オリジナル/"
                 "VID_20260527_171757_00_007.insv")

@pytest.mark.skipif(not REAL_INSV.exists(), reason="real .insv not mounted")
def test_read_imu_real_file_plausible():
    series = read_imu(REAL_INSV)
    assert len(series) > 100_000
    mags = [math.sqrt(sum(c * c for c in s.accel)) for s in series[:500]]
    mean_mag = sum(mags) / len(mags)
    assert 8.0 < mean_mag < 11.5  # gravity in m/s^2
