from __future__ import annotations
import struct
from pathlib import Path

import pytest

from dcwb.vendor.tesla_dashcam import dashcam_pb2


def _sei_nal(gear: int, frame_seq: int, speed: float = 0.0) -> bytes:
    """One AVCC-framed Tesla SEI NAL: len-prefix + 0x06 0x05 size 'BBBi' proto 0x80."""
    proto = dashcam_pb2.SeiMetadata(
        version=1, gear_state=gear, frame_seq_no=frame_seq, vehicle_speed_mps=speed
    ).SerializeToString()
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


def test_max_speed_tracked(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    # two DRIVE frames at 5.0 and 12.5 m/s
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x00\x00"
    nals = _sei_nal(1, 0, 5.0) + _sei_nal(1, 1, 12.5)
    mdat = struct.pack(">I", 8 + len(nals)) + b"mdat" + nals
    clip = _write(tmp_path, "speed.mp4", ftyp + mdat)
    tel = read_segment_telemetry(clip)
    assert tel.max_speed_mps == 12.5


def test_speed_summary_fields(tmp_path):
    from dcwb.telemetry import read_segment_telemetry
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x00\x00"
    nals = _sei_nal(1, 0, 2.0) + _sei_nal(1, 1, 8.0) + _sei_nal(1, 2, 5.0)
    mdat = struct.pack(">I", 8 + len(nals)) + b"mdat" + nals
    clip = _write(tmp_path, "speed-summary.mp4", ftyp + mdat)

    tel = read_segment_telemetry(clip)

    assert tel.speed_sample_count == 3
    assert tel.avg_speed_mps == pytest.approx(5.0)
    assert tel.speed_delta_mps == pytest.approx(6.0)
    assert tel.max_speed_mps == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Per-frame SEI iterator tests
# ---------------------------------------------------------------------------

def _sei_nal_full(gear, seq, speed=0.0, heading=0.0, steer=0.0, accel_x=0.0,
                  lat=0.0, lon=0.0) -> bytes:
    proto = dashcam_pb2.SeiMetadata(
        version=1, gear_state=gear, frame_seq_no=seq, vehicle_speed_mps=speed,
        heading_deg=heading, steering_wheel_angle=steer,
        linear_acceleration_mps2_x=accel_x, latitude_deg=lat, longitude_deg=lon,
    ).SerializeToString()
    body = b"\x06\x05" + bytes([len(proto) + 4]) + b"\x42\x42\x42\x69" + proto + b"\x80"
    return struct.pack(">I", len(body)) + body


def _build_full(frames) -> bytes:
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x00\x00"
    nals = b"".join(_sei_nal_full(**f) for f in frames)
    mdat = struct.pack(">I", 8 + len(nals)) + b"mdat" + nals
    return ftyp + mdat


def test_iter_segment_frames_per_frame_fields(tmp_path):
    from dcwb.telemetry import iter_segment_frames, FrameTelemetry
    frames = [
        dict(gear=1, seq=0, speed=5.0, heading=10.0, steer=2.0, accel_x=0.5, lat=35.6, lon=139.6),
        dict(gear=1, seq=1, speed=6.0, heading=20.0, steer=-3.0, accel_x=-0.2, lat=35.7, lon=139.7),
    ]
    clip = tmp_path / "drive.mp4"; clip.write_bytes(_build_full(frames))
    out = list(iter_segment_frames(clip))
    assert len(out) == 2
    assert isinstance(out[0], FrameTelemetry)
    assert out[0].frame_index == 0 and out[0].gear == "DRIVE"
    assert out[0].speed_mps == pytest.approx(5.0)
    assert out[1].heading_deg == pytest.approx(20.0)
    assert out[1].steering_deg == pytest.approx(-3.0, abs=1e-4)
    assert out[0].accel_x == pytest.approx(0.5, abs=1e-6)
    assert out[1].lat == pytest.approx(35.7) and out[1].lon == pytest.approx(139.7)


def test_iter_segment_frames_missing_file_yields_nothing(tmp_path):
    from dcwb.telemetry import iter_segment_frames
    assert list(iter_segment_frames(tmp_path / "nope.mp4")) == []


REAL_FRONT = Path("/mnt/sentryusb/RecentClips/2026-05-27/2026-05-27_17-18-05-front.mp4")


@pytest.mark.skipif(not REAL_FRONT.exists(), reason="real front clip not mounted")
def test_iter_segment_frames_real_clip():
    from dcwb.telemetry import iter_segment_frames
    frames = list(iter_segment_frames(REAL_FRONT))
    assert len(frames) > 100
    assert any(f.gear == "DRIVE" for f in frames)
