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
