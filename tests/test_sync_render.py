import json
import subprocess
from pathlib import Path
import pytest
from dcwb.ffmpeg_wrap import render_sidebyside, probe_duration
from dcwb.sync import telemetry_ass
from tests.fixtures.make_synthetic import make_motion_clip

def _probe_dims(p: Path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(p)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]

def test_render_sidebyside_hstacks_with_subs(tmp_path):
    left = tmp_path / "insta.mp4"; right = tmp_path / "tesla.mp4"
    make_motion_clip(left, duration_sec=4.0, width=320, height=240)
    make_motion_clip(right, duration_sec=4.0, width=320, height=240)
    ass = tmp_path / "tele.ass"
    ass.write_text(telemetry_ass([(0.0, 10.0, 0.0, "DRIVE")], play_w=640, play_h=240))
    dst = tmp_path / "combined.mp4"
    render_sidebyside(left, right, dst, left_start=0.5, right_start=0.0,
                      duration=3.0, ass_path=ass, encoder="libx264",
                      bitrate_kbps=2000, panel_h=240)
    assert dst.exists()
    w, h = _probe_dims(dst)
    assert w in (638, 640, 642) and h == 240   # two ~320-wide panels hstacked
    assert 2.5 < probe_duration(dst) < 3.5


def test_render_sidebyside_applies_right_matrix(tmp_path):
    """A right_matrix is a colorchannelmixer applied only to the right (Tesla)
    panel before hstack — used to bake the reference match gain into the composite."""
    import numpy as np
    from dcwb.matrix import from_diag
    from dcwb.ffmpeg_wrap import extract_frame
    from tests.fixtures.make_synthetic import make_clip

    left = tmp_path / "insta.mp4"; right = tmp_path / "tesla.mp4"
    make_clip(left, (1.0, 1.0, 1.0), duration_sec=2.0, width=320, height=240)
    make_clip(right, (1.0, 1.0, 1.0), duration_sec=2.0, width=320, height=240)  # neutral gray
    dst = tmp_path / "combined.mp4"
    render_sidebyside(left, right, dst, left_start=0.0, right_start=0.0,
                      duration=1.5, encoder="libx264", bitrate_kbps=2000, panel_h=240,
                      right_matrix=from_diag(1.4, 1.0, 0.6))  # warm-boost the right panel
    img = extract_frame(dst, t=0.5)
    w = img.shape[1]
    left_px = img[:, w // 4].reshape(-1, 3).mean(axis=0)
    right_px = img[:, 3 * w // 4].reshape(-1, 3).mean(axis=0)
    # left panel stays neutral; right panel is warmed (R >> B)
    assert abs(float(left_px[0]) - float(left_px[2])) < 15
    assert float(right_px[0]) - float(right_px[2]) > 60


_HAS_V360 = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
    capture_output=True, text=True).stdout.find(" v360 ") != -1


def _make_dual_fisheye(path: Path):
    # two video streams in one mp4, both 240x240, 2s
    # Force -f mp4 so ffmpeg doesn't trip on the .insv extension it doesn't know.
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","lavfi","-i","testsrc=d=2:s=240x240:r=30",
        "-f","lavfi","-i","testsrc2=d=2:s=240x240:r=30",
        "-map","0:v","-map","1:v","-c:v","libx264","-pix_fmt","yuv420p",
        "-preset","ultrafast","-f","mp4", str(path)], check=True, capture_output=True)


@pytest.mark.skipif(not _HAS_V360, reason="ffmpeg build lacks the v360 filter")
def test_reframe_insv_outputs_flat(tmp_path):
    from dcwb.ffmpeg_wrap import reframe_insv
    src = tmp_path / "dual.insv"; _make_dual_fisheye(src)
    dst = tmp_path / "flat.mp4"
    reframe_insv(src, dst, yaw=0.0, pitch=-10.0, out_w=480, out_h=270,
                 encoder="libx264", bitrate_kbps=2000)
    assert dst.exists()
    w, h = _probe_dims(dst)
    assert (w, h) == (480, 270)
