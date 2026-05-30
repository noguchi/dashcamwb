import json
import subprocess
from pathlib import Path
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
