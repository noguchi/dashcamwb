import numpy as np
import pytest
from pathlib import Path
from dcwb.ffmpeg_wrap import probe_duration, extract_frame, extract_frames, render_with_matrix
from dcwb.matrix import from_diag
from tests.fixtures.make_synthetic import make_clip, make_motion_clip

@pytest.fixture
def sample_clip(tmp_path) -> Path:
    p = tmp_path / "sample.mp4"
    make_clip(p, cast_rgb=(1.0, 1.0, 1.0), duration_sec=3.0)
    return p

def test_probe_duration_returns_seconds(sample_clip):
    d = probe_duration(sample_clip)
    assert 2.5 < d < 3.5  # generated as 3.0s

def test_extract_frame_returns_rgb_numpy(sample_clip):
    img = extract_frame(sample_clip, t=1.0)
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    # color was 180,180,180 → mean ≈ 180 (some compression noise)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=10)

def test_render_with_identity_preserves_color(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, np.eye(3), bitrate_kbps=4000, encoder="libx264")
    assert out.exists()
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=15)

def test_render_with_red_attenuation_reduces_red(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, from_diag(0.5, 1.0, 1.0), bitrate_kbps=4000, encoder="libx264")
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    # R は約半分、G/B は維持
    assert mean[0] < 110
    assert 160 < mean[1] < 200
    assert 160 < mean[2] < 200


def test_extract_frames_returns_requested_count(tmp_path):
    clip = tmp_path / "m.mp4"
    make_motion_clip(clip, duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0, 1.4])
    assert len(frames) == 4
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames)


def test_extract_frames_static_clip_frames_near_identical(tmp_path):
    clip = tmp_path / "s.mp4"
    make_clip(clip, (1.0, 1.0, 1.0), duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0])
    assert len(frames) == 3
    for a, b in zip(frames, frames[1:]):
        d = np.abs(a.astype(int) - b.astype(int)).mean()
        assert d < 2.0


def test_cut_clip_writes_playable_excerpt(tmp_path):
    from dcwb.ffmpeg_wrap import cut_clip, probe_duration
    from tests.fixtures.make_synthetic import make_motion_clip
    src = tmp_path / "src.mp4"
    dst = tmp_path / "cut.mp4"
    make_motion_clip(src, duration_sec=2.0)

    cut_clip(src, dst, start_sec=0.25, duration_sec=0.75, encoder="libx264", bitrate_kbps=1000)

    assert dst.exists()
    assert 0.4 <= probe_duration(dst) <= 1.2


def test_cut_clip_applies_color_matrix(tmp_path):
    import numpy as np
    from dcwb.ffmpeg_wrap import cut_clip, extract_frame
    from tests.fixtures.make_synthetic import make_clip
    src = tmp_path / "src.mp4"
    make_clip(src, (1.0, 1.0, 1.0), duration_sec=1.0)  # neutral gray

    plain = tmp_path / "plain.mp4"
    boosted = tmp_path / "boosted.mp4"
    cut_clip(src, plain, 0.0, 0.5, encoder="libx264", bitrate_kbps=2000)
    cut_clip(
        src, boosted, 0.0, 0.5, encoder="libx264", bitrate_kbps=2000,
        matrix=np.array([[1.6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.6]]),
    )

    plain_r = float(extract_frame(plain, 0.2)[..., 0].mean())
    boosted_r = float(extract_frame(boosted, 0.2)[..., 0].mean())
    boosted_b = float(extract_frame(boosted, 0.2)[..., 2].mean())
    plain_b = float(extract_frame(plain, 0.2)[..., 2].mean())
    assert boosted_r > plain_r + 5   # red channel boosted
    assert boosted_b < plain_b - 5   # blue channel attenuated


def test_look_config_from_dict_merges_defaults_and_ignores_unknown():
    from dcwb.ffmpeg_wrap import LookConfig
    cfg = LookConfig.from_dict({"saturation": 1.2, "bogus": 9})
    assert cfg.saturation == 1.2
    assert cfg.gamma == LookConfig().gamma
    assert not hasattr(cfg, "bogus")


def test_cut_clip_look_boosts_saturation(tmp_path):
    from dcwb.ffmpeg_wrap import cut_clip, extract_frame, LookConfig
    from tests.fixtures.make_synthetic import make_clip
    src = tmp_path / "src.mp4"
    make_clip(src, (1.2, 1.0, 0.8), duration_sec=1.0)  # colored (R high, B low)

    plain = tmp_path / "plain.mp4"
    looked = tmp_path / "looked.mp4"
    cut_clip(src, plain, 0.0, 0.5, encoder="libx264", bitrate_kbps=3000)
    cut_clip(
        src, looked, 0.0, 0.5, encoder="libx264", bitrate_kbps=3000,
        look=LookConfig(saturation=1.6),
    )

    pf = extract_frame(plain, 0.2).reshape(-1, 3).mean(axis=0)
    lf = extract_frame(looked, 0.2).reshape(-1, 3).mean(axis=0)
    # saturation boost widens the R-B channel spread of a colored frame
    assert (lf[0] - lf[2]) > (pf[0] - pf[2]) + 3


def _color_tags(path):
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_space,color_primaries,color_transfer",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def test_cut_clip_look_tags_bt709(tmp_path):
    from dcwb.ffmpeg_wrap import cut_clip, LookConfig
    from tests.fixtures.make_synthetic import make_clip
    src = tmp_path / "src.mp4"
    make_clip(src, (1.0, 1.0, 1.0), duration_sec=1.0)
    dst = tmp_path / "out.mp4"
    cut_clip(src, dst, 0.0, 0.5, encoder="libx264", bitrate_kbps=2000, look=LookConfig(tag_bt709=True))
    tags = _color_tags(dst)
    assert tags["color_primaries"] == "bt709"
    assert tags["color_transfer"] == "bt709"
    assert tags["color_space"] == "bt709"


def test_concat_clips_tags_bt709_on_final_output(tmp_path):
    from dcwb.ffmpeg_wrap import concat_clips
    from tests.fixtures.make_synthetic import make_motion_clip
    a = tmp_path / "a.mp4"; b = tmp_path / "b.mp4"; out = tmp_path / "j.mp4"
    make_motion_clip(a, duration_sec=1.0)
    make_motion_clip(b, duration_sec=1.0)
    concat_clips([a, b], out, encoder="libx264", bitrate_kbps=1000, tag_bt709=True)
    tags = _color_tags(out)
    assert tags["color_primaries"] == "bt709"
    assert tags["color_transfer"] == "bt709"


def test_concat_clips_writes_playable_video(tmp_path):
    from dcwb.ffmpeg_wrap import concat_clips, probe_duration
    from tests.fixtures.make_synthetic import make_motion_clip
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    out = tmp_path / "joined.mp4"
    make_motion_clip(first, duration_sec=1.0)
    make_motion_clip(second, duration_sec=1.0)

    concat_clips([first, second], out, encoder="libx264", bitrate_kbps=1000)

    assert out.exists()
    assert probe_duration(out) >= 1.5


def test_resolve_encoder_falls_back_when_requested_unavailable(monkeypatch):
    from dcwb import ffmpeg_wrap
    monkeypatch.setattr(ffmpeg_wrap, "_available_encoders", lambda: frozenset({"libx264"}))
    assert ffmpeg_wrap.resolve_encoder("h264_videotoolbox") == "libx264"


def test_resolve_encoder_keeps_requested_when_available(monkeypatch):
    from dcwb import ffmpeg_wrap
    monkeypatch.setattr(
        ffmpeg_wrap, "_available_encoders", lambda: frozenset({"h264_videotoolbox", "libx264"})
    )
    assert ffmpeg_wrap.resolve_encoder("h264_videotoolbox") == "h264_videotoolbox"


def test_resolve_encoder_keeps_request_when_probe_empty(monkeypatch):
    from dcwb import ffmpeg_wrap
    monkeypatch.setattr(ffmpeg_wrap, "_available_encoders", lambda: frozenset())
    assert ffmpeg_wrap.resolve_encoder("h264_videotoolbox") == "h264_videotoolbox"


def test_concat_clips_mismatched_timescales_preserves_total_duration(tmp_path):
    """Real Tesla front clips have per-clip timescales (e.g. 18432 vs 7170000).
    Concatenating mismatched segments via the concat demuxer corrupts PTS and
    produces a multi-hour file; concat_clips must yield ~sum-of-durations."""
    from dcwb.ffmpeg_wrap import concat_clips, probe_duration
    from tests.fixtures.make_synthetic import make_motion_clip
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    out = tmp_path / "joined.mp4"
    make_motion_clip(a, duration_sec=2.0, fps=36, timescale=18432)
    make_motion_clip(b, duration_sec=2.0, fps=36, timescale=7170000)

    concat_clips([a, b], out, encoder="libx264", bitrate_kbps=1000)

    assert out.exists()
    assert 3.5 <= probe_duration(out) <= 4.5
