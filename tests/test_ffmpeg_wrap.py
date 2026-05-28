import numpy as np
import pytest
from pathlib import Path
from dcwb.ffmpeg_wrap import probe_duration, extract_frame, render_with_matrix
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
    from dcwb.ffmpeg_wrap import extract_frames
    clip = tmp_path / "m.mp4"
    make_motion_clip(clip, duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0, 1.4])
    assert len(frames) == 4
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames)


def test_extract_frames_static_clip_frames_near_identical(tmp_path):
    from dcwb.ffmpeg_wrap import extract_frames
    clip = tmp_path / "s.mp4"
    make_clip(clip, (1.0, 1.0, 1.0), duration_sec=2.0)
    frames = extract_frames(clip, [0.2, 0.6, 1.0])
    assert len(frames) == 3
    d = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
    assert d < 2.0
