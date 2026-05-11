import numpy as np
import pytest
from pathlib import Path
from dcwb.calibrate import (
    find_neutral_pixels,
    is_multicolor,
    geometric_median,
    calibrate_camera,
)
from tests.fixtures.make_synthetic import make_event

# 合成 mp4 は均一カラーで is_multicolor が False になるため、
# calibrate のテストでは常に True を返すようパッチする。
@pytest.fixture(autouse=True)
def _bypass_multicolor(monkeypatch):
    monkeypatch.setattr("dcwb.calibrate.is_multicolor",
                       lambda img, threshold=0.05: True)

def test_find_neutral_pixels_picks_high_v_low_s():
    # 上半分: グレー (180, 180, 180) → ニュートラル
    # 下半分: 赤 (200, 50, 50) → 高彩度なので除外
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, :] = 180
    img[50:, 0, ...] = 200
    img[50:, 1, ...] = 50
    img[50:, 2, ...] = 50
    pixels = find_neutral_pixels(img)
    assert pixels.shape[0] > 0
    # 採用ピクセルは平均がほぼ (180,180,180)
    mean = pixels.mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=5)

def test_find_neutral_pixels_excludes_saturated():
    img = np.full((100, 100, 3), 252, dtype=np.uint8)  # ほぼ全飽和
    pixels = find_neutral_pixels(img)
    assert pixels.shape[0] == 0

def test_is_multicolor_passes_diverse_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:33, :, 0] = 200  # 赤帯
    img[33:66, :, 1] = 200  # 緑帯
    img[66:, :, 2] = 200  # 青帯
    assert is_multicolor(img) is True

def test_is_multicolor_rejects_monocolor_image():
    img = np.full((100, 100, 3), 180, dtype=np.uint8)  # 完全均一
    assert is_multicolor(img) is False

def test_geometric_median_robust_to_outliers():
    points = np.array([
        [100.0, 100.0, 100.0],
        [101.0, 102.0, 99.0],
        [99.0, 100.0, 101.0],
        [500.0, 500.0, 500.0],  # 外れ値
    ])
    med = geometric_median(points)
    # 外れ値の影響を強く受けない (≈ 100)
    assert np.all(np.abs(med - 100.0) < 5.0)

def test_calibrate_camera_caps_total_samples_per_event(tmp_path, monkeypatch):
    """max_per_event is a per-event budget spread across the event's clips, not
    a per-clip budget. Three segments × max_per_event=3 must produce 3 frame
    extractions in total, not 9."""
    from tests.fixtures.make_synthetic import make_clip
    import dcwb.calibrate as cal_mod

    ev_dir = tmp_path / "SentryClips" / "evt"
    ev_dir.mkdir(parents=True)
    for ts in ("2026-05-05_13-49-39", "2026-05-05_13-50-39", "2026-05-05_13-51-39"):
        make_clip(ev_dir / f"{ts}-front.mp4", cast_rgb=(1.0, 1.0, 1.0))
    (ev_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T13:49:56","est_lat":"35.68","est_lon":"139.65"}'
    )

    calls: list[str] = []
    real_extract = cal_mod.extract_frame

    def spy(clip, t):
        calls.append(clip.name)
        return real_extract(clip, t)

    monkeypatch.setattr(cal_mod, "extract_frame", spy)
    cal_mod.calibrate_camera("front", source_root=tmp_path, max_per_event=3)
    assert len(calls) == 3, f"expected 3 frame extractions for the event, got {len(calls)}"


def test_calibrate_camera_filters_recentclips_night_by_filename(tmp_path, monkeypatch):
    """RecentClips day-dirs lack event.json so the per-event timestamp filter
    is None. The per-clip filename ts must drive the daylight filter so
    nighttime clips don't bias the profile."""
    from tests.fixtures.make_synthetic import make_clip
    import dcwb.calibrate as cal_mod

    day_dir = tmp_path / "RecentClips" / "2026-05-05"
    day_dir.mkdir(parents=True)
    make_clip(day_dir / "2026-05-05_13-00-00-front.mp4", cast_rgb=(1.0, 1.0, 1.0))
    make_clip(day_dir / "2026-05-05_22-00-00-front.mp4", cast_rgb=(1.0, 1.0, 1.0))

    sampled: list[str] = []
    real_extract = cal_mod.extract_frame

    def spy(clip, t):
        sampled.append(clip.name)
        return real_extract(clip, t)

    monkeypatch.setattr(cal_mod, "extract_frame", spy)
    cal_mod.calibrate_camera("front", source_root=tmp_path, max_per_event=3)
    assert sampled, "expected the daytime clip to be sampled"
    assert all("22-00-00" not in n for n in sampled), \
        f"night clip leaked into calibration: {sampled}"
    assert any("13-00-00" in n for n in sampled)


def test_calibrate_camera_recovers_synthetic_cast(tmp_path):
    # 合成イベントで front カメラに既知のキャスト (R=1.10) を設定。
    # Sentry イベントの保存ルート構造を再現: <root>/SentryClips/<event>/...
    event_dir = tmp_path / "SentryClips" / "2026-05-05_13-49-39"
    cast = (1.10, 1.00, 0.90)  # R 強め, B 弱め
    make_event(
        event_dir,
        casts={
            "front": cast,
            "back": (1.0, 1.0, 1.0),
            "left_pillar": (1.0, 1.0, 1.0),
            "right_pillar": (1.0, 1.0, 1.0),
            "left_repeater": (1.0, 1.0, 1.0),
            "right_repeater": (1.0, 1.0, 1.0),
        },
    )
    # event.json の timestamp は "2026-05-05T13:49:56" → JST 13時 → 昼判定 OK
    profile = calibrate_camera(
        camera="front",
        source_root=tmp_path,
        max_per_event=3,
    )
    # 期待ゲイン: gain_r ≈ G/R = 1.0/1.10 ≈ 0.909
    assert abs(profile.gain_r - 1.0 / 1.10) < 0.02
    assert abs(profile.gain_b - 1.0 / 0.90) < 0.02
    assert profile.calibration.samples_used > 0
