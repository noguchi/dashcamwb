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
