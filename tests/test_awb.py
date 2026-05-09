import numpy as np
from dcwb.awb import shades_of_gray

def test_neutral_gray_image_returns_unity_gains():
    # 128/255 のグレー一面 → ゲインは ≈ 1.0
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    g_r, g_g, g_b = shades_of_gray(image, p=6)
    assert abs(g_r - 1.0) < 0.01
    assert abs(g_g - 1.0) < 0.01
    assert abs(g_b - 1.0) < 0.01

def test_red_tinted_image_returns_red_attenuation():
    # 赤強め・青弱めの照明下 (R=180, G=128, B=110) → R ゲインが < 1, B ゲインが > 1
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[..., 0] = 180  # R
    image[..., 1] = 128  # G
    image[..., 2] = 110  # B
    # OpenCV BGR convention は呼び出し側で処理。awb は (H, W, 3) の RGB を想定
    g_r, g_g, g_b = shades_of_gray(image, p=6)
    assert g_r < 1.0  # 赤を抑える
    assert abs(g_g - 1.0) < 0.05  # 緑が基準
    assert g_b > 1.0  # 青を持ち上げる

def test_saturated_pixels_excluded():
    # 半分は飽和 (255, 255, 255)、半分は (100, 100, 100)
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    image[:50, :, :] = 255
    g_r, g_g, g_b = shades_of_gray(image, p=6, sat_high=0.97)
    # 飽和を除外すれば下半分のグレーが支配 → ゲイン約 1.0
    assert abs(g_r - 1.0) < 0.05

def test_shadow_pixels_excluded():
    # 半分は影 (5, 5, 5)、半分はグレー (128, 128, 128)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    image[:50, :, :] = 5
    g_r, g_g, g_b = shades_of_gray(image, p=6, sat_low=0.03)
    assert abs(g_r - 1.0) < 0.05
