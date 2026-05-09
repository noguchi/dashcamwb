import numpy as np

def shades_of_gray(
    image_rgb: np.ndarray,
    p: int = 6,
    sat_high: float = 0.97,
    sat_low: float = 0.03,
) -> tuple[float, float, float]:
    """Estimate illuminant via Shades of Gray (Minkowski p-norm) and return
    per-channel gains (g_r, g_g, g_b) such that applying them neutralises
    the estimated illuminant.

    Args:
        image_rgb: HxWx3 uint8 array in RGB order
        p: Minkowski exponent (1 = Gray World, ∞ ≈ White Patch, 6 = balanced)
        sat_high: discard pixels where any channel exceeds this fraction (0..1)
        sat_low:  discard pixels where any channel falls below this fraction
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    img = image_rgb.astype(np.float64) / 255.0
    flat = img.reshape(-1, 3)
    mask = np.all((flat <= sat_high) & (flat >= sat_low), axis=1)
    valid = flat[mask]
    if valid.shape[0] == 0:
        return 1.0, 1.0, 1.0
    # Minkowski p-norm per channel (Shades of Gray illuminant estimate)
    e = np.power(np.mean(np.power(valid, p), axis=0), 1.0 / p)
    # Normalise gains by the green channel so g_g == 1 (downstream
    # compose_clip_matrix uses gain_min/gain_max symmetric around 1).
    e_g = float(e[1])
    g_r = e_g / float(e[0])
    g_g = 1.0
    g_b = e_g / float(e[2])
    return g_r, g_g, g_b
