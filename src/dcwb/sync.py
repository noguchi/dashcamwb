from __future__ import annotations
import numpy as np

def unwrap_yaw_rate(headings_deg: np.ndarray, dt: float) -> np.ndarray:
    """deg/s yaw rate from a heading series, robust to 0/360 wraparound."""
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, float))))
    return np.gradient(unwrapped, dt)

def resample_uniform(t: np.ndarray, v: np.ndarray, rate_hz: float):
    """Linearly resample irregular (t, v) onto a uniform grid at rate_hz."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    grid_t = np.arange(t[0], t[-1] + 1e-9, 1.0 / rate_hz)
    grid_v = np.interp(grid_t, t, v)
    return grid_t, grid_v

def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Return (best_lag, peak) maximizing normalized cross-correlation of a vs b
    over integer lags in [-max_lag, max_lag]. Positive lag means b lags a."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    best_lag, best = 0, -np.inf
    n = min(len(a), len(b))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:n - lag], b[lag:n]
        else:
            x, y = a[-lag:n], b[:n + lag]
        if len(x) < 2:
            continue
        score = float(np.dot(x, y) / len(x))
        if score > best:
            best, best_lag = score, lag
    return best_lag, best
