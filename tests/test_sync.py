import numpy as np
from dcwb.sync import unwrap_yaw_rate, resample_uniform, normalized_xcorr

def test_unwrap_yaw_rate_handles_360_wrap():
    # heading crosses 360->0; constant 10 deg/s at 1 Hz
    headings = np.array([350.0, 0.0, 10.0, 20.0])  # +10 deg each step
    rate = unwrap_yaw_rate(headings, dt=1.0)
    assert np.allclose(rate, 10.0, atol=1e-6)

def test_resample_uniform_linear():
    t = np.array([0.0, 1.0, 3.0])
    v = np.array([0.0, 10.0, 30.0])
    grid_t, grid_v = resample_uniform(t, v, rate_hz=1.0)
    assert np.allclose(grid_t, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(grid_v, [0.0, 10.0, 20.0, 30.0])

def test_normalized_xcorr_recovers_known_lag():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(500)
    lag = 37
    a = base
    b = np.concatenate([np.zeros(lag), base])[:500]  # b is base delayed by 37
    best_lag, peak = normalized_xcorr(a, b, max_lag=100)
    assert best_lag == lag
    assert peak > 0.9
