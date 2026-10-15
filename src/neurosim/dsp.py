from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter


def smooth(x: np.ndarray, seconds: float, rate: float) -> np.ndarray:
    sigma = seconds * rate
    if sigma <= 0 or len(x) < 2:
        return np.asarray(x, dtype=float)
    return gaussian_filter1d(np.asarray(x, dtype=float), sigma=sigma, mode="nearest")


def leaky(x: np.ndarray, tau: float, rate: float) -> np.ndarray:
    a = np.exp(-1.0 / (tau * rate))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(x, dtype=float))


def to_grid(t_src: np.ndarray, values: np.ndarray, n: int, rate: float) -> np.ndarray:
    grid = np.arange(n) / rate
    if len(t_src) == 0:
        return np.zeros(n)
    return np.interp(grid, t_src, values)


def impulses(times: np.ndarray, strengths: np.ndarray, n: int, rate: float, t0: float = 0.0) -> np.ndarray:
    out = np.zeros(n)
    idx = np.round((np.asarray(times) - t0) * rate).astype(int)
    keep = (idx >= 0) & (idx < n)
    np.add.at(out, idx[keep], np.asarray(strengths)[keep])
    return out
