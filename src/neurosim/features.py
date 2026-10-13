from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import welch

from .brain import ASYMMETRY_PAIRS, BAND_NAMES, BANDS, CHANNELS

WINDOW_S = 2.0
STEP_S = 0.5


@dataclass
class EEGFeatures:
    times: np.ndarray
    band_power: np.ndarray
    values: np.ndarray
    names: list[str]

    def during_stimulus(self) -> np.ndarray:
        return self.times - WINDOW_S / 2 >= -1e-9


def feature_names(ch_names: tuple[str, ...] = CHANNELS) -> list[str]:
    names = [f"{band} {ch}" for ch in ch_names for band in BAND_NAMES]
    names += [f"{band} asymmetry {left}-{right}" for left, right in ASYMMETRY_PAIRS for band in BAND_NAMES]
    names += [f"spectral entropy {ch}" for ch in ch_names]
    return names


def extract_features(data: np.ndarray, sfreq: float, t0: float, 
                     ch_names: tuple[str, ...] = CHANNELS, baseline_correct: bool = True,) -> EEGFeatures:
    win, step = int(WINDOW_S * sfreq), int(STEP_S * sfreq)
    windows = sliding_window_view(data, win, axis=1)[:, ::step]
    times = t0 + (np.arange(windows.shape[1]) * step + win / 2) / sfreq

    nperseg = int(sfreq)
    freqs, psd = welch(windows, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    df = freqs[1] - freqs[0]
    band_power = np.stack(
        [psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df for lo, hi in BANDS.values()], axis=-1)
    log_power = np.log10(band_power + 1e-12).transpose(1, 0, 2)

    index = {name: i for i, name in enumerate(ch_names)}
    left = [index[a] for a, _ in ASYMMETRY_PAIRS]
    right = [index[b] for _, b in ASYMMETRY_PAIRS]
    asymmetry = log_power[:, right] - log_power[:, left]

    in_range = (freqs >= 1.0) & (freqs <= 45.0)
    p = psd[..., in_range] / psd[..., in_range].sum(axis=-1, keepdims=True)
    entropy = (-(p * np.log(p + 1e-12)).sum(axis=-1) / np.log(in_range.sum())).T

    if baseline_correct:
        rest = times + WINDOW_S / 2 <= 1e-9
        if rest.any():
            log_power = log_power - np.median(log_power[rest], axis=0)
            asymmetry = asymmetry - np.median(asymmetry[rest], axis=0)
            entropy = entropy - np.median(entropy[rest], axis=0)

    values = np.hstack(
        [log_power.reshape(len(times), -1), asymmetry.reshape(len(times), -1), entropy])
    return EEGFeatures(times=times, band_power=log_power, values=values, names=feature_names(ch_names))


def window_average(times: np.ndarray, signal: np.ndarray, centres: np.ndarray, width: float = WINDOW_S) -> np.ndarray:
    cumulative = np.r_[0.0, np.cumsum(signal)]
    start = np.searchsorted(times, centres - width / 2)
    stop = np.searchsorted(times, centres + width / 2)
    count = np.maximum(stop - start, 1)
    return (cumulative[stop] - cumulative[start]) / count
