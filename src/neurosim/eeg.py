from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

from .brain import BAND_NAMES, BANDS, CHANNELS, REGION_KEYS, REGIONS, lead_field, unit_vectors
from .dsp import impulses
from .stimulus import Events
from .subject import Subject

SFREQ = 128.0  # Hz

REST_AMPLITUDE = {
    "frontal_left": (7.0, 4.5, 6.0, 3.0, 1.0),
    "frontal_right": (7.0, 4.5, 6.0, 3.0, 1.0),
    "frontal_midline": (7.0, 5.5, 5.0, 3.0, 1.0),
    "temporal_left": (6.0, 4.0, 5.0, 3.0, 1.2),
    "temporal_right": (6.0, 4.0, 5.0, 3.0, 1.2),
    "central": (6.0, 4.0, 6.5, 3.5, 1.0),
    "parietal": (6.0, 4.0, 9.0, 3.0, 1.0),
    "occipital": (6.0, 4.0, 12.0, 3.0, 1.0),}

DRIVES = ("valence", "arousal", "auditory", "visual", "motion")

MODULATION = {
    "arousal": {
        "delta": {"*": -0.25},
        "theta": {"frontal_midline": 0.2},
        "alpha": {"*": -0.4, "central": -0.45, "parietal": -0.6, "occipital": -0.6},
        "beta": {
            "*": 0.15, "frontal_left": 0.3, "frontal_right": 0.3, "frontal_midline": 0.3,
            "central": 0.25, "temporal_left": 0.2, "temporal_right": 0.2,},
        "gamma": {
            "*": 0.15, "frontal_left": 0.25, "frontal_right": 0.25, "frontal_midline": 0.25,
            "temporal_left": 0.3, "temporal_right": 0.3,},},

    "valence": {
        "alpha": {"frontal_left": -0.6, "frontal_right": 0.6},
        "beta": {"temporal_left": 0.35, "temporal_right": 0.35},
        "gamma": {"temporal_left": 0.4, "temporal_right": 0.4},},

    "auditory": {
        "alpha": {"temporal_left": -0.3, "temporal_right": -0.3},
        "beta": {"temporal_left": 0.2, "temporal_right": 0.2},
        "gamma": {"temporal_left": 0.45, "temporal_right": 0.45},},
        
    "visual": {
        "alpha": {"occipital": -0.35, "parietal": -0.15},
        "gamma": {"occipital": 0.15},},

    "motion": {
        "alpha": {"occipital": -0.3, "parietal": -0.15},
        "beta": {"occipital": 0.3},
        "gamma": {"occipital": 0.5, "parietal": 0.2},},}

AUDITORY_ERP_UV = 5.0
AUDITORY_ERP_REGIONS = {"central": 1.0, "frontal_midline": 0.8, "temporal_left": 0.7, "temporal_right": 0.7}
AUDITORY_ADAPTATION_S = 1.0
VISUAL_ERP_UV = 6.0
VISUAL_ERP_REGIONS = {"occipital": 1.0, "parietal": 0.4}
LPP_UV = 5.0
LPP_REGIONS = {"parietal": 1.0, "central": 0.6}

BACKGROUND_UV = 5.0
SENSOR_NOISE_UV = 1.0
BLINK_UV = (60.0, 120.0)  # at the frontopolar electrodes


@dataclass
class Drives:

    rate: float
    times: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    auditory: np.ndarray
    visual: np.ndarray
    motion: np.ndarray
    audio_onsets: Events
    visual_onsets: Events

    def matrix(self) -> np.ndarray:
        return np.stack([getattr(self, name) for name in DRIVES])


@dataclass
class EEG:
    data: np.ndarray
    sfreq: float
    t0: float
    log_power: np.ndarray
    blinks: np.ndarray
    ch_names: tuple[str, ...] = CHANNELS

    @property
    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.data.shape[1]) / self.sfreq

    def cleaned(self, residual: float = 0.1) -> np.ndarray:
        return self.data - (1.0 - residual) * self.blinks


@lru_cache(maxsize=1)
def modulation_array() -> np.ndarray:
    out = np.zeros((len(DRIVES), len(REGIONS), len(BAND_NAMES)))
    for d, drive in enumerate(DRIVES):
        for band, per_region in MODULATION.get(drive, {}).items():
            b = BAND_NAMES.index(band)
            out[d, :, b] = per_region.get("*", 0.0)
            for region, value in per_region.items():
                if region != "*":
                    out[d, REGION_KEYS.index(region), b] = value
    return out


def region_log_power(drives: Drives, subject: Subject) -> np.ndarray:
    rest = 2.0 * np.log(np.array([REST_AMPLITUDE[key] for key in REGION_KEYS]))
    rest += 2.0 * np.log(subject.amplitude_scale)
    alpha = BAND_NAMES.index("alpha")
    rest[:, alpha] += 2.0 * np.log(subject.alpha_scale)
    rest[REGION_KEYS.index("frontal_right"), alpha] += subject.faa_trait / 2.0
    rest[REGION_KEYS.index("frontal_left"), alpha] -= subject.faa_trait / 2.0
    return rest[:, :, None] + np.einsum("drb,dt->rbt", modulation_array(), drives.matrix())


def region_activation(log_power: np.ndarray, times: np.ndarray) -> np.ndarray:
    rest = log_power[..., times < 0] if (times < 0).any() else log_power[..., :1]
    change = log_power - rest.mean(axis=-1, keepdims=True)
    b = {band: BAND_NAMES.index(band) for band in ("alpha", "beta", "gamma")}
    return 0.5 * (change[:, b["beta"]] + change[:, b["gamma"]]) - change[:, b["alpha"]]


def synthesize_eeg(drives: Drives, subject: Subject, rng: np.random.Generator, sfreq: float = SFREQ) -> EEG:
    log_power = region_log_power(drives, subject)
    n = int(round(len(drives.times) / drives.rate * sfreq))
    t = drives.times[0] + np.arange(n) / sfreq

    sources = np.zeros((len(REGIONS), n))
    for b, band in enumerate(BAND_NAMES):
        low, high = (subject.iaf - 2.0, subject.iaf + 2.0) if band == "alpha" else BANDS[band]
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
        carrier = sosfiltfilt(sos, rng.standard_normal((len(REGIONS), n)), axis=1)
        carrier /= carrier.std(axis=1, keepdims=True)
        log_envelope = np.stack([np.interp(t, drives.times, row) for row in log_power[:, b]])
        sources += np.exp(0.5 * log_envelope) * carrier
    sources += _evoked(drives, t, sfreq)

    blinks = _blinks(rng, t, sfreq, subject)
    scalp = lead_field() @ sources + _background(rng, n, sfreq, subject) + blinks
    return EEG(
        data=scalp.astype(np.float32), sfreq=sfreq, t0=float(t[0]), log_power=log_power, blinks=blinks.astype(np.float32))


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _evoked(drives: Drives, t: np.ndarray, sfreq: float) -> np.ndarray:
    n = len(t)
    out = np.zeros((len(REGIONS), n))
    k = np.arange(int(1.2 * sfreq)) / sfreq

    def add(train: np.ndarray, kernel: np.ndarray, regions: dict[str, float]) -> None:
        response = fftconvolve(train, kernel)[:n]
        for region, weight in regions.items():
            out[REGION_KEYS.index(region)] += weight * response

    sounds = drives.audio_onsets
    if len(sounds.times):
        gaps = np.diff(sounds.times, prepend=-np.inf)
        size = sounds.strengths * (1.0 - np.exp(-gaps / AUDITORY_ADAPTATION_S))
        n1_p2 = AUDITORY_ERP_UV * (-_gauss(k, 0.10, 0.025) + 0.7 * _gauss(k, 0.20, 0.04))
        add(impulses(sounds.times, size, n, sfreq, t[0]), n1_p2, AUDITORY_ERP_REGIONS)

    sights = drives.visual_onsets
    if len(sights.times):
        p1_n1 = VISUAL_ERP_UV * (0.8 * _gauss(k, 0.10, 0.02) - _gauss(k, 0.17, 0.03))
        add(impulses(sights.times, sights.strengths, n, sfreq, t[0]), p1_n1, VISUAL_ERP_REGIONS)
        arousal = np.interp(sights.times, drives.times, drives.arousal)
        size = sights.strengths * (0.3 + 0.35 * (1.0 + arousal))
        add(impulses(sights.times, size, n, sfreq, t[0]), LPP_UV * _gauss(k, 0.55, 0.18), LPP_REGIONS)
    return out


@lru_cache(maxsize=1)
def _spatial_mixing() -> np.ndarray:
    v = unit_vectors()
    angles = np.arccos(np.clip(v @ v.T, -1.0, 1.0))
    mix = np.exp(-(angles**2) / (2 * 0.5**2))
    return mix / np.linalg.norm(mix, axis=1, keepdims=True)


def _background(rng: np.random.Generator, n: int, sfreq: float, subject: Subject) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal((len(CHANNELS), n)), axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    spectrum /= np.sqrt(np.maximum(freqs, 1.0))
    pink = np.fft.irfft(spectrum, n=n, axis=1)
    pink /= pink.std(axis=1, keepdims=True)
    background = BACKGROUND_UV * subject.amplitude_scale * (_spatial_mixing() @ pink)
    sensor = SENSOR_NOISE_UV * rng.standard_normal((len(CHANNELS), n))
    return subject.noise_scale * (background + sensor)


def _blinks(rng: np.random.Generator, t: np.ndarray, sfreq: float, subject: Subject) -> np.ndarray:
    duration = t[-1] - t[0]
    count = rng.poisson(subject.blink_rate * duration)
    times = rng.uniform(t[0], t[-1], count)
    train = impulses(times, rng.uniform(*BLINK_UV, count), len(t), sfreq, t[0])
    kernel = _gauss(np.arange(int(0.5 * sfreq)) / sfreq, 0.15, 0.06)
    waveform = fftconvolve(train, kernel)[: len(t)]
    # Blinks are strongest just above the eyes and fade with distance.
    eyes = np.array([0.0, 1.0, -0.3]) / np.linalg.norm([0.0, 1.0, -0.3])
    angles = np.arccos(np.clip(unit_vectors() @ eyes, -1.0, 1.0))
    gains = np.exp(-angles / 0.6)
    return (gains / gains.max())[:, None] * waveform[None, :]
