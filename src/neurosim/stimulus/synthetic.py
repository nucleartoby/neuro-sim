from __future__ import annotations

import numpy as np

from ..dsp import impulses, smooth
from . import FEATURE_RATE, Events, Stimulus

MODALITIES = ("audio", "video", "image")


def random_stimulus(rng: np.random.Generator, duration: float = 40.0, modality: str | None = None,
    name: str = "synthetic", rate: float = FEATURE_RATE,) -> Stimulus:
    modality = modality or str(rng.choice(MODALITIES, p=[0.5, 0.3, 0.2]))
    n = int(round(duration * rate))
    features: dict[str, np.ndarray] = {}
    events: dict[str, Events] = {}

    if modality == "audio" or (modality == "video" and rng.random() < 0.8):
        audio, onsets = _audio_track(rng, n, rate)
        features.update(audio)
        events["audio_onset"] = onsets
    if modality == "image":
        features.update(_still(rng, n))
        events["visual_onset"] = Events(np.array([0.0]), np.array([1.0]))
    elif modality == "video":
        visual, cuts = _video_track(rng, n, rate)
        features.update(visual)
        events["visual_onset"] = cuts

    return Stimulus(name=name, modality=modality, duration=n / rate, features=features, events=events)


def _sections(rng: np.random.Generator, base: float, spread: float, jitter: float, n: int, rate: float) -> np.ndarray:
    n_sections = max(1, int(rng.poisson(n / rate / 15.0)) + 1)
    bounds = np.sort(rng.integers(0, n, n_sections - 1))
    levels = base + spread * rng.standard_normal(n_sections)
    curve = levels[np.searchsorted(bounds, np.arange(n), side="right")]
    wobble = smooth(rng.standard_normal(n), 1.5, rate)
    return smooth(curve, 1.0, rate) + jitter * wobble / (wobble.std() + 1e-9)


def _audio_track(rng: np.random.Generator, n: int, rate: float) -> tuple[dict[str, np.ndarray], Events]:
    speed = np.clip(_sections(rng, rng.uniform(0.05, 0.95), 0.08, 0.03, n, rate), 0, 1)
    loudness = np.clip(_sections(rng, rng.uniform(0.2, 0.95), 0.15, 0.05, n, rate), 0, 1)
    pulse_base = 0.4 * speed.mean() + 0.6 * rng.uniform(0.0, 1.0)
    features = {
        "loudness": loudness,
        "speed": speed,
        "pulse": np.clip(_sections(rng, pulse_base, 0.15, 0.05, n, rate), 0, 1),
        "mode": np.clip(_sections(rng, rng.uniform(-1.0, 1.0), 0.3, 0.05, n, rate), -1, 1),
        "brightness": np.clip(_sections(rng, rng.uniform(0.15, 0.95), 0.1, 0.03, n, rate), 0, 1),
        "dissonance": np.clip(_sections(rng, rng.beta(3.0, 4.0), 0.1, 0.03, n, rate), 0, 1),}
    # Sound onsets at the rate implied by speed
    onsets_per_second = 0.5 * 2.0 ** (4.0 * speed)
    count = rng.poisson(onsets_per_second.sum() / rate)
    frames = rng.choice(n, size=count, p=onsets_per_second / onsets_per_second.sum())
    times = np.sort((frames + rng.random(count)) / rate)
    level = np.interp(times, np.arange(n) / rate, loudness)
    return features, Events(times, rng.uniform(0.3, 1.0, count) * (0.3 + 0.7 * level))


def _still(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    saturation = rng.beta(2.0, 3.0)
    values = {
        "luminance": rng.uniform(0.05, 0.95),
        "saturation": saturation,
        "colourfulness": np.clip(0.8 * saturation + rng.normal(0.0, 0.1), 0, 1),
        "warmth": rng.uniform(0.0, 1.0),
        "contrast": rng.uniform(0.1, 0.9),}
    return {name: np.full(n, value) for name, value in values.items()}


def _video_track(rng: np.random.Generator, n: int, rate: float) -> tuple[dict[str, np.ndarray], Events]:
    duration = n / rate
    cuts_per_second = rng.uniform(0.02, 0.5)
    cut_times = np.sort(rng.uniform(0.5, duration, rng.poisson(cuts_per_second * duration)))
    shot = np.searchsorted(cut_times, np.arange(n) / rate, side="right")
    n_shots = len(cut_times) + 1

    look = _still(rng, 1)
    features = {}
    for name, value in look.items():
        per_shot = np.clip(value[0] + 0.12 * rng.standard_normal(n_shots), 0, 1)
        features[name] = smooth(per_shot[shot], 0.05, rate)
    features["motion"] = np.clip(_sections(rng, rng.beta(1.5, 2.5), 0.15, 0.05, n, rate), 0, 1)
    rate_curve = smooth(impulses(cut_times, np.ones(len(cut_times)), n, rate), 2.0, rate) * rate
    features["cut_rate"] = np.clip(rate_curve / 0.8, 0, 1)

    onsets = Events(np.r_[0.0, cut_times], np.r_[1.0, rng.uniform(0.3, 1.0, len(cut_times))])
    return features, onsets
