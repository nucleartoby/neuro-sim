from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .brain import LIMBIC, REGION_KEYS
from .dsp import impulses
from .stimulus import Events

NEURONS_PER_REGION = 32
SPIKE_RATE = 100.0  # Hz time resolution of the spiking model 10 ms bins
MAX_RATE_HZ = 120.0
ONSET_ADAPTATION_S = 1.0  # a burst right after another is smaller as for evoked potentials

ONSET_TUNING = {
    "temporal_left": ("audio_onset", 1.0, 0.02, 0.04),
    "temporal_right": ("audio_onset", 1.0, 0.02, 0.04),
    "occipital": ("visual_onset", 1.0, 0.06, 0.08),
    "parietal": ("visual_onset", 0.3, 0.10, 0.10),}


@dataclass
class Spikes:
    region: np.ndarray
    times: np.ndarray
    offsets: np.ndarray
    population_rate: np.ndarray

    @property
    def n_neurons(self) -> int:
        return len(self.region)

    def of(self, neuron: int) -> np.ndarray:
        return self.times[self.offsets[neuron] : self.offsets[neuron + 1]]


def simulate_spikes(activation: np.ndarray, times: np.ndarray, rate: float,
    events: dict[str, Events], rng: np.random.Generator, per_region: int = NEURONS_PER_REGION,) -> Spikes:
    keys = list(REGION_KEYS) + [LIMBIC.key]
    n_frames = activation.shape[1]
    bins_per_frame = int(round(SPIKE_RATE / rate))
    n_bins = n_frames * bins_per_frame
    t_bins = times[0] + np.arange(n_bins) / SPIKE_RATE

    spike_times, counts = [], []
    population = np.zeros((len(keys), n_frames))
    for r, key in enumerate(keys):
        drive = np.interp(t_bins, times, activation[r])
        resting = np.clip(rng.lognormal(np.log(2.5), 0.6, per_region), 0.3, 12.0)
        gain = rng.uniform(0.6, 1.4, per_region)
        rates = resting[:, None] * np.exp(gain[:, None] * drive[None, :])
        if key in ONSET_TUNING:
            kind, strength, latency, length = ONSET_TUNING[key]
            bursts = _bursts(events.get(kind, Events.empty()), t_bins[0], n_bins, latency, length)
            tuning = strength * 100.0 * rng.uniform(0.0, 1.0, per_region) ** 2
            rates = rates + tuning[:, None] * bursts[None, :]
        rates = np.minimum(rates, MAX_RATE_HZ)
        population[r] = rates.mean(axis=0).reshape(n_frames, bins_per_frame).mean(axis=1)

        fired = rng.random(rates.shape) < -np.expm1(-rates / SPIKE_RATE)
        for row in fired:
            bins = np.flatnonzero(row)
            spike_times.append(t_bins[bins] + rng.random(len(bins)) / SPIKE_RATE)
            counts.append(len(bins))

    return Spikes(
        region=np.repeat(np.arange(len(keys)), per_region),
        times=np.concatenate(spike_times),
        offsets=np.r_[0, np.cumsum(counts)],
        population_rate=population,)


def _bursts(events: Events, t0: float, n: int, latency: float, length: float) -> np.ndarray:
    if not len(events.times):
        return np.zeros(n)
    gaps = np.diff(events.times, prepend=-np.inf)
    size = events.strengths * (1.0 - np.exp(-gaps / ONSET_ADAPTATION_S))
    train = impulses(events.times + latency, size, n, SPIKE_RATE, t0)
    k = np.arange(int(5 * length * SPIKE_RATE)) / SPIKE_RATE
    return fftconvolve(train, np.exp(-k / length))[:n]
