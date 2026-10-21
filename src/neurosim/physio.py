"""Body signals: heart rate and skin conductance, both driven mainly by arousal.

Arousal speeds the heart (with breathing-related ripple on top) and raises skin
conductance, both as a slow tonic level and as discrete skin-conductance responses
that become more frequent when arousal is high or something new happens
(Boucsein, 2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .appraisal import EmotionTrace
from .dsp import leaky
from .subject import Subject

BREATHING_HZ = 0.25
SCR_LATENCY_S = 2.0
SCR_RISE_S = 0.75
SCR_DECAY_S = 3.0


@dataclass
class Physio:
    heart_rate: np.ndarray  # bpm, on the emotion trace's time grid
    skin_conductance: np.ndarray  # µS


def simulate_physio(emotion: EmotionTrace, subject: Subject, rng: np.random.Generator) -> Physio:
    rate, t = emotion.rate, emotion.times
    n = len(t)
    rest = emotion.arousal[t < 0].mean() if (t < 0).any() else emotion.arousal[0]

    def lagged(x: np.ndarray, tau: float) -> np.ndarray:
        return x[0] + leaky(x - x[0], tau, rate)

    wander = leaky(rng.standard_normal(n), 5.0, rate)
    wander /= wander.std() + 1e-9
    heart_rate = (
        subject.hr_baseline
        + subject.hr_gain * (lagged(emotion.arousal, 3.0) - rest)
        + 1.5 * np.sin(2 * np.pi * BREATHING_HZ * t + rng.uniform(0, 2 * np.pi))
        + 1.0 * wander)

    tonic = subject.scl_baseline * np.exp(0.25 * (lagged(emotion.arousal, 10.0) - rest))
    scr_rate = 0.03 + 0.25 * np.maximum(emotion.arousal, 0.0) + 0.5 * emotion.novelty  # per second
    fired = rng.random(n) < scr_rate / rate
    sizes = subject.scr_gain * rng.lognormal(np.log(0.15), 0.5, n) * fired
    k = np.arange(int(15 * rate)) / rate - SCR_LATENCY_S
    shape = np.where(k > 0, np.exp(-k / SCR_DECAY_S) - np.exp(-k / SCR_RISE_S), 0.0)
    phasic = fftconvolve(sizes, shape / shape.max())[:n]
    skin_conductance = tonic + phasic + 0.005 * rng.standard_normal(n)
    return Physio(heart_rate=heart_rate, skin_conductance=skin_conductance)
