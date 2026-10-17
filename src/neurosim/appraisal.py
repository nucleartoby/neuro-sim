from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dsp import impulses, leaky
from .stimulus import Stimulus
from .subject import Subject

APPRAISAL = {
    "mode": (0.0, 0.45, 0.00),
    "speed": (0.5, 0.15, 0.40),
    "loudness": (0.6, -0.05, 0.35),
    "pulse": (0.5, 0.10, 0.20),
    "brightness": (0.55, 0.15, 0.15),
    "dissonance": (0.4, -0.40, 0.20),
    "luminance": (0.45, 0.50, -0.20),
    "saturation": (0.35, 0.15, 0.40),
    "colourfulness": (0.35, 0.20, 0.10),
    "warmth": (0.5, 0.10, 0.15),
    "contrast": (0.5, 0.00, 0.10),
    "motion": (0.3, 0.00, 0.35),
    "cut_rate": (0.25, 0.00, 0.25),}

NOVELTY_AROUSAL = 0.35
NOVELTY_TAU = 2.0
LOUDNESS_JUMP = 0.15


@dataclass
class EmotionTrace:
    rate: float
    times: np.ndarray  # seconds from stimulus onset negative resting baseline
    valence: np.ndarray  # felt emotion the simulation's ground truth
    arousal: np.ndarray
    valence_target: np.ndarray  # what the stimulus is pushing toward
    arousal_target: np.ndarray
    novelty: np.ndarray
    contributions: dict[str, tuple[float, float]]  # mean valence, arousal push per feature


def centred(name: str, values: np.ndarray) -> np.ndarray:
    typical = APPRAISAL[name][0]
    low = -1.0 if name == "mode" else 0.0
    return np.where(values >= typical, (values - typical) / (1.0 - typical), (values - typical) / (typical - low))


def appraise(stimulus: Stimulus, subject: Subject, baseline_s: float, rng: np.random.Generator) -> EmotionTrace:
    rate = stimulus.rate
    n_rest = int(round(baseline_s * rate))
    n = n_rest + stimulus.n_frames
    times = (np.arange(n) - n_rest) / rate

    drive_v = np.zeros(n)
    drive_a = np.zeros(n)
    contributions = {}
    for name, values in stimulus.features.items():
        _, w_v, w_a = APPRAISAL[name]
        taste_v, taste_a = subject.taste[name]
        c = centred(name, values)
        push_v, push_a = w_v * taste_v * c, w_a * taste_a * c
        drive_v[n_rest:] += push_v
        drive_a[n_rest:] += push_a
        contributions[name] = (float(push_v.mean()), float(push_a.mean()))

    novelty = leaky(_novelty_events(stimulus, n, n_rest), NOVELTY_TAU, rate)
    novelty = np.clip(novelty * NOVELTY_TAU * rate, 0.0, 1.0)
    drive_a += NOVELTY_AROUSAL * novelty

    v_target = np.tanh(subject.baseline_valence + subject.valence_gain * drive_v)
    a_target = np.tanh(subject.baseline_arousal + subject.arousal_gain * drive_a)
    valence = _follow(v_target, subject.tau_valence, subject.emotion_noise, rate, rng)
    arousal = _follow(a_target, subject.tau_arousal, subject.emotion_noise, rate, rng)
    return EmotionTrace(rate, times, valence, arousal, v_target, a_target, novelty, contributions)


def _novelty_events(stimulus: Stimulus, n: int, n_rest: int) -> np.ndarray:
    rate = stimulus.rate
    t0 = -n_rest / rate
    events = impulses(np.array([0.0]), np.array([1.0]), n, rate, t0)
    cuts = stimulus.event("visual_onset")
    events += impulses(cuts.times, cuts.strengths, n, rate, t0)
    if stimulus.has_audio:
        lag = max(1, int(round(0.3 * rate)))
        loud = stimulus.features["loudness"]
        rise = np.r_[np.zeros(lag), loud[lag:] - loud[:-lag]]
        bursts = np.where(rise > LOUDNESS_JUMP, rise, 0.0)
        peaks = (bursts > 0) & (bursts >= np.r_[0.0, bursts[:-1]]) & (bursts > np.r_[bursts[1:], 0.0])
        events[n_rest:] += np.where(peaks, np.clip(rise * 2.0, 0.0, 1.0), 0.0)
    return np.minimum(events, 1.0)


def _follow(target: np.ndarray, tau: float, noise: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    dt = 1.0 / rate
    kick = noise * np.sqrt(2.0 * dt / tau) * rng.standard_normal(len(target))
    out = np.empty(len(target))
    x = target[0]
    for i, goal in enumerate(target):
        x += dt / tau * (goal - x) + kick[i]
        x = min(1.0, max(-1.0, x))
        out[i] = x
    return out
