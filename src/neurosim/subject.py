from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .stimulus import AUDIO_FEATURES, VISUAL_FEATURES


@dataclass(frozen=True)
class Subject:
    seed: int
    baseline_valence: float  # resting mood
    baseline_arousal: float
    valence_gain: float  # emotional reactivity
    arousal_gain: float
    tau_valence: float  # seconds emotional inertia
    tau_arousal: float
    emotion_noise: float 
    taste: dict[str, tuple[float, float]] = field(repr=False)
    iaf: float  # individual alpha peak frequency
    alpha_scale: float  # alpha power relative to the population
    amplitude_scale: float  # overall EEG amplitude
    faa_trait: float  # trait frontal alpha asymmetry ln right - ln left
    noise_scale: float  # background activity and sensor noise
    blink_rate: float  # blinks per second
    # Physiology.
    hr_baseline: float  # resting heart rate, bpm
    hr_gain: float  # bpm per unit arousal
    scl_baseline: float  # resting skin conductance, µS
    scr_gain: float  # size of skinconductance responses

    def summary(self) -> dict[str, str]:
        return {
            "Resting mood": f"valence {self.baseline_valence:+.2f}, arousal {self.baseline_arousal:+.2f}",
            "Reactivity": f"valence ×{self.valence_gain:.2f}, arousal ×{self.arousal_gain:.2f}",
            "Emotional inertia": f"{self.tau_valence:.1f} s (valence), {self.tau_arousal:.1f} s (arousal)",
            "Alpha peak": f"{self.iaf:.1f} Hz",
            "Trait frontal asymmetry": f"{self.faa_trait:+.2f}",
            "Resting heart rate": f"{self.hr_baseline:.0f} bpm",}


def sample_subject(seed: int = 0) -> Subject:
    rng = np.random.default_rng(seed)
    taste = {
        name: (
            float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0)),
            float(np.clip(rng.normal(1.0, 0.2), 0.3, 1.8)),)
        for name in AUDIO_FEATURES + VISUAL_FEATURES}
    return Subject(
        seed=seed,
        baseline_valence=float(rng.normal(0.05, 0.1)),
        baseline_arousal=float(rng.normal(-0.15, 0.1)),
        valence_gain=float(rng.lognormal(0.0, 0.25)),
        arousal_gain=float(rng.lognormal(0.0, 0.25)),
        tau_valence=float(rng.uniform(2.5, 5.0)),
        tau_arousal=float(rng.uniform(1.5, 3.0)),
        emotion_noise=float(rng.uniform(0.05, 0.12)),
        taste=taste,
        iaf=float(np.clip(rng.normal(10.0, 0.8), 8.0, 12.5)),
        alpha_scale=float(rng.lognormal(0.0, 0.3)),
        amplitude_scale=float(rng.lognormal(0.0, 0.2)),
        faa_trait=float(rng.normal(0.0, 0.15)),
        noise_scale=float(rng.lognormal(0.0, 0.25)),
        blink_rate=float(rng.uniform(0.05, 0.25)),
        hr_baseline=float(np.clip(rng.normal(70.0, 8.0), 50.0, 95.0)),
        hr_gain=float(rng.uniform(4.0, 10.0)),
        scl_baseline=float(rng.lognormal(np.log(5.0), 0.4)),
        scr_gain=float(rng.lognormal(0.0, 0.4)),)
