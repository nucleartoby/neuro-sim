from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .appraisal import EmotionTrace, appraise
from .brain import LIMBIC, REGIONS
from .eeg import EEG, SFREQ, Drives, region_activation, synthesize_eeg
from .neurons import Spikes, simulate_spikes
from .physio import Physio, simulate_physio
from .stimulus import Stimulus
from .subject import Subject

BASELINE_S = 10.0


@dataclass
class Simulation:
    stimulus: Stimulus
    subject: Subject
    emotion: EmotionTrace
    drives: Drives
    eeg: EEG
    physio: Physio
    activation: np.ndarray
    spikes: Spikes | None = None

    @property
    def region_labels(self) -> list[str]:
        return [region.label for region in REGIONS] + [LIMBIC.label]


def simulate(stimulus: Stimulus, subject: Subject, *, baseline_s: float = BASELINE_S, seed: int | None = None,
    sfreq: float = SFREQ, neurons: bool = True,) -> Simulation:
    rng = np.random.default_rng(seed)
    emotion = appraise(stimulus, subject, baseline_s, rng)
    drives = sensory_drives(stimulus, emotion)
    eeg = synthesize_eeg(drives, subject, rng, sfreq)
    physio = simulate_physio(emotion, subject, rng)
    activation = np.vstack([region_activation(eeg.log_power, emotion.times), limbic_activity(emotion)])
    spikes = simulate_spikes(activation, emotion.times, emotion.rate, stimulus.events, rng) if neurons else None
    return Simulation(stimulus, subject, emotion, drives, eeg, physio, activation, spikes)


def sensory_drives(stimulus: Stimulus, emotion: EmotionTrace) -> Drives:
    n_rest = int((emotion.times < 0).sum())
    rest = np.zeros(n_rest)
    silent = np.zeros(stimulus.n_frames)
    features = stimulus.features
    return Drives(
        rate=emotion.rate,
        times=emotion.times,
        valence=emotion.valence,
        arousal=emotion.arousal,
        auditory=np.r_[rest, features.get("loudness", silent)],
        visual=np.r_[rest, np.ones(stimulus.n_frames) if stimulus.has_visual else silent],
        motion=np.r_[rest, features.get("motion", silent)],
        audio_onsets=stimulus.event("audio_onset"),
        visual_onsets=stimulus.event("visual_onset"),)


def limbic_activity(emotion: EmotionTrace) -> np.ndarray:
    rest = emotion.times < 0
    a_rest = emotion.arousal[rest].mean() if rest.any() else emotion.arousal[0]
    v_rest = emotion.valence[rest].mean() if rest.any() else emotion.valence[0]
    return 0.7 * (emotion.arousal - a_rest) + 0.3 * np.abs(emotion.valence - v_rest) + 0.4 * emotion.novelty
