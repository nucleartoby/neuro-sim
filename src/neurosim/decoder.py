from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .emotion import quadrant
from .features import STEP_S, extract_features, window_average
from .simulate import BASELINE_S, simulate
from .stimulus import Stimulus
from .stimulus.synthetic import random_stimulus
from .subject import sample_subject

DEFAULT_MODEL_PATH = Path("models/decoder.joblib")
SMOOTHING_S = 3.0
TARGETS = ("valence", "arousal")


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    stimulus: np.ndarray
    trial: np.ndarray
    times: np.ndarray
    band_power: np.ndarray
    names: list[str]


def build_dataset(n_subjects: int = 32, n_stimuli: int = 40, duration: float = 40.0, seed: int = 0,
                  stimuli: Sequence[Stimulus] | None = None, progress: Callable[[int, int], None] | None = None,) -> Dataset:
    rng = np.random.default_rng(seed)
    if stimuli is None:
        stimuli = [random_stimulus(rng, duration, name=f"synthetic-{k}") for k in range(n_stimuli)]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("X", "y", "subject", "stimulus", "trial", "times", "bp")}
    names: list[str] = []
    trial = 0
    for s in range(n_subjects):
        subject = sample_subject(seed * 100_003 + s)
        for k, stimulus in enumerate(stimuli):
            sim = simulate(stimulus, subject, baseline_s=BASELINE_S, seed=int(rng.integers(2**32)), neurons=False)
            feats = extract_features(sim.eeg.cleaned(), sim.eeg.sfreq, sim.eeg.t0)
            keep = feats.during_stimulus()
            centres = feats.times[keep]
            labels = [window_average(sim.emotion.times, sig, centres) for sig in (sim.emotion.valence, sim.emotion.arousal)]
            parts["X"].append(feats.values[keep])
            parts["y"].append(np.stack(labels, axis=1))
            parts["bp"].append(feats.band_power[keep])
            parts["times"].append(centres)
            for key, value in (("subject", s), ("stimulus", k), ("trial", trial)):
                parts[key].append(np.full(keep.sum(), value))
            names = feats.names
            trial += 1
        if progress:
            progress(s + 1, n_subjects)
    return Dataset(
        X=np.vstack(parts["X"]),
        y=np.vstack(parts["y"]),
        subject=np.concatenate(parts["subject"]),
        stimulus=np.concatenate(parts["stimulus"]),
        trial=np.concatenate(parts["trial"]),
        times=np.concatenate(parts["times"]),
        band_power=np.concatenate(parts["bp"]),
        names=names,)


@dataclass
class Decoder:
    model: Pipeline
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)
    smoothing_s: float = SMOOTHING_S

    def predict_series(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(smooth_windows(X, self.smoothing_s)), -1.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "smoothing_s": self.smoothing_s,
                "version": __version__,},path,)

    @classmethod
    def load(cls, path: Path) -> Decoder:
        state = joblib.load(path)
        if state.get("version") != __version__:
            raise ValueError(f"{path} was trained by neurosim {state.get('version')}; retrain it")
        return cls(state["model"], state["feature_names"], state["metrics"], state["smoothing_s"])


@dataclass
class Evaluation:
    metrics: dict
    y_true: np.ndarray  # held out windows
    y_pred: np.ndarray  # decoded
    trial: np.ndarray
    times: np.ndarray
    correlation: np.ndarray  # 2 channels bands. band power vs valence/arousal


def train_decoder(data: Dataset, seed: int = 0, folds: int = 4) -> tuple[Decoder, Evaluation]:
    rng = np.random.default_rng(seed)
    subject_fold = _fold_of(rng, data.subject, folds)
    stimulus_fold = _fold_of(rng, data.stimulus, folds)
    X = np.empty_like(data.X)
    for rows in _trial_slices(data.trial):
        X[rows] = smooth_windows(data.X[rows], SMOOTHING_S)

    y_pred = np.full(data.y.shape, np.nan)
    for k in range(folds):
        train = (subject_fold != k) & (stimulus_fold != k)
        test = (subject_fold == k) & (stimulus_fold == k)
        y_pred[test] = np.clip(_fit(X[train], data.y[train]).predict(X[test]), -1.0, 1.0)
    tested = ~np.isnan(y_pred[:, 0])

    metrics = score(data.y[tested], y_pred[tested])
    metrics["per_trial"] = _trial_score(data.y[tested], y_pred[tested], data.trial[tested])
    metrics["split"] = {
        "folds": folds,
        "subjects": int(len(np.unique(data.subject))),
        "stimuli": int(len(np.unique(data.stimulus))),
        "windows_tested": int(tested.sum()),}
    
    decoder = Decoder(_fit(X, data.y), data.names, metrics)
    correlation = _correlation_maps(data.band_power, data.y)
    evaluation = Evaluation(
        metrics, data.y[tested], y_pred[tested], data.trial[tested], data.times[tested], correlation)
    return decoder, evaluation


def smooth_windows(X: np.ndarray, seconds: float) -> np.ndarray:
    return gaussian_filter1d(X, sigma=seconds / STEP_S, axis=0, mode="nearest")


def _trial_slices(trial: np.ndarray) -> list[slice]:
    starts = np.flatnonzero(np.r_[True, trial[1:] != trial[:-1]])
    stops = np.r_[starts[1:], len(trial)]
    return [slice(a, b) for a, b in zip(starts, stops)]


def _fit(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11), alpha_per_target=True))
    return model.fit(X, y)


def _fold_of(rng: np.random.Generator, ids: np.ndarray, folds: int) -> np.ndarray:
    unique = np.unique(ids)
    fold = np.empty(len(unique), dtype=int)
    fold[rng.permutation(len(unique))] = np.arange(len(unique)) % folds
    return fold[np.searchsorted(unique, ids)]


def _trial_score(y_true: np.ndarray, y_pred: np.ndarray, trial: np.ndarray) -> dict:
    trials = np.unique(trial)
    true_mean = np.array([y_true[trial == t].mean(axis=0) for t in trials])
    pred_mean = np.array([y_pred[trial == t].mean(axis=0) for t in trials])
    return {name: float(np.corrcoef(true_mean[:, i], pred_mean[:, i])[0, 1]) for i, name in enumerate(TARGETS)}


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out: dict = {}
    for i, name in enumerate(TARGETS):
        t, p = y_true[:, i], y_pred[:, i]
        high = t >= 0
        out[name] = {
            "r": float(np.corrcoef(t, p)[0, 1]),
            "r2": float(r2_score(t, p)),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "sign_accuracy": float(np.mean(high == (p >= 0))),
            "chance": float(max(high.mean(), 1 - high.mean())),}

    q_true = quadrant(y_true[:, 0], y_true[:, 1])
    q_pred = quadrant(y_pred[:, 0], y_pred[:, 1])
    out["quadrant_accuracy"] = float(np.mean(q_true == q_pred))
    out["quadrant_chance"] = float(np.bincount(q_true, minlength=4).max() / len(q_true))
    return out


def _correlation_maps(band_power: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between each channel × band power and valence / arousal."""
    flat = band_power.reshape(len(band_power), -1)
    flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-12)
    targets = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-12)
    r = targets.T @ flat / len(flat)
    return r.reshape(2, *band_power.shape[1:])
