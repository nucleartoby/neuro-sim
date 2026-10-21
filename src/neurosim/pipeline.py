from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mne


from .brain import CHANNELS
from .decoder import DEFAULT_MODEL_PATH, Decoder, Evaluation, build_dataset, train_decoder
from .features import EEGFeatures, extract_features, window_average
from .simulate import BASELINE_S, Simulation, simulate
from .stimulus import Stimulus, load_stimulus
from .subject import sample_subject

Log = Callable[[str], None]
MIN_CHANGE_FOR_R = 0.15  # felt emotion must vary at least this much to report a correlation


@dataclass
class Run:
    simulation: Simulation
    features: EEGFeatures
    decoded: np.ndarray 
    agreement: dict 
    decoder_metrics: dict

    def decoded_on_grid(self) -> np.ndarray:
        t = self.simulation.emotion.times
        out = np.full((len(t), 2), np.nan)
        inside = (t >= self.features.times[0]) & (t <= self.features.times[-1])
        for i in range(2):
            out[inside, i] = np.interp(t[inside], self.features.times, self.decoded[:, i])
        return out


def analyse(media: Stimulus | str | Path, decoder: Decoder, *, subject_seed: int = 0, noise_seed: int = 0,
    image_seconds: float = 15.0, baseline_s: float = BASELINE_S,) -> Run:
    stimulus = media if isinstance(media, Stimulus) else load_stimulus(media, image_duration=image_seconds)
    sim = simulate(stimulus, sample_subject(subject_seed), baseline_s=baseline_s, seed=noise_seed)
    features = extract_features(sim.eeg.cleaned(), sim.eeg.sfreq, sim.eeg.t0)
    decoded = decoder.predict_series(features.values)

    during = features.during_stimulus()
    felt = np.stack(
        [window_average(sim.emotion.times, s, features.times[during]) for s in (sim.emotion.valence, sim.emotion.arousal)],
        axis=1,)
    return Run(sim, features, decoded, agreement(felt, decoded[during]), decoder.metrics)


def agreement(felt: np.ndarray, decoded: np.ndarray) -> dict:
    out: dict = {"windows": len(felt)}
    for i, name in enumerate(("valence", "arousal")):
        changes = len(felt) > 2 and felt[:, i].std() >= MIN_CHANGE_FOR_R
        out[name] = {
            "felt_mean": float(felt[:, i].mean()),
            "decoded_mean": float(decoded[:, i].mean()),
            "mean_abs_error": float(np.abs(felt[:, i] - decoded[:, i]).mean()),
            "r": float(np.corrcoef(felt[:, i], decoded[:, i])[0, 1]) if changes else np.nan,}
    return out


def load_or_train_decoder(path: Path = DEFAULT_MODEL_PATH, log: Log = print) -> Decoder:
    if path.exists():
        try:
            return Decoder.load(path)
        except ValueError as exc:
            log(f"{exc}.")
    log("No trained decoder yet: training one on simulated recordings (about 30 s, once)...")
    decoder, _ = train_and_save(path, log=log)
    return decoder


def train_and_save(path: Path = DEFAULT_MODEL_PATH, *, n_subjects: int = 32, n_stimuli: int = 40,
    duration: float = 40.0, seed: int = 0, stimuli: list[Stimulus] | None = None, log: Log = print,) -> tuple[Decoder, Evaluation]:
    count = len(stimuli) if stimuli else n_stimuli
    log(f"Simulating {n_subjects} participants × {count} stimuli...")
    data = build_dataset(
        n_subjects, n_stimuli, duration, seed, stimuli, progress=lambda done, total: _progress(log, done, total))
    log(f"Training on {len(data.X):,} EEG windows with 4-fold cross-validation (new people and new stimuli)...")
    decoder, evaluation = train_decoder(data, seed=seed)
    decoder.save(path)
    from .report.static import evaluation_figure

    evaluation_figure(evaluation).savefig(path.with_name("decoder_evaluation.png"), dpi=150)
    path.with_name("decoder_metrics.json").write_text(json.dumps(jsonable(evaluation.metrics), indent=2))
    m = evaluation.metrics
    log(
        f"Held-out accuracy: valence r = {m['valence']['r']:.2f}, arousal r = {m['arousal']['r']:.2f}, "
        f"quadrant {m['quadrant_accuracy']:.0%} (chance {m['quadrant_chance']:.0%}). "
        f"Saved to {path}, with an evaluation figure beside it.")
    return decoder, evaluation


def _progress(log: Log, done: int, total: int) -> None:
    if done == total or done % max(1, total // 4) == 0:
        log(f"  {done}/{total} participants simulated")


def write_outputs(run: Run, out_dir: Path, log: Log = print) -> dict[str, Path]:
    from .report.html import write_report
    from .report.static import brain_figure, emotion_figure, neurons_figure

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "emotion figure": out_dir / "emotion.png",
        "brain figure": out_dir / "brain.png",
        "timeline data": out_dir / "timeline.csv",
        "simulated EEG": out_dir / "simulated_eeg_raw.fif",
        "summary": out_dir / "summary.json",}
    emotion_figure(run).savefig(paths["emotion figure"], dpi=150)
    brain_figure(run).savefig(paths["brain figure"], dpi=150)
    if run.simulation.spikes is not None:
        paths["neurons figure"] = out_dir / "neurons.png"
        neurons_figure(run).savefig(paths["neurons figure"], dpi=150)
    _write_timeline(run, paths["timeline data"])
    _write_fif(run, paths["simulated EEG"])
    paths["summary"].write_text(json.dumps(jsonable(_summary(run)), indent=2))

    media = run.simulation.stimulus.source
    media_copy = None
    if media is not None:
        media_copy = out_dir / "media" / media.name
        media_copy.parent.mkdir(exist_ok=True)
        if media.resolve() != media_copy.resolve():
            shutil.copy2(media, media_copy)
    paths["report"] = write_report(run, out_dir / "report.html", media_copy)
    import matplotlib.pyplot as plt

    plt.close("all")
    return paths


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _summary(run: Run) -> dict:
    sim = run.simulation
    stimulus = sim.stimulus
    during = sim.emotion.times >= 0
    return {
        "media": stimulus.name,
        "modality": stimulus.modality,
        "duration_s": round(stimulus.duration, 2),
        "media_info": stimulus.info,
        "participant_seed": sim.subject.seed,
        "participant": sim.subject.summary(),
        "mean_felt_emotion": {
            "valence": round(float(sim.emotion.valence[during].mean()), 3),
            "arousal": round(float(sim.emotion.arousal[during].mean()), 3),},
        "decoder_agreement": run.agreement,}


def _write_timeline(run: Run, path: Path) -> None:
    sim = run.simulation
    decoded = run.decoded_on_grid()
    columns = {
        "time_s": sim.emotion.times,
        "felt_valence": sim.emotion.valence,
        "felt_arousal": sim.emotion.arousal,
        "decoded_valence": decoded[:, 0],
        "decoded_arousal": decoded[:, 1],
        "heart_rate_bpm": sim.physio.heart_rate,
        "skin_conductance_uS": sim.physio.skin_conductance,}
    for label, values in zip(sim.region_labels, sim.activation):
        columns[f"activity: {label}"] = values
    if sim.spikes is not None:
        for label, values in zip(sim.region_labels, sim.spikes.population_rate):
            columns[f"firing rate (Hz): {label}"] = values
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*columns.values()):
            writer.writerow("" if np.isnan(v) else f"{v:.4f}" for v in row)


def _write_fif(run: Run, path: Path) -> None:
    eeg = run.simulation.eeg
    info = mne.create_info(list(CHANNELS), eeg.sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg.data.astype(np.float64) * 1e-6, info, verbose="error")  # µV -> V
    raw.set_montage("standard_1020", verbose="error")
    rest = -eeg.t0
    stimulus = raw.times[-1] - rest  # up to the last sample, so the annotation stays inside the recording
    raw.set_annotations(mne.Annotations(onset=[0.0, rest], duration=[rest, stimulus], description=["rest", "stimulus"]))
    raw.save(path, overwrite=True, verbose="error")
