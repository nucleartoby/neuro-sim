from __future__ import annotations

from pathlib import Path

import av
import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..dsp import impulses, smooth, to_grid
from . import FEATURE_RATE, Events, Stimulus

SAMPLE_RATE = 22050
HOP = 512  # onset analysis ~23 ms
TONAL_FFT = 4096
TONAL_HOP = 2048
KEY_WINDOW_S = 10.0  # harmonic context used to judge key and mode

MAJOR_PROFILE = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
MINOR_PROFILE = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

ROUGHNESS_PEAKS = 24  # strongest spectral peaks considered per frame


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def audio_stimulus(path: Path) -> Stimulus:
    y = load_audio(path)
    if y is None or len(y) < SAMPLE_RATE // 2:
        raise ValueError(f"{path} has no usable audio")
    features, events, info = audio_features(y)
    return Stimulus(
        name=path.stem,
        modality="audio",
        duration=len(y) / SAMPLE_RATE,
        features=features,
        events=events,
        info=info,
        source=path,)


def audio_features(y: np.ndarray, sr: int = SAMPLE_RATE, rate: float = FEATURE_RATE,
                    n_frames: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Events], dict[str, str]]:
    n = n_frames if n_frames is not None else int(round(len(y) / sr * rate))
    frame_rate = sr / HOP

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=HOP)
    level_db = 20 * np.log10(rms + 1e-6)
    loudness = np.clip((level_db + 45.0) / 35.0, 0.0, 1.0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    onset_times = t[onset_frames]
    density = smooth(impulses(onset_times, np.ones(len(onset_times)), len(t), frame_rate), 2.0, frame_rate) * frame_rate
    speed = np.clip(np.log2(np.maximum(density, 1e-3) / 0.5) / 4.0, 0.0, 1.0)

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    lags = np.arange(tempogram.shape[0]) / frame_rate
    beat_lags = (lags >= 0.3) & (lags <= 1.5)  # 40–200 bpm
    pulse = np.clip((tempogram[beat_lags].max(axis=0) - 0.1) / 0.6, 0.0, 1.0)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    brightness = np.clip(np.log10(np.maximum(centroid, 1.0) / 500.0), 0.0, 1.0)

    # Harmony is judged on the harmonic part of the sound with drums separated out
    spectrum = np.abs(librosa.stft(y, n_fft=TONAL_FFT, hop_length=TONAL_HOP))
    harmonic, _ = librosa.decompose.hpss(spectrum, kernel_size=17)
    t_tonal = librosa.times_like(spectrum, sr=sr, hop_length=TONAL_HOP)
    chroma = librosa.feature.chroma_stft(S=harmonic**2, sr=sr, n_fft=TONAL_FFT, hop_length=TONAL_HOP, tuning=0.0)
    window = max(1, int(round(KEY_WINDOW_S * sr / TONAL_HOP)))
    mode = mode_curve(chroma, window)
    roughness = spectral_roughness(harmonic, librosa.fft_frequencies(sr=sr, n_fft=TONAL_FFT))
    roughness = np.clip((roughness - 0.03) / 0.09, 0.0, 1.0)
    tonal_weight = np.clip(2.0 * np.interp(t_tonal, t, loudness), 0.0, 1.0)

    features = {
        "loudness": to_grid(t, smooth(loudness, 0.3, frame_rate), n, rate),
        "speed": to_grid(t, speed, n, rate),
        "pulse": to_grid(t, smooth(pulse, 1.0, frame_rate), n, rate),
        "mode": to_grid(t_tonal, mode * tonal_weight, n, rate),
        "brightness": to_grid(t, smooth(brightness, 0.5, frame_rate), n, rate),
        "dissonance": to_grid(t_tonal, smooth(roughness, 1.0, sr / TONAL_HOP) * tonal_weight, n, rate),}
    onset_norm = np.clip(onset_env / (np.percentile(onset_env, 99) + 1e-9), 0.0, 1.0)
    events = {"audio_onset": Events(times=onset_times, strengths=np.clip(onset_norm[onset_frames], 0.05, 1.0))}

    info = {
        "Notes per second": f"{len(onset_times) / (len(y) / sr):.1f}",
        "Key": estimate_key(chroma),
        "Average level": f"{np.median(level_db):.0f} dBFS",}
    return features, events, info


def mode_curve(chroma: np.ndarray, window: int) -> np.ndarray:
    profile = uniform_filter1d(chroma, size=window, axis=1, mode="nearest")
    r_major = _key_correlations(profile, MAJOR_PROFILE).max(axis=0)
    r_minor = _key_correlations(profile, MINOR_PROFILE).max(axis=0)
    return np.tanh(5.0 * (r_major - r_minor))


def spectral_roughness(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    out = np.zeros(magnitude.shape[1])
    for j, column in enumerate(magnitude.T):
        if column.max() <= 0:
            continue
        peaks, _ = find_peaks(column, height=column.max() * 0.02)
        peaks = peaks[np.argsort(column[peaks])[-ROUGHNESS_PEAKS:]]
        if len(peaks) < 2:
            continue
        f, a = freqs[peaks], column[peaks]
        a = a / np.linalg.norm(a)
        f1, f2 = np.meshgrid(f, f)
        s = 0.24 / (0.0207 * np.minimum(f1, f2) + 18.96)
        d = np.abs(f2 - f1)
        pair = np.outer(a, a) * (np.exp(-3.5 * s * d) - np.exp(-5.75 * s * d))
        out[j] = np.triu(pair, 1).sum()
    return out


def estimate_key(chroma: np.ndarray) -> str:
    profile = chroma.sum(axis=1, keepdims=True)
    if profile.sum() <= 0:
        return "unknown"
    major = _key_correlations(profile, MAJOR_PROFILE)[:, 0]
    minor = _key_correlations(profile, MINOR_PROFILE)[:, 0]
    if major.max() >= minor.max():
        return f"{PITCH_CLASSES[int(major.argmax())]} major"
    return f"{PITCH_CLASSES[int(minor.argmax())]} minor"


def _key_correlations(profile: np.ndarray, template: np.ndarray) -> np.ndarray:
    keys = np.stack([np.roll(template, k) for k in range(12)])
    keys = (keys - keys.mean(axis=1, keepdims=True)) / keys.std(axis=1, keepdims=True)
    centred = profile - profile.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True) + 1e-9
    return keys @ (centred / scale) / 12.0
