from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps

from ..dsp import impulses, smooth, to_grid
from . import FEATURE_RATE, Events, Stimulus
from .audio import SAMPLE_RATE, audio_features, load_audio

ANALYSIS_WIDTH = 160
CUT_THRESHOLD = 0.5


def frame_stats(rgb: np.ndarray) -> dict[str, float]:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hsv = rgb_to_hsv(rgb)
    hue, sat = hsv[..., 0], hsv[..., 1]

    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    lean = np.sum(sat * np.cos(2 * np.pi * (hue - 1 / 12))) / (np.sum(sat) + 1e-6)
    warmth = 0.5 + 0.5 * lean * min(1.0, sat.mean() / 0.25)

    return {
        "luminance": float(luma.mean()),
        "saturation": float(sat.mean()),
        "colourfulness": float(np.clip(colourfulness / 0.45, 0.0, 1.0)),
        "warmth": float(np.clip(warmth, 0.0, 1.0)),
        "contrast": float(np.clip(luma.std() / 0.3, 0.0, 1.0)),}


def image_stimulus(path: Path, duration: float = 15.0, rate: float = FEATURE_RATE) -> Stimulus:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        im.thumbnail((256, 256))
        stats = frame_stats(np.asarray(im))
    n = int(round(duration * rate))
    features = {name: np.full(n, value) for name, value in stats.items()}
    return Stimulus(
        name=path.stem,
        modality="image",
        duration=n / rate,
        features=features,
        events={"visual_onset": Events(np.array([0.0]), np.array([1.0]))},
        info={"Size": f"{width} × {height}", "Viewing time": f"{duration:.0f} s"},
        source=path,)


def video_stimulus(path: Path, rate: float = FEATURE_RATE) -> Stimulus:
    times, stats, lumas, hists = [], [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        width = ANALYSIS_WIDTH
        height = max(2, int(round(ANALYSIS_WIDTH * stream.height / stream.width / 2)) * 2)
        next_t = 0.0
        last_t = 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            last_t = frame.time
            if frame.time + 1e-6 < next_t:
                continue
            next_t += 1.0 / rate
            rgb = frame.to_ndarray(format="rgb24", width=width, height=height)
            times.append(frame.time)
            stats.append(frame_stats(rgb))
            lumas.append(rgb.astype(np.float32).mean(axis=2) / 255.0)
            hists.append(_colour_histogram(rgb))
        fps = float(stream.average_rate) if stream.average_rate else rate
    if not times:
        raise ValueError(f"{path} has no decodable video frames")

    duration = last_t + 1.0 / fps
    n = int(round(duration * rate))
    t = np.asarray(times)
    features = {name: to_grid(t, np.array([s[name] for s in stats]), n, rate) for name in stats[0]}

    hist_change = np.r_[0.0, np.abs(np.diff(np.asarray(hists), axis=0)).sum(axis=1)]
    is_cut = hist_change > CUT_THRESHOLD
    motion = np.r_[0.0, [np.abs(b - a).mean() for a, b in zip(lumas[:-1], lumas[1:])]]
    motion = np.interp(t, t[~is_cut], motion[~is_cut])
    features["motion"] = to_grid(t, np.clip(smooth(motion, 0.5, rate) / 0.06, 0.0, 1.0), n, rate)

    cut_times = t[is_cut]
    cuts_per_second = smooth(impulses(cut_times, np.ones(len(cut_times)), n, rate), 2.0, rate) * rate
    features["cut_rate"] = np.clip(cuts_per_second / 0.8, 0.0, 1.0)

    onset_times = np.r_[0.0, cut_times]
    onset_strengths = np.r_[1.0, np.clip(hist_change[is_cut], 0.3, 1.0)]
    events = {"visual_onset": Events(onset_times, onset_strengths)}
    info = {"Frame rate": f"{fps:.0f} fps", "Scene cuts": str(len(cut_times))}

    y = load_audio(path)
    if y is not None and len(y) >= SAMPLE_RATE // 2:
        audio, audio_events, audio_info = audio_features(y, n_frames=n)
        features.update(audio)
        events.update(audio_events)
        info.update(audio_info)

    return Stimulus(
        name=path.stem,
        modality="video",
        duration=n / rate,
        features=features,
        events=events,
        info=info,
        source=path,)


def _colour_histogram(rgb: np.ndarray, bins: int = 4) -> np.ndarray:
    quantised = (rgb.reshape(-1, 3).astype(np.int32) * bins) // 256
    index = (quantised[:, 0] * bins + quantised[:, 1]) * bins + quantised[:, 2]
    counts = np.bincount(index, minlength=bins**3).astype(float)
    return counts / counts.sum()
