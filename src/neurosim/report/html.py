from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .. import __version__
from ..brain import BAND_NAMES, BANDS, CHANNELS, LIMBIC, REGION_KEYS, REGION_SYSTEM, SYSTEMS, ScalpMap, region_positions
from ..emotion import CIRCUMPLEX, NEUTRAL_RADIUS
from ..stimulus import FEATURE_LABELS
from .theme import DIVERGING_DARK, DIVERGING_LIGHT, SYSTEM_COLOURS_DARK, lookup_table

if TYPE_CHECKING:
    from ..pipeline import Run

TEMPLATE = Path(__file__).with_name("template.html")
EEG_RESOLUTION_UV = 0.1  # EEG is shipped as int16 in steps of 0.1 µV
SCALP_RESOLUTION = 56
SPIKE_TICK_S = 0.01  # spike times are shipped as integer ticks of 10 ms


def write_report(run: Run, path: Path, media: Path | None) -> Path:
    media_src = os.path.relpath(media, path.parent) if media is not None else None
    payload = json.dumps(build_payload(run, media_src), separators=(",", ":"), allow_nan=False)
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__TITLE__", html.escape(run.simulation.stimulus.name))
    page = page.replace("__DATA__", payload.replace("</", "<\\/"))
    path.write_text(page, encoding="utf-8")
    return path


def build_payload(run: Run, media_src: str | None) -> dict:
    from ..pipeline import jsonable

    sim = run.simulation
    stimulus = sim.stimulus
    emotion = sim.emotion
    decoded = run.decoded_on_grid()
    n_rest = int((emotion.times < 0).sum())

    contributions = emotion.contributions
    ranked = sorted(stimulus.features, key=lambda k: -(abs(contributions[k][0]) + abs(contributions[k][1])))
    features = [
        {
            "key": key,
            "label": FEATURE_LABELS[key],
            "signed": key == "mode",
            "valence": round(contributions[key][0], 3),
            "arousal": round(contributions[key][1], 3),
            "values": _numbers(np.r_[np.full(n_rest, np.nan), stimulus.features[key]]),}
        for key in ranked]

    band_power = median_filter(run.features.band_power, size=(9, 1, 1), mode="nearest")
    band_power = gaussian_filter1d(band_power, sigma=2, axis=0)
    during = run.features.during_stimulus()
    limit = max(0.05, float(np.percentile(np.abs(band_power[during]), 98)))
    scalp = ScalpMap(resolution=SCALP_RESOLUTION)

    metrics = {}
    if run.decoder_metrics:
        m = run.decoder_metrics
        metrics = {
            "valence_r": m["valence"]["r"],
            "arousal_r": m["arousal"]["r"],
            "quadrant": m["quadrant_accuracy"],
            "quadrant_chance": m["quadrant_chance"],
            "subjects": m["split"]["subjects"],
            "stimuli": m["split"]["stimuli"],}

    eeg = sim.eeg
    return {
        "meta": {
            "name": stimulus.name,
            "file": stimulus.source.name if stimulus.source else stimulus.name,
            "modality": stimulus.modality,
            "duration": stimulus.duration,
            "baseline": -float(emotion.times[0]),
            "media": media_src,
            "info": stimulus.info,
            "participant": sim.subject.seed,
            "traits": sim.subject.summary(),
            "agreement": jsonable(run.agreement),
            "decoder": metrics,
            "version": __version__,
        },
        "grid": {"t0": float(emotion.times[0]), "rate": emotion.rate, "n": len(emotion.times)},
        "felt": {"valence": _numbers(emotion.valence), "arousal": _numbers(emotion.arousal)},
        "decoded": {"valence": _numbers(decoded[:, 0]), "arousal": _numbers(decoded[:, 1])},
        "body": {"heart_rate": _numbers(sim.physio.heart_rate, 2), "skin_conductance": _numbers(sim.physio.skin_conductance, 3)},
        "features": features,
        "regions": {
            "labels": sim.region_labels,
            "deep": [False] * (len(sim.region_labels) - 1) + [True],
            "activation": [_numbers(row) for row in sim.activation],
        },
        "eeg": {
            "sfreq": eeg.sfreq,
            "t0": eeg.t0,
            "channels": list(CHANNELS),
            "step": EEG_RESOLUTION_UV,
            "data": _b64(np.clip(np.round(eeg.data / EEG_RESOLUTION_UV), -32767, 32767).astype("<i2")),
        },
        "topo": {
            "times": _numbers(run.features.times),
            "bands": [{"name": b.title(), "range": f"{BANDS[b][0]:g}–{BANDS[b][1]:g} Hz"} for b in BAND_NAMES],
            "limit": limit,
            "values": _b64(band_power.transpose(0, 2, 1).astype("<f4")),  # (windows, bands, channels)
        },
        "scalp": {
            "resolution": SCALP_RESOLUTION,
            "inside": _b64(scalp.inside.astype(np.uint8)),
            "weights": _b64(scalp.weights.astype("<f4")),
            "electrodes": np.round(scalp.electrodes, 4).tolist(),
        },
        "neurons": _neurons(run),
        "palette": {"light": lookup_table(DIVERGING_LIGHT), "dark": lookup_table(DIVERGING_DARK)},
        "circumplex": [[label, angle] for label, angle in CIRCUMPLEX],
        "neutral_radius": NEUTRAL_RADIUS,}


def _neurons(run: Run) -> dict | None:
    sim = run.simulation
    spikes = sim.spikes
    if spikes is None:
        return None
    t0 = float(sim.emotion.times[0])
    ticks = np.round((spikes.times - t0) / SPIKE_TICK_S)
    wide = len(ticks) and ticks.max() >= 2**16
    keys = list(REGION_KEYS) + [LIMBIC.key]
    # Positions on the top down map the scalp map frame the limbic system sits deep, not under the scalp.
    positions = region_positions().round(4).tolist() + [None]
    short = [label.replace(" cortex", "") for label in sim.region_labels[:-1]] + ["Limbic system"]
    return {
        "per_region": spikes.n_neurons // len(keys),
        "regions": [
            {"key": key, "label": label, "system": REGION_SYSTEM[key], "position": position}
            for key, label, position in zip(keys, short, positions)],
        "systems": list(SYSTEMS),
        "colours": SYSTEM_COLOURS_DARK,
        "t0": t0,
        "tick": SPIKE_TICK_S,
        "bits": 32 if wide else 16,
        "spikes": _b64(ticks.astype("<u4" if wide else "<u2")),
        "offsets": _b64(spikes.offsets.astype("<u4")),
        "rates": [_numbers(row, 1) for row in spikes.population_rate],}


def _numbers(values: np.ndarray, digits: int = 3) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")



INDEX_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neuro-sim demo</title>
<style>
:root { color-scheme: dark; --font: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --page: #0b0b0b; --ink: #e8e8e3; --ink2: #a6a59e; --muted: #7b7a74; --line: #34342f; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) { color-scheme: light; --page: #f6f6f2; --ink: #111110; --ink2: #4a4a45; --muted: #7c7b75; --line: #c9c8c0; }
}
:root[data-theme="light"] { color-scheme: light; --page: #f6f6f2; --ink: #111110; --ink2: #4a4a45; --muted: #7c7b75; --line: #c9c8c0; }
* { box-sizing: border-box; border-radius: 0; }
body { margin: 0; background: var(--page); color: var(--ink); font: 13px/1.55 var(--font); }
main { max-width: 1200px; margin: 0 auto; padding: 32px 16px 48px; }
.prompt { margin: 0 0 8px; color: var(--ink2); }
.prompt span { color: var(--muted); }
h1 { margin: 0 0 6px; font-size: 26px; font-weight: 700; }
.lead { margin: 0 0 34px; color: var(--ink2); max-width: 80ch; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 32px 16px; }
.card { position: relative; display: block; border: 1px solid var(--line); padding: 22px 12px 12px; color: inherit; text-decoration: none; }
.card:hover, .card:focus-visible { border-color: var(--ink2); outline: none; }
.card h2 { position: absolute; top: -0.85em; left: 10px; margin: 0; padding: 0 6px; background: var(--page); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; overflow-wrap: anywhere; }
.card h2::before { content: "[ "; color: var(--muted); font-weight: 400; }
.card h2::after { content: " ]"; color: var(--muted); font-weight: 400; }
.card img { display: block; width: 100%; border: 1px solid var(--line); background: #fcfcfb; }
.card p { margin: 10px 0 0; color: var(--ink2); font-size: 12px; }
</style>
</head>
<body>
<main>
  <p class="prompt"><span>$</span> neurosim demo</p>
  <h1>neuro-sim demo</h1>
  <p class="lead">Synthetic songs, images and a video, each experienced by the same virtual participant. Open one to watch the simulated brain respond as it plays.</p>
  <div class="cards">
__CARDS__
  </div>
</main>
</body>
</html>
"""


def write_index(entries: list[tuple[Run, dict[str, Path]]], path: Path) -> Path:
    from ..emotion import emotion_label

    def link(target: Path) -> str:
        return html.escape(os.path.relpath(target, path.parent))

    cards = []
    for run, paths in entries:
        stimulus = run.simulation.stimulus
        g = run.agreement
        felt = emotion_label(g["valence"]["felt_mean"], g["arousal"]["felt_mean"])
        decoded = emotion_label(g["valence"]["decoded_mean"], g["arousal"]["decoded_mean"])
        name = html.escape(stimulus.name)
        cards.append(
            f'    <a class="card" href="{link(paths["report"])}">\n'
            f"      <h2>{name}</h2>\n"
            f'      <img src="{link(paths["emotion figure"])}" alt="Emotion summary for {name}" loading="lazy">\n'
            f"      <p>{stimulus.modality} · {stimulus.duration:.0f} s · felt mostly <b>{felt}</b>;"
            f" the EEG decoder read <b>{decoded}</b></p>\n"
            "    </a>")
    path.write_text(INDEX_PAGE.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    return path
