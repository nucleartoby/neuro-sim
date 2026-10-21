from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from scipy.ndimage import gaussian_filter1d

from ..brain import BAND_NAMES, BANDS, CHANNELS, LIMBIC, REGION_KEYS, REGION_SYSTEM, SYSTEMS, ScalpMap
from ..emotion import CIRCUMPLEX, NEUTRAL_RADIUS, emotion_label
from ..stimulus import FEATURE_LABELS
from .theme import LIGHT, SYSTEM_COLOURS_LIGHT, diverging_cmap, sequential_cmap, use_style

if TYPE_CHECKING:
    from ..decoder import Evaluation
    from ..pipeline import Run

FELT_LABEL = "Felt (simulated)"
DECODED_LABEL = "Decoded from EEG"
TIME_LABEL = "Time from stimulus onset (s)"


def emotion_figure(run: Run) -> Figure:
    use_style()
    sim = run.simulation
    t = sim.emotion.times
    decoded = run.decoded_on_grid()
    fig = plt.figure(figsize=(12, 8.8))
    _header(fig, run, "What the simulated participant felt, and what an EEG decoder could read back")

    fig.text(
        0.06, 0.885, "What the model perceived: the features that pushed the emotion hardest, with their average push",
        color=LIGHT["ink2"], fontsize=9,)
    
    grid = fig.add_gridspec(
        4, 3, height_ratios=[0.75, 1, 1, 0.75], width_ratios=[1, 1, 0.95],
        left=0.06, right=0.97, top=0.81, bottom=0.07, hspace=0.75, wspace=0.3,)

    _drivers(fig, grid[0, :2], run)
    ax_v = fig.add_subplot(grid[1, :2])
    _emotion_series(ax_v, t, sim.emotion.valence, decoded[:, 0], "Valence: unpleasant (−1) to pleasant (+1)")
    ax_v.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, handlelength=1.6, borderaxespad=0.2)
    ax_a = fig.add_subplot(grid[2, :2], sharex=ax_v)
    _emotion_series(ax_a, t, sim.emotion.arousal, decoded[:, 1], "Arousal: calm (−1) to excited (+1)")
    ax_a.set_xlabel(TIME_LABEL)

    during = t >= 0
    ax_c = fig.add_subplot(grid[:3, 2])
    _circumplex(ax_c, sim.emotion.valence[during], sim.emotion.arousal[during], decoded[during, 0], decoded[during, 1])

    ax_hr = fig.add_subplot(grid[3, 0])
    _body_series(ax_hr, t, sim.physio.heart_rate, "Heart rate (bpm)")
    ax_eda = fig.add_subplot(grid[3, 1])
    _body_series(ax_eda, t, sim.physio.skin_conductance, "Skin conductance (µS)")
    for ax in (ax_hr, ax_eda):
        ax.set_xlabel(TIME_LABEL)
    _agreement_note(fig.add_subplot(grid[3, 2]), run)
    return fig


def brain_figure(run: Run) -> Figure:
    use_style()
    sim = run.simulation
    feats = run.features
    fig = plt.figure(figsize=(12, 8.8))
    _header(fig, run, "Simulated brain activity: scalp EEG, band power and region activity")
    grid = fig.add_gridspec(
        3, 2, width_ratios=[0.75, 1.45], height_ratios=[1.05, 1, 1],
        left=0.06, right=0.97, top=0.83, bottom=0.07, hspace=0.6, wspace=0.42,)
    
    _eeg_excerpt(fig.add_subplot(grid[:, 0]), run)

    change = np.median(feats.band_power[feats.during_stimulus()], axis=0)
    limit = max(0.05, float(np.percentile(np.abs(change), 98)))
    maps = grid[0, 1].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.12], wspace=0.15)
    scalp = ScalpMap()
    for b, band in enumerate(BAND_NAMES):
        ax = fig.add_subplot(maps[0, b])
        image = draw_scalp(ax, scalp, change[:, b], limit)
        lo, hi = BANDS[band]
        ax.set_title(f"{band.title()}\n{lo:g}–{hi:g} Hz", loc="center", fontsize=9)
    _percent_colorbar(fig, fig.add_subplot(maps[0, 5]), image, limit)

    fig.text(
        0.425, 0.885, "Band power during the stimulus vs rest (red = more power, blue = less)",
        color=LIGHT["ink2"], fontsize=9,)

    ax_regions = fig.add_subplot(grid[1, 1])
    limit_act = max(0.2, float(np.percentile(np.abs(sim.activation), 99)))
    ax_regions.imshow(
        sim.activation, aspect="auto", cmap=diverging_cmap(), vmin=-limit_act, vmax=limit_act, interpolation="nearest",
        extent=(sim.emotion.times[0], sim.emotion.times[-1], len(sim.activation), 0),)
    
    labels = sim.region_labels[:-1] + ["Limbic system (deep: not in EEG)"]
    ax_regions.set_yticks(np.arange(len(labels)) + 0.5, labels, fontsize=8)
    ax_regions.set_title("Simulated activity by brain region (red = busier than at rest)")
    ax_regions.grid(False)
    ax_regions.axvline(0, color=LIGHT["ink"], lw=0.8)

    ax_alpha = fig.add_subplot(grid[2, 1], sharex=ax_regions)
    alpha = gaussian_filter1d(feats.band_power[:, :, BAND_NAMES.index("alpha")], sigma=3, axis=0).T
    limit_alpha = max(0.05, float(np.percentile(np.abs(alpha), 99)))
    half_step = (feats.times[1] - feats.times[0]) / 2
    ax_alpha.imshow(
        alpha, aspect="auto", cmap=diverging_cmap(), vmin=-limit_alpha, vmax=limit_alpha, interpolation="nearest",
        extent=(feats.times[0] - half_step, feats.times[-1] + half_step, len(CHANNELS), 0),)
    
    ax_alpha.set_yticks(np.arange(len(CHANNELS))[::2] + 0.5, CHANNELS[::2], fontsize=8)
    ax_alpha.set_title("Alpha power by electrode vs rest (blue = alpha suppressed = cortex engaged)")
    ax_alpha.grid(False)
    ax_alpha.axvline(0, color=LIGHT["ink"], lw=0.8)
    ax_alpha.set_xlabel(TIME_LABEL)
    return fig


def neurons_figure(run: Run) -> Figure:
    use_style()
    sim = run.simulation
    spikes = sim.spikes
    t = sim.emotion.times
    keys = list(REGION_KEYS) + [LIMBIC.key]
    colour = [SYSTEM_COLOURS_LIGHT[REGION_SYSTEM[key]] for key in keys]
    per_region = spikes.n_neurons // len(keys)

    fig = plt.figure(figsize=(12, 8.8))
    _header(fig, run, f"Simulated neurons: every spike from {per_region} model neurons in each brain region")
    ax = fig.add_axes((0.2, 0.07, 0.77, 0.79))
    _rest_band(ax, t)
    ax.eventplot(
        [spikes.of(i) for i in range(spikes.n_neurons)], lineoffsets=np.arange(spikes.n_neurons), linelengths=0.8,
        linewidths=0.6, colors=[colour[r] for r in spikes.region],)
    for k in range(1, len(keys)):
        ax.axhline(k * per_region - 0.5, color=LIGHT["grid"], lw=0.8)
    ax.axvline(0, color=LIGHT["ink"], lw=0.8)
    ax.set_ylim(spikes.n_neurons - 0.5, -0.5)
    ax.set_yticks(np.arange(len(keys)) * per_region + per_region / 2 - 0.5, sim.region_labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(TIME_LABEL)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in SYSTEM_COLOURS_LIGHT]
    ax.legend(handles, SYSTEMS, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, borderaxespad=0.2)
    return fig


def evaluation_figure(evaluation: Evaluation) -> Figure:
    use_style()
    m = evaluation.metrics
    split = m["split"]
    fig = plt.figure(figsize=(12, 8.8))
    fig.text(0.06, 0.955, "How well can emotion be read from the simulated EEG?", fontsize=15, weight="bold")
    fig.text(
        0.06, 0.92,
        f"{split['folds']}-fold cross-validation over {split['subjects']} participants × {split['stimuli']} stimuli: "
        "every score comes from people and stimuli the decoder never trained on.",
        color=LIGHT["ink2"], fontsize=9.5,)
    
    fig.text(
        0.06, 0.895,
        f"Quadrant (happy / tense / sad / calm side of the circumplex) correct {m['quadrant_accuracy']:.0%}, "
        f"chance {m['quadrant_chance']:.0%}.",
        color=LIGHT["ink2"], fontsize=9.5,)

    top = fig.add_gridspec(1, 4, left=0.06, right=0.97, top=0.80, bottom=0.5, wspace=0.38)
    for i, name in enumerate(("valence", "arousal")):
        ax = fig.add_subplot(top[0, i])
        ax.hexbin(
            evaluation.y_true[:, i], evaluation.y_pred[:, i], gridsize=36, extent=(-1, 1, -1, 1),
            cmap=sequential_cmap(), mincnt=1, linewidths=0,)
        
        ax.plot([-1, 1], [-1, 1], color=LIGHT["axis"], lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Felt {name}")
        ax.set_ylabel(f"Decoded {name}")
        s = m[name]
        ax.set_title(f"{name.title()}: r = {s['r']:.2f}")
        ax.text(
            0.03, 0.97, f"high/low correct {s['sign_accuracy']:.0%}\n(chance {s['chance']:.0%})",
            transform=ax.transAxes, va="top", fontsize=8, color=LIGHT["ink2"],)

    example = _most_dynamic_trial(evaluation)
    rows = evaluation.trial == example
    sub = top[0, 2:].subgridspec(2, 1, hspace=0.65)
    for i, name in enumerate(("valence", "arousal")):
        ax = fig.add_subplot(sub[i])
        ax.plot(evaluation.times[rows], evaluation.y_true[rows, i], color=LIGHT["felt"], label=FELT_LABEL)
        ax.plot(evaluation.times[rows], evaluation.y_pred[rows, i], color=LIGHT["decoded"], label=DECODED_LABEL)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Example held-out recording: {name}", fontsize=9)
        if i == 0:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.15), ncols=2, borderaxespad=0)
        else:
            ax.set_xlabel(TIME_LABEL)

    fig.text(
        0.06, 0.41,
        "Where each band tracks emotion (correlation with felt emotion): valence shows up as frontal alpha "
        "asymmetry, arousal as widespread alpha suppression",
        color=LIGHT["ink2"], fontsize=9.5,)
    bottom = fig.add_gridspec(
        2, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], left=0.1, right=0.93, top=0.37, bottom=0.03, wspace=0.08, hspace=0.15,)
    scalp = ScalpMap()
    limit = max(0.1, float(np.abs(evaluation.correlation).max()))
    for row, name in enumerate(("valence", "arousal")):
        for b, band in enumerate(BAND_NAMES):
            ax = fig.add_subplot(bottom[row, b])
            image = draw_scalp(ax, scalp, evaluation.correlation[row, :, b], limit)
            if row == 0:
                ax.set_title(band.title(), loc="center", fontsize=9)
            if b == 0:
                ax.text(-0.72, 0, name.title(), rotation=90, va="center", ha="center", fontsize=9, color=LIGHT["ink"])
    bar = fig.colorbar(image, cax=fig.add_subplot(bottom[:, 5]))
    bar.set_label("correlation (r)", color=LIGHT["ink2"])
    bar.outline.set_visible(False)
    return fig


def draw_scalp(ax: Axes, scalp: ScalpMap, values: np.ndarray, limit: float, dots: bool = True):
    image = ax.imshow(
        scalp.image(values), extent=(-0.5, 0.5, -0.5, 0.5), cmap=diverging_cmap(), vmin=-limit, vmax=limit,
        interpolation="bilinear", origin="upper",)
    outline = dict(fill=False, ec=LIGHT["ink2"], lw=0.9)
    ax.add_patch(Circle((0, 0), 0.5, **outline))
    ax.add_patch(Polygon([[-0.06, 0.495], [0, 0.575], [0.06, 0.495]], closed=False, **outline))
    for side in (-1, 1):
        ax.add_patch(Polygon([[side * 0.5, 0.08], [side * 0.54, 0.05], [side * 0.54, -0.07], [side * 0.5, -0.1]], closed=False, **outline))
    if dots:
        ax.scatter(*scalp.electrodes.T, s=3, color=LIGHT["ink"], linewidths=0)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.56, 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    return image


def _header(fig: Figure, run: Run, subtitle: str) -> None:
    stimulus = run.simulation.stimulus
    fig.text(0.06, 0.955, stimulus.name, fontsize=15, weight="bold")
    details = f"{stimulus.modality} · {stimulus.duration:.0f} s · virtual participant #{run.simulation.subject.seed}"
    fig.text(0.06, 0.925, f"{subtitle}   ({details})", color=LIGHT["ink2"], fontsize=9.5)


def _rest_band(ax: Axes, t: np.ndarray) -> None:
    ax.axvspan(t[0], 0, color=LIGHT["rest"], lw=0, zorder=0)
    ax.text(t[0] + 0.4, 0.04, "rest", transform=ax.get_xaxis_transform(), color=LIGHT["muted"], fontsize=8)
    ax.set_xlim(t[0], t[-1])


def _emotion_series(ax: Axes, t: np.ndarray, felt: np.ndarray, decoded: np.ndarray, title: str) -> None:
    _rest_band(ax, t)
    ax.axhline(0, color=LIGHT["axis"], lw=0.8)
    ax.plot(t, felt, color=LIGHT["felt"], label=FELT_LABEL)
    ax.plot(t, decoded, color=LIGHT["decoded"], label=DECODED_LABEL)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title(title)


def _body_series(ax: Axes, t: np.ndarray, values: np.ndarray, title: str) -> None:
    _rest_band(ax, t)
    ax.plot(t, values, color=LIGHT["felt"], lw=1.2)
    ax.set_title(title, fontsize=9)


def _drivers(fig: Figure, cell, run: Run, count: int = 4) -> None:
    stimulus = run.simulation.stimulus
    contributions = run.simulation.emotion.contributions
    ranked = sorted(contributions, key=lambda k: -(abs(contributions[k][0]) + abs(contributions[k][1])))[:count]
    sub = cell.subgridspec(1, count, wspace=0.3)
    for i, name in enumerate(ranked):
        ax = fig.add_subplot(sub[0, i])
        ax.plot(stimulus.times, stimulus.features[name], color=LIGHT["ink2"], lw=1.1)
        ax.set_ylim((-1.05, 1.05) if name == "mode" else (-0.03, 1.03))
        ax.set_yticks([-1, 0, 1] if name == "mode" else [0, 1])
        ax.set_xlim(0, stimulus.duration)
        ax.tick_params(labelsize=7)
        ax.set_title(FEATURE_LABELS[name], fontsize=8.5, pad=14)
        v, a = contributions[name]
        ax.text(0, 1.04, f"valence {v:+.2f} · arousal {a:+.2f}", transform=ax.transAxes, fontsize=7.5, color=LIGHT["muted"])


def _circumplex(ax: Axes, felt_v, felt_a, decoded_v, decoded_a) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=LIGHT["axis"], lw=0.8))
    ax.add_patch(Circle((0, 0), NEUTRAL_RADIUS, fill=True, fc=LIGHT["rest"], ec="none"))
    ax.plot([-1, 1], [0, 0], color=LIGHT["axis"], lw=0.8)
    ax.plot([0, 0], [-1, 1], color=LIGHT["axis"], lw=0.8)
    for label, angle in CIRCUMPLEX:
        x, y = 1.15 * np.cos(np.radians(angle)), 1.1 * np.sin(np.radians(angle))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=LIGHT["ink2"])
    for v, a, colour in ((felt_v, felt_a, LIGHT["felt"]), (decoded_v, decoded_a, LIGHT["decoded"])):
        keep = ~np.isnan(v)
        if keep.sum() < 2:
            continue
        ax.plot(v[keep], a[keep], color=colour, lw=1.4, alpha=0.9)
        ax.scatter(v[keep][-1], a[keep][-1], s=55, color=colour, edgecolor=LIGHT["surface"], linewidth=2, zorder=5)
        ax.scatter(v[keep][0], a[keep][0], s=28, facecolor=LIGHT["surface"], edgecolor=colour, linewidth=1.5, zorder=5)
    ax.set_title("Path through emotion space\nx = valence, y = arousal; open dot = start", loc="center")


def _agreement_note(ax: Axes, run: Run) -> None:
    ax.axis("off")
    g = run.agreement
    fv, fa = g["valence"]["felt_mean"], g["arousal"]["felt_mean"]
    dv, da = g["valence"]["decoded_mean"], g["arousal"]["decoded_mean"]
    lines = [
        "During the stimulus, on average:",
        f"Felt: {emotion_label(fv, fa)} (valence {fv:+.2f}, arousal {fa:+.2f})",
        f"Decoded: {emotion_label(dv, da)} (valence {dv:+.2f}, arousal {da:+.2f})",
        "",]
    for name in ("valence", "arousal"):
        r = g[name]["r"]
        tracking = "steady, so no correlation" if np.isnan(r) else f"tracked over time with r = {r:.2f}"
        lines.append(f"{name.title()}: mean error {g[name]['mean_abs_error']:.2f}, {tracking}")
    ax.text(0, 0.95, "\n".join(lines), va="top", fontsize=8.5, color=LIGHT["ink2"], linespacing=1.5)


def _eeg_excerpt(ax: Axes, run: Run, seconds: float = 10.0, spacing: float = 50.0) -> None:
    sim = run.simulation
    eeg = sim.eeg
    during = sim.emotion.times >= 0
    peak = sim.emotion.times[during][np.argmax(sim.emotion.arousal[during])]
    start = float(np.clip(peak - seconds / 2, eeg.t0, eeg.times[-1] - seconds))
    i0 = int((start - eeg.t0) * eeg.sfreq)
    i1 = i0 + int(seconds * eeg.sfreq)
    t = eeg.times[i0:i1]
    for k in range(len(CHANNELS)):
        trace = np.clip(eeg.data[k, i0:i1], -1.6 * spacing, 1.6 * spacing)  # clip large blinks as EEG viewers do
        ax.plot(t, trace - k * spacing, color=LIGHT["ink2"], lw=0.5)
    ax.set_yticks(-np.arange(len(CHANNELS)) * spacing, CHANNELS, fontsize=8)
    ax.set_ylim(-(len(CHANNELS) - 0.2) * spacing, 1.2 * spacing)
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(TIME_LABEL)
    ax.set_title(f"Simulated scalp EEG around peak arousal ({t[0]:.0f}–{t[-1]:.0f} s)")
    x = t[-1] - 0.3
    ax.plot([x, x], [0.4 * spacing, 1.4 * spacing], color=LIGHT["ink"], lw=1.2, clip_on=False)
    ax.text(x - 0.15, 0.9 * spacing, f"{spacing:.0f} µV", ha="right", va="center", fontsize=7.5, color=LIGHT["ink2"])


def _percent_colorbar(fig: Figure, cax: Axes, image, limit: float) -> None:
    bar = fig.colorbar(image, cax=cax)
    ticks = [-limit, 0, limit]
    bar.set_ticks(ticks, labels=[f"{(10**v - 1) * 100:+.0f}%" if v else "0" for v in ticks])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7.5)


def _most_dynamic_trial(evaluation: Evaluation) -> int:
    trials = np.unique(evaluation.trial)
    spread = [np.ptp(evaluation.y_true[evaluation.trial == t], axis=0).sum() for t in trials]
    return int(trials[int(np.argmax(spread))])
