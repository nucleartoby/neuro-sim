from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "felt": "#2a78d6",
    "decoded": "#eb6834",
    "rest": "#f0efec",}

SYSTEM_COLOURS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SYSTEM_COLOURS_DARK = ["#3987e5", "#d95926", "#199e70"]

DIVERGING_LIGHT = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#f0efec", "#f3958d", "#e34948", "#a82428", "#681013"]
DIVERGING_DARK = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#383835", "#a82428", "#e34948", "#f3958d", "#ffd4d0"]
SEQUENTIAL_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def diverging_cmap(dark: bool = False) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("neurosim_diverging", DIVERGING_DARK if dark else DIVERGING_LIGHT)


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("neurosim_sequential", SEQUENTIAL_LIGHT)


def lookup_table(anchors: list[str], size: int = 256) -> list[int]:
    rgb = np.array([to_rgb(c) for c in anchors])
    positions = np.linspace(0.0, 1.0, len(anchors))
    samples = np.linspace(0.0, 1.0, size)
    table = np.stack([np.interp(samples, positions, rgb[:, i]) for i in range(3)], axis=1)
    return (np.round(table * 255).astype(int)).reshape(-1).tolist()


def use_style() -> None:
    c = LIGHT
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "axes.edgecolor": c["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": c["ink2"],
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": c["ink"],
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": c["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.labelcolor": c["muted"],
            "ytick.labelcolor": c["muted"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "text.color": c["ink"],
        }
    )
