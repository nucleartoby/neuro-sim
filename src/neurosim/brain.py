from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_POSITIONS = {
    "Fp1": (90, -18), "Fp2": (90, 18),
    "F7": (90, -54), "F3": (60, -39), "Fz": (45, 0), "F4": (60, 39), "F8": (90, 54),
    "T7": (90, -90), "C3": (45, -90), "Cz": (0, 0), "C4": (45, 90), "T8": (90, 90),
    "P7": (90, -126), "P3": (60, -141), "Pz": (45, 180), "P4": (60, 141), "P8": (90, 126),
    "O1": (90, -162), "O2": (90, 162),}

CHANNELS = tuple(_POSITIONS)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),}

BAND_NAMES = tuple(BANDS)

ASYMMETRY_PAIRS = (
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T7", "T8"),
    ("C3", "C4"), ("P7", "P8"), ("P3", "P4"), ("O1", "O2"),)

_EQUATOR_RADIUS = 0.45


@dataclass(frozen=True)
class Region:
    key: str
    label: str
    electrodes: tuple[str, ...]
    spread: float  # radians how far its field reaches across the scalp


REGIONS = (
    Region("frontal_left", "Left prefrontal cortex", ("Fp1", "F3", "F7"), 0.40),
    Region("frontal_right", "Right prefrontal cortex", ("Fp2", "F4", "F8"), 0.40),
    Region("frontal_midline", "Medial frontal cortex", ("Fz",), 0.40),
    Region("temporal_left", "Left auditory cortex", ("T7",), 0.40),
    Region("temporal_right", "Right auditory cortex", ("T8",), 0.40),
    Region("central", "Sensorimotor cortex", ("C3", "Cz", "C4"), 0.40),
    Region("parietal", "Parietal cortex", ("P3", "Pz", "P4"), 0.40),
    Region("occipital", "Visual cortex", ("O1", "O2"), 0.40),)

REGION_KEYS = tuple(r.key for r in REGIONS)

LIMBIC = Region("limbic", "Amygdala / limbic system", (), 0.0)

SYSTEMS = ("Sensory cortex", "Emotion and evaluation", "Movement and integration")

REGION_SYSTEM = {
    "temporal_left": 0, "temporal_right": 0, "occipital": 0,
    "frontal_left": 1, "frontal_right": 1, "frontal_midline": 1, "limbic": 1,
    "central": 2, "parietal": 2,}


def unit_vectors(names: tuple[str, ...] = CHANNELS) -> np.ndarray:
    theta = np.radians([_POSITIONS[n][0] for n in names])
    azimuth = np.radians([_POSITIONS[n][1] for n in names])
    return np.stack(
        [np.sin(theta) * np.sin(azimuth), np.sin(theta) * np.cos(azimuth), np.cos(theta)], axis=1)


def topo_positions(names: tuple[str, ...] = CHANNELS) -> np.ndarray:
    theta = np.array([_POSITIONS[n][0] for n in names], dtype=float)
    azimuth = np.radians([_POSITIONS[n][1] for n in names])
    r = theta / 90.0 * _EQUATOR_RADIUS
    return np.stack([r * np.sin(azimuth), r * np.cos(azimuth)], axis=1)


def region_positions() -> np.ndarray:
    pos = topo_positions()
    index = {name: i for i, name in enumerate(CHANNELS)}
    return np.array([pos[[index[e] for e in r.electrodes]].mean(axis=0) for r in REGIONS])


def lead_field() -> np.ndarray:
    channels = unit_vectors()
    gains = np.zeros((len(CHANNELS), len(REGIONS)))
    for j, region in enumerate(REGIONS):
        centres = unit_vectors(region.electrodes)
        angles = np.arccos(np.clip(channels @ centres.T, -1.0, 1.0))
        g = np.exp(-(angles**2) / (2 * region.spread**2)).max(axis=1)
        gains[:, j] = g / g.max()
    return gains


class ScalpMap:

    def __init__(self, resolution: int = 64, smoothing: float = 1e-3):
        self.resolution = resolution
        self.electrodes = topo_positions()
        axis = np.linspace(-0.5, 0.5, resolution)
        gx, gy = np.meshgrid(axis, axis[::-1])
        self.inside = np.hypot(gx, gy) <= 0.5
        points = np.stack([gx[self.inside], gy[self.inside]], axis=1)

        n = len(self.electrodes)
        kernel = _thin_plate(_distances(self.electrodes, self.electrodes)) + smoothing * np.eye(n)
        poly = np.hstack([np.ones((n, 1)), self.electrodes])
        system = np.block([[kernel, poly], [poly.T, np.zeros((3, 3))]])
        coef = np.linalg.solve(system, np.vstack([np.eye(n), np.zeros((3, n))]))
        basis = np.hstack(
            [_thin_plate(_distances(points, self.electrodes)), np.ones((len(points), 1)), points])
        self.weights = basis @ coef

    def image(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.inside.shape, np.nan)
        out[self.inside] = self.weights @ np.asarray(values, dtype=float)
        return out


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _thin_plate(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r**2 * np.log(r), 0.0)
