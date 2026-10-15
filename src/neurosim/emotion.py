from __future__ import annotations

import numpy as np

CIRCUMPLEX = (
    ("happy", 0.0),
    ("excited", 45.0),
    ("tense", 90.0),
    ("afraid", 135.0),
    ("unhappy", 180.0),
    ("sad", 225.0),
    ("tired", 270.0),
    ("calm", 315.0),)

NEUTRAL_RADIUS = 0.2


def emotion_label(valence: float, arousal: float) -> str:
    if np.hypot(valence, arousal) < NEUTRAL_RADIUS:
        return "neutral"
    angle = np.degrees(np.arctan2(arousal, valence)) % 360.0
    sector = int(((angle + 22.5) % 360.0) // 45.0)
    return CIRCUMPLEX[sector][0]


def quadrant(valence: np.ndarray, arousal: np.ndarray) -> np.ndarray:
    v = np.asarray(valence)
    a = np.asarray(arousal)
    return np.where(a >= 0, np.where(v >= 0, 0, 1), np.where(v < 0, 2, 3))
