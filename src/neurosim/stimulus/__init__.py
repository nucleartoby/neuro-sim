from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURE_RATE = 10.0  # Hz

AUDIO_FEATURES = ("loudness", "speed", "pulse", "mode", "brightness", "dissonance")
VISUAL_FEATURES = ("luminance", "saturation", "colourfulness", "warmth", "contrast", "motion", "cut_rate")

FEATURE_LABELS = {
    "loudness": "Loudness",
    "speed": "Speed (notes per second)",
    "pulse": "Beat strength",
    "mode": "Major (+) / minor (−)",
    "brightness": "Timbral brightness",
    "dissonance": "Dissonance (roughness)",
    "luminance": "Brightness",
    "saturation": "Colour saturation",
    "colourfulness": "Colourfulness",
    "warmth": "Warm vs cool colours",
    "contrast": "Contrast",
    "motion": "Motion",
    "cut_rate": "Editing pace",}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".aif", ".aiff", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass
class Events:

    times: np.ndarray  # seconds from stimulus onset
    strengths: np.ndarray  # 0..1

    @classmethod
    def empty(cls) -> Events:
        return cls(np.zeros(0), np.zeros(0))


@dataclass
class Stimulus:
    name: str
    modality: str
    duration: float  # seconds
    features: dict[str, np.ndarray]
    events: dict[str, Events] = field(default_factory=dict)  # "audio_onset visual_onset"
    info: dict[str, str] = field(default_factory=dict)
    source: Path | None = None
    rate: float = FEATURE_RATE

    @property
    def n_frames(self) -> int:
        return int(round(self.duration * self.rate))

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.rate

    @property
    def has_audio(self) -> bool:
        return "loudness" in self.features

    @property
    def has_visual(self) -> bool:
        return "luminance" in self.features

    def event(self, kind: str) -> Events:
        return self.events.get(kind, Events.empty())


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    import av

    try:
        with av.open(str(path)) as container:
            return "video" if container.streams.video else "audio"
    except av.FFmpegError as exc:
        raise ValueError(f"{path} is not a media file neurosim can read") from exc


def load_stimulus(path: str | Path, image_duration: float = 15.0) -> Stimulus:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    kind = media_kind(path)
    if kind == "image":
        from .visual import image_stimulus

        return image_stimulus(path, duration=image_duration)
    if kind == "video":
        from .visual import video_stimulus

        return video_stimulus(path)
    from .audio import audio_stimulus

    return audio_stimulus(path)
