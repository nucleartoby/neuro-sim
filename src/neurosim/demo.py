from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

SAMPLE_RATE = 22050


@dataclass(frozen=True)
class Style:
    bpm: float
    chords: tuple[tuple[int, ...], ...]  # MIDI notes one chord per bar
    harmonics: int  # more harmonics = brighter timbre
    rolloff: float  # how fast harmonics fade
    attack: float  # seconds
    level_db: float  # target loudness RMS, dBFS
    arpeggio: int  # arpeggio notes per beat 0 = none
    drums: str  # "", "steady" or "erratic"


STYLES = {
    # Major, fast, bright, loud and rhythmic: high valence, high arousal.
    "happy": Style(132, ((60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)), 8, 1.0, 0.01, -14, 2, "steady"),
    # Minor, slow, dark and quiet: low valence, low arousal.
    "sad": Style(58, ((57, 60, 64), (50, 53, 57), (52, 55, 59), (57, 60, 64)), 3, 2.2, 0.4, -30, 1, ""),
    # Dissonant clusters, harsh, loud and erratic: low valence, high arousal.
    "tense": Style(104, ((48, 49, 54, 55), (50, 51, 56, 57), (47, 48, 53, 54), (49, 50, 55, 56)), 12, 0.6, 0.005, -12, 4, "erratic"),
    # Major, slow, soft and warm: high valence, low arousal.
    "calm": Style(66, ((53, 57, 60), (58, 62, 65), (60, 64, 67), (53, 57, 60)), 3, 1.8, 0.3, -26, 1, ""),}


def synth_music(style: str, seconds: float = 30.0, sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    s = STYLES[style]
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr))
    beat = 60.0 / s.bpm
    bar = 4 * beat
    for i, start in enumerate(np.arange(0.0, seconds, bar)):
        chord = s.chords[i % len(s.chords)]
        for note in chord:  # sustained pad
            _add_note(out, sr, start, bar, note, s, 0.5 / len(chord))
        _add_note(out, sr, start, bar, chord[0] - 12, s, 0.25)  # bass
        if s.arpeggio:
            step = beat / s.arpeggio
            for k in range(int(round(bar / step))):
                note = chord[k % len(chord)] + 12
                _add_note(out, sr, start + k * step, step * 0.9, note, s, 0.25, decay=step * 2)
        if s.drums:
            _add_drums(out, sr, start, beat, s.drums, rng)
    out = _fade(out, sr)
    rms = np.sqrt(np.mean(out**2)) + 1e-12
    out *= 10 ** (s.level_db / 20) / rms
    return np.tanh(out).astype(np.float32)  # soft limit the loudest peaks


def _add_note(out: np.ndarray, sr: int, start: float, 
            length: float, midi: int, s: Style, gain: float, decay: float | None = None) -> None:
    i0 = int(start * sr)
    n = min(int((length + 0.3) * sr), len(out) - i0)
    if n <= 0:
        return
    t = np.arange(n) / sr
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    wave_ = np.zeros(n)
    for k in range(1, s.harmonics + 1):
        if k * freq < sr / 2:
            wave_ += np.sin(2 * np.pi * k * freq * t) / k**s.rolloff
    envelope = np.minimum(1.0, t / max(s.attack, 1e-3)) * np.where(t < length, 1.0, np.exp(-(t - length) / 0.08))
    if decay is not None:
        envelope *= np.exp(-t / decay)
    out[i0 : i0 + n] += gain * wave_ * envelope


def _add_drums(out: np.ndarray, sr: int, start: float, beat: float, kind: str, rng: np.random.Generator) -> None:
    def hit(at: float, sound: np.ndarray) -> None:
        i0 = int(at * sr)
        n = min(len(sound), len(out) - i0)
        if n > 0:
            out[i0 : i0 + n] += sound[:n]

    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 70 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.12)
    noise = rng.standard_normal(len(t))
    snare = 0.5 * noise * np.exp(-t / 0.06)
    hat = 0.15 * np.diff(noise, prepend=0.0) * np.exp(-t / 0.015)
    if kind == "steady":
        for b in range(4):
            hit(start + b * beat, kick if b % 2 == 0 else snare)
            hit(start + (b + 0.5) * beat, hat)
    else:
        for b in range(8):
            if rng.random() < 0.5:
                hit(start + (b + rng.uniform(-0.2, 0.2)) * beat / 2, 1.4 * (kick + snare))


def _fade(x: np.ndarray, sr: int, seconds: float = 0.5) -> np.ndarray:
    n = int(seconds * sr)
    ramp = np.linspace(0.0, 1.0, n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def synth_image(kind: str, width: int = 640, height: int = 400, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / np.array([height, width])[:, None, None]
    if kind == "sunrise":
        sky = np.stack([1.0 - 0.1 * y, 0.55 + 0.35 * y, 0.15 + 0.25 * y], axis=-1)
        sun = np.exp(-(((x - 0.5) * 1.6) ** 2 + ((y - 0.62) * 2.6) ** 2) / 0.02)[..., None]
        rgb = sky * (1 - sun) + np.array([1.0, 0.95, 0.6]) * sun
        rgb[y > 0.75] = np.array([0.95, 0.45, 0.1]) * (0.85 + 0.15 * rng.random((int((y > 0.75).sum()), 1)))
    elif kind == "storm":
        base = 0.18 + 0.1 * (1 - y)
        clouds = _blur(rng.random((height // 8, width // 8)), height, width)
        grey = base + 0.08 * clouds
        rgb = np.stack([grey * 0.85, grey * 0.9, grey * 1.1], axis=-1)
    else:
        raise ValueError(f"unknown image kind {kind!r}")
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def _blur(small: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)) / 255.0


SHOT_COLOURS = np.array([[0.9, 0.15, 0.1], [1.0, 0.55, 0.0], [0.85, 0.1, 0.5], [0.95, 0.8, 0.1]])


def synth_video(path: Path, seconds: float = 24.0, fps: int = 24, size: tuple[int, int] = (320, 180)) -> None:
    width, height = size
    half = seconds / 2
    sr = 44100
    audio = np.concatenate([synth_music("calm", half, sr), synth_music("tense", half, sr, seed=1)])
    yy, xx = np.mgrid[0:height, 0:width]

    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        sound = container.add_stream("aac", rate=sr, layout="mono")
        chunk = 1024
        audio = np.pad(audio, (0, (-len(audio)) % chunk)).astype(np.float32)
        audio_pos = 0
        for i in range(int(seconds * fps)):
            t = i / fps
            if t < half:
                cx = width * (0.5 + 0.3 * np.sin(0.3 * t))
                cy = height * (0.5 + 0.2 * np.cos(0.25 * t))
                blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 50.0**2))[..., None]
                rgb = np.array([0.25, 0.4, 0.45]) * (1 - blob) + np.array([0.45, 0.65, 0.7]) * blob
            else:
                shot = int((t - half) / 0.6)
                cx = width * (0.5 + 0.45 * np.sin(9.0 * t + shot))
                cy = height * (0.5 + 0.4 * np.cos(7.0 * t))
                square = ((np.abs(xx - cx) < 30) & (np.abs(yy - cy) < 30))[..., None]
                rgb = np.where(square, np.array([1.0, 1.0, 0.9]), SHOT_COLOURS[shot % len(SHOT_COLOURS)])
            frame = av.VideoFrame.from_ndarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
            while audio_pos < min(len(audio), (t + 1 / fps) * sr):
                block = av.AudioFrame.from_ndarray(audio[None, audio_pos : audio_pos + chunk], format="fltp", layout="mono")
                block.sample_rate, block.pts = sr, audio_pos
                for packet in sound.encode(block):
                    container.mux(packet)
                audio_pos += chunk
        for packet in video.encode():
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)


def make_demo_media(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for style in STYLES:
        path = directory / f"{style}_song.wav"
        write_wav(path, synth_music(style))
        paths.append(path)
    for kind in ("sunrise", "storm"):
        path = directory / f"{kind}.png"
        synth_image(kind).save(path)
        paths.append(path)
    path = directory / "calm_then_intense.mp4"
    synth_video(path)
    paths.append(path)
    return paths
