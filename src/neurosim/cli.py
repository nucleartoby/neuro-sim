from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
import numpy as np

from .decoder import DEFAULT_MODEL_PATH
from .emotion import emotion_label
from .stimulus import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, load_stimulus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neurosim",
        description="Simulate how a brain responds emotionally to a song, video or image, and visualise it.",)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate a virtual participant's response to a media file")
    run.add_argument("media", type=Path, help="an audio, video or image file")
    run.add_argument("-o", "--out", type=Path, help="output folder (default: outputs/<file name>)")
    run.add_argument("--participant", type=int, default=0, help="which virtual participant; each number is a different person")
    run.add_argument("--seed", type=int, default=0, help="seed for moment-to-moment neural noise")
    run.add_argument("--image-seconds", type=float, default=15.0, help="how long an image is viewed (default: 15)")
    run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="trained decoder; trained automatically if missing")
    run.add_argument("--open", action="store_true", help="open the interactive report when done")

    train = sub.add_parser("train", help="train the EEG emotion decoder on simulated recordings")
    train.add_argument("--participants", type=int, default=32, help="virtual participants (default: 32)")
    train.add_argument("--stimuli", type=int, default=40, help="stimuli each participant experiences (default: 40)")
    train.add_argument("--seconds", type=float, default=40.0, help="length of each synthetic stimulus (default: 40)")
    train.add_argument("--media-dir", type=Path, help="also use the media files in this folder as stimuli")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    demo = sub.add_parser("demo", help="synthesise demo songs, images and a video, and report on each")
    demo.add_argument("-o", "--out", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    demo.add_argument("--open", action="store_true", help="open the demo index when done")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "train":
        return _train(args)
    return _demo(args)


def _run(args: argparse.Namespace) -> int:
    from .pipeline import analyse, load_or_train_decoder, write_outputs

    if not args.media.exists():
        print(f"neurosim: {args.media} not found")
        return 1
    decoder = load_or_train_decoder(args.model)
    print(f"Simulating virtual participant #{args.participant} experiencing {args.media.name}...")
    run = analyse(args.media, decoder, subject_seed=args.participant, noise_seed=args.seed, image_seconds=args.image_seconds)
    paths = write_outputs(run, args.out or Path("outputs") / args.media.stem)
    _describe(run)
    print(f"Interactive report: {paths['report']}")
    if args.open:
        webbrowser.open(paths["report"].resolve().as_uri())
    return 0


def _train(args: argparse.Namespace) -> int:
    from .pipeline import train_and_save
    from .stimulus.synthetic import random_stimulus

    stimuli = None
    if args.media_dir:
        extensions = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
        files = sorted(p for p in args.media_dir.iterdir() if p.suffix.lower() in extensions)
        print(f"Extracting features from {len(files)} media files in {args.media_dir}...")
        stimuli = [load_stimulus(p) for p in files]
        rng = np.random.default_rng(args.seed)
        stimuli += [random_stimulus(rng, args.seconds, name=f"synthetic-{k}") for k in range(max(0, args.stimuli - len(stimuli)))]
    train_and_save(
        args.model, n_subjects=args.participants, n_stimuli=args.stimuli, duration=args.seconds, seed=args.seed, stimuli=stimuli)
    return 0


def _demo(args: argparse.Namespace) -> int:
    from .demo import make_demo_media
    from .pipeline import analyse, load_or_train_decoder, write_outputs
    from .report.html import write_index

    print(f"Synthesising demo media in {args.out / 'media'}...")
    media = make_demo_media(args.out / "media")
    decoder = load_or_train_decoder(args.model)
    entries = []
    for path in media:
        run = analyse(path, decoder)
        paths = write_outputs(run, args.out / path.stem)
        entries.append((run, paths))
        _describe(run)
    index = write_index(entries, args.out / "index.html")
    print(f"Demo index: {index}")
    if args.open:
        webbrowser.open(index.resolve().as_uri())
    return 0


def _describe(run) -> None:
    g = run.agreement
    fv, fa = g["valence"]["felt_mean"], g["arousal"]["felt_mean"]
    dv, da = g["valence"]["decoded_mean"], g["arousal"]["decoded_mean"]
    print(
        f"  {run.simulation.stimulus.name}: felt mostly {emotion_label(fv, fa)} (valence {fv:+.2f}, arousal {fa:+.2f}); "
        f"the EEG decoder read {emotion_label(dv, da)} (valence {dv:+.2f}, arousal {da:+.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
