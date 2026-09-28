"""Compare configured cloud STT providers on caller-supplied WAV files.

Provider API keys must already be present in the process environment. The
optional --local mode also benchmarks the installed Vosk and faster-whisper
models. Audio files are never copied by this script.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lari import stt_backends  # noqa: E402



def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16000 or audio.getnchannels() != 1:
            raise ValueError(f"{path} must be mono PCM WAV at 16 kHz")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="also run installed local STT backends")
    parser.add_argument("wav", nargs="+", type=Path, help="one or more mono 16 kHz WAV files")
    args = parser.parse_args()

    jobs = []
    for backend in stt_backends.BACKENDS:
        key_env = stt_backends.KEY_ENV[backend]
        if os.environ.get(key_env, "").strip():
            jobs.append((backend, lambda pcm, selected=backend: stt_backends.transcribe(pcm, selected)))
    if args.local:
        from lari import server
        jobs.extend((("vosk (local)", server.transcribe_vosk),
                     ("faster-whisper (local)", server.transcribe)))
    if not jobs:
        parser.error("no provider keys are set; use --local or configure a provider key")

    for clip in args.wav:
        pcm = load_audio(clip)
        print(f"\n=== {clip.name} ({len(pcm) / 16000:.2f}s)")
        for name, transcribe in jobs:
            started = time.monotonic()
            try:
                text = transcribe(pcm)
            except Exception as exc:
                text = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"  {name:24} {time.monotonic() - started:6.2f}s  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
