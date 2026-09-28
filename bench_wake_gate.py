#!/usr/bin/env python
"""A/B benchmark for the second local wake gate on real recordings.

Runs the proven Vosk candidate gate and then the second-gate decision over
each WAV, printing the three decision numbers: false negatives on real wake
clips, added latency on candidates, and candidate seconds a veto would keep
away from the paid provider.  Private recordings stay outside the repository.

Usage:
    BUDDY_WAKE_PHRASE='ehi lari' .venv/bin/python bench_wake_gate.py <wav...>
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

import server
import wake_config


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1, path
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def main(argv: list[str]) -> int:
    cfg = wake_config.from_env()
    server.WAKE_CONFIG = cfg
    print(f"phrase={cfg.phrase!r} model={server.STT_MODEL!r}")
    print(f"{'clip':<42} {'dur':>5} {'gateA':>5} {'free/whisper clean':>20} "
          f"{'verdict':>7} {'t_add':>6}")
    saved, candidates, added = 0.0, 0, []
    for arg in argv:
        path = Path(arg)
        pcm = load(path)
        duration = len(pcm) / 16000
        prefix = pcm[: int(2.5 * server.SAMPLE_RATE)]
        gate = server.vosk_wake(pcm, cfg)
        if not gate:
            print(f"{path.name:<42} {duration:>4.1f}s {'-':>5} {'-':>20} "
                  f"{'A-reject':>7} {'-':>6}")
            continue
        candidates += 1
        t0 = time.monotonic()
        free = server.transcribe_vosk(prefix)
        free_clean = cfg.confidently_clean(free)
        heard = ""
        if free_clean:
            heard = server.transcribe(prefix)
        elapsed = time.monotonic() - t0
        added.append(elapsed)
        passed = not (free_clean and cfg.confidently_clean(heard))
        if not passed:
            saved += duration
        detail = f"{int(free_clean)}/{int(bool(heard) and cfg.confidently_clean(heard))}"
        print(f"{path.name:<42} {duration:>4.1f}s {'pass':>5} {detail:>20} "
              f"{'PASS' if passed else 'VETO':>7} {elapsed:>5.1f}s")
        print(f"    free={free[:60]!r}")
        if heard:
            print(f"    whisper={heard[:60]!r}")
    if added:
        print(f"\nlatency added per candidate: mean={sum(added)/len(added):.1f}s "
              f"max={max(added):.1f}s (n={len(added)})")
    print(f"candidates={candidates} vetoed_seconds={saved:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
