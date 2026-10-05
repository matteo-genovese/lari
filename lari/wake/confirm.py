"""Second local gate: doubt passes, only confident agreement vetoes."""
from __future__ import annotations
import logging
import numpy as np
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
import time
from . import config as wake_config
from ..stt.local import transcribe
from ..stt.vosk import transcribe_vosk

def confirm_candidate(pcm: np.ndarray, cfg: wake_config.WakeConfig, *, model: str, language: str) -> bool:
    """Second local gate on the 2.5 s candidate only, before Realtime opens.

    Returns True (pass) unless both local recognizers confidently transcribe
    clear non-wake speech.  A false negative costs more than the credits it
    saves, so every doubt passes.  The slow model runs only when the fast
    Vosk transcription is already clean, keeping true positives cheap.
    """
    prefix = pcm[:int(2.5 * SAMPLE_RATE)]
    t0 = time.monotonic()
    try:
        free = transcribe_vosk(prefix, cfg)
    except Exception:
        log.exception("second gate: Vosk libero non disponibile")
        free = ""
    if not cfg.confidently_clean(free):
        log.info("second gate %.2fs: dubbio su %r -> pass",
                 time.monotonic() - t0, free[:60])
        return True
    try:
        heard = transcribe(prefix, cfg, model, language)
    except Exception:
        log.exception("second gate: faster-whisper non disponibile")
        return True
    passed = not cfg.confidently_clean(heard)
    log.info("second gate %.2fs: free=%r whisper=%r -> %s",
             time.monotonic() - t0, free[:60], heard[:60],
             "pass" if passed else "veto")
    return passed

