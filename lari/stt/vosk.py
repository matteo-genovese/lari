"""Unconstrained local Vosk transcription; wake grammar lives in wake."""
from __future__ import annotations
import logging
import numpy as np
from ..config import get_settings
_SETTINGS = get_settings()
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
WAKE_CONFIG = _SETTINGS.wake_config
import json
from ..wake.detector import get_vosk

def transcribe_vosk(pcm: np.ndarray) -> str:
    """Local CPU Italian transcription, fed in utterance-sized blocks."""
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(get_vosk(), SAMPLE_RATE)
    words = []
    raw = pcm.astype(np.int16, copy=False).tobytes()
    for i in range(0, len(raw), 8000):
        if rec.AcceptWaveform(raw[i:i + 8000]):
            words.append(json.loads(rec.Result()).get("text", ""))
    words.append(json.loads(rec.FinalResult()).get("text", ""))
    text = " ".join(filter(None, words)).strip()
    log.info("Vosk STT -> %r", text)
    return text

