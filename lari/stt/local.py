"""Lazy faster-whisper transcription."""
from __future__ import annotations
import logging
import numpy as np
from ..config import get_settings
_SETTINGS = get_settings()
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
WAKE_CONFIG = _SETTINGS.wake_config
import re
import threading
STT_MODEL = _SETTINGS.stt_model
STT_LANG = _SETTINGS.stt_lang
_stt = {}
_stt_lock = threading.Lock()

def get_stt(model: str | None = None):
    """Lazy singleton for faster-whisper (the base model weighs ~150 MB; load it once)."""
    with _stt_lock:
        name = model or STT_MODEL
        if name not in _stt:
            from faster_whisper import WhisperModel
            log.info("caricamento STT model=%s", name)
            _stt[name] = WhisperModel(name, device="cpu", compute_type="int8")
        return _stt[name]


def transcribe(pcm: np.ndarray, model: str | None = None) -> str:
    """pcm: int16 16 kHz mono -> text."""
    audio = pcm.astype(np.float32) / 32768.0
    # FIXED language: auto-detection on short noisy clips goes haywire
    # (it once returned Japanese). Empty ("") = auto.
    lang = STT_LANG or None
    segments, info = get_stt(model).transcribe(
        audio,
        language=lang,
        beam_size=1,                        # beam5 measured slower on low-power CPUs
        vad_filter=True,
        vad_parameters={"threshold": 0.6,   # trims noise fragments harder
                        "min_silence_duration_ms": 400},
        hallucination_silence_threshold=2.0,  # trims hallucinated sentences at the tail
        condition_on_previous_text=False,   # otherwise the model feeds itself repetitions
        # SHORT prompt: a long prompt gets repeated by the ASR instead of the audio
        # (hallucination), especially on noisy recordings
        initial_prompt=WAKE_CONFIG.prompt,
    )
    text = " ".join(s.text for s in segments).strip()
    log.info("STT lang=%s prob=%.2f -> %r", info.language, info.language_probability, text)
    # Anti-hallucination guard: on near-silence or noise Whisper invents short,
    # recurring sentences ("I'm sorry.", words in other languages...). If there
    # is not even one real word, the text is discarded instead of sent to the agent.
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ']{3,}", text)]
    black = {"im sorry", "i'm sorry", "sorry", "thanks for watching", "subscrib", "musica"}
    low = text.lower().strip(" .!?")
    if len(words) < 1 or low in black:
        log.info("STT scartato come fantasma: %r", text[:60])
        return ""
    return text

