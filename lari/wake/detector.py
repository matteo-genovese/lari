"""Local wake detection and command resolution."""
from __future__ import annotations
import logging
import numpy as np
from ..config import get_settings
_SETTINGS = get_settings()
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
WAKE_CONFIG = _SETTINGS.wake_config
import json
import sys
import threading
from . import config as wake_config
sys.path.insert(0, str(_SETTINGS.hermes_root))
VOSK_MODEL_DIR = _SETTINGS.vosk_model_dir
WAKE_PROVIDER = _SETTINGS.wake_provider
WAKE_PHRASE = WAKE_CONFIG.phrase
WAKE_SENSITIVITY = _SETTINGS.wake_sensitivity
CONFIRM_FRAMES = _SETTINGS.confirm_frames
_vosk_model = None
_vosk_lock = threading.Lock()

def _wake_engine_builder():
    """Lazy loader for the optional openwakeword engine from the Hermes source.

    The default wake provider (energy VAD + Vosk) never needs it; importing it
    lazily keeps the bridge importable without a Hermes checkout (CI, fresh
    installs).
    """
    try:
        from tools.wake_word import _build_engine  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the openwakeword wake engine needs the Hermes source in "
            "LARI_HERMES_ROOT"
        ) from exc
    return _build_engine


def wake_command(text: str, cfg: wake_config.WakeConfig | None = None) -> str | None:
    """Return the request after the wake; None when not addressed to us."""
    return (cfg or WAKE_CONFIG).command(text)


def resolve_command(text: str, conversation_until: float, now: float,
                    cfg: wake_config.WakeConfig | None = None) -> str | None:
    """Wake on the first turn, then free dialog only inside the follow-up window."""
    command = wake_command(text, cfg)
    if command is not None:
        return command
    if now < conversation_until and text.strip():
        return text.strip()
    return None


def get_vosk():
    """Load the small Italian model once; one recognizer per call."""
    global _vosk_model
    with _vosk_lock:
        if _vosk_model is None:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            _vosk_model = Model(str(VOSK_MODEL_DIR))
        return _vosk_model


def vosk_wake(pcm: np.ndarray, cfg: wake_config.WakeConfig | None = None) -> bool:
    """Detect the wake in the first 2.5 s without forcing the rest through the grammar."""
    from vosk import KaldiRecognizer
    cfg = cfg or WAKE_CONFIG
    rec = KaldiRecognizer(get_vosk(), SAMPLE_RATE, json.dumps(list(cfg.grammar)))
    rec.AcceptWaveform(pcm[:int(2.5 * SAMPLE_RATE)].astype(np.int16, copy=False).tobytes())
    heard = json.loads(rec.FinalResult()).get("text", "")
    log.info("Vosk wake: %r", heard)
    return bool(cfg.loose_re.search(heard))


def resolve_vosk_command(text: str, wake: bool, followup: bool,
                         cfg: wake_config.WakeConfig | None = None) -> str | None:
    if not wake and not followup:
        return None
    cfg = cfg or WAKE_CONFIG
    if wake:
        command = wake_command(text, cfg)
        if command is not None:
            return command
        # Free-form ASR sometimes renders the wake as one spurious word; the
        # separate grammar already confirmed it, so strip up to two junk words.
        text = cfg.strip_junk(text)
    return text.strip()


def make_engine():
    cfg = {
        "provider": WAKE_PROVIDER,
        "phrase": WAKE_PHRASE,
        "sensitivity": WAKE_SENSITIVITY,
        "confirmation_frames": CONFIRM_FRAMES,
        "profile_routing": False,   # only the bridge's phrase: no Hermes profile routing
        "openwakeword": {"model": "hey_hermes"},
    }
    return _wake_engine_builder()(cfg)


def __getattr__(name):
    if name == "transcribe_vosk":
        from ..stt.vosk import transcribe_vosk
        return transcribe_vosk
    raise AttributeError(name)
