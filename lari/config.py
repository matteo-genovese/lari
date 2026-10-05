"""Validated process-wide configuration; environment access lives here only."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .wake.config import WakeConfig, build_wake_config, default_phrase

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings, with credentials excluded from repr."""
    hermes_root: Path
    token: str = field(repr=False)
    port: int
    hermes_api: str
    hermes_key: str = field(repr=False)
    session_key: str
    hermes_provider: str
    hermes_model: str
    voice_system: str
    tts_voice: str
    stt_model: str
    stt_backend: str
    vosk_model_dir: Path
    stt_lang: str
    stt_gate: str
    wake_provider: str
    wake_confirm: bool
    wake_sensitivity: float
    confirm_frames: int
    ambient_pause_s: float
    echo_mute_s: float
    followup_s: float
    playback_ack_timeout_s: float
    stream_tts_queue_max: int
    stream_text_max_chars: int
    stream_sentence_max_chars: int
    vad_min_rms: float
    vad_noise_mult: float
    silence_end_s: float
    max_utterance_s: float
    min_speech_s: float
    idle_abort_s: float
    agent_timeout_s: float
    cli_timeout_s: float
    elevenlabs_api_key: str = field(repr=False)
    groq_api_key: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    groq_model: str
    realtime_usage_file: Path
    realtime_daily_seconds: float
    usage_ledger: Path
    usage_eur_per_min: float | None
    wake_phrase: str
    wake_aliases: str
    stt_keyterms: str
    wake_re: str
    wake_config: WakeConfig
    hf_hub_offline: str


def _number(env, name, default, kind, *, minimum=0, maximum=None, positive=False):
    message = f"{name}: expected a finite {kind.__name__} "
    message += f"{'>' if positive else '>='} {minimum}"
    if maximum is not None:
        message += f" and <= {maximum}"
    try:
        value = kind(env.get(name, default))
    except (ValueError, TypeError):
        raise ValueError(message) from None
    if ((kind is float and not math.isfinite(value)) or value < minimum
            or (positive and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(message)
    return value


def _bool(env, name, default):
    value = env.get(name, default).strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name}: expected 0 or 1")
    return value != "0"


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Read and validate a mapping without mutating it or the environment."""
    env = os.environ if environ is None else environ
    phrase = env.get("LARI_WAKE_PHRASE", "").strip() or default_phrase(
        env.get("LARI_STT_LANG", "")
    )
    aliases = env.get("LARI_WAKE_ALIASES", "")
    vocab = env.get("LARI_STT_KEYTERMS", "")
    wake_re = env.get("LARI_WAKE_RE", "").strip()
    rate = env.get("LARI_USAGE_EUR_PER_MIN", "").strip()
    return Settings(
        hermes_root=Path(env.get("LARI_HERMES_ROOT") or (Path.home() / ".hermes" / "hermes-agent")).expanduser(),
        token=env.get("LARI_TOKEN", "").strip(),
        port=_number(env, 'LARI_PORT', '8643', int, minimum=1, maximum=65535),
        hermes_api=env.get("LARI_HERMES_API", "http://127.0.0.1:8642"),
        hermes_key=env.get("LARI_HERMES_KEY", ""),
        session_key=env.get("LARI_SESSION_KEY", "lari"),
        hermes_provider=env.get("LARI_HERMES_PROVIDER", "deepseek"),
        hermes_model=env.get("LARI_HERMES_MODEL", "deepseek-flash"),
        voice_system=env.get(
            "LARI_VOICE_SYSTEM",
            "Sei l'assistente vocale di un assistente personale. Le battute ti arrivano da un "
            "microfono, quindi possono essere trascritte in modo imperfetto: se il senso è "
            "intuibile rispondi comunque con la tua interpretazione migliore («ehi ora sono» va "
            "letto come «che ora sono»), chiedendo di ripetere SOLO se è davvero incomprensibile "
            "e in tal caso in una sola riga e senza strumenti. Rispondi SEMPRE in italiano, con "
            "frasi brevi e naturali (massimo 30 secondi di lettura), pensate per essere ascoltate "
            "a voce alta. Non fare liste, non usare markdown, non citare file o percorsi. "
            "Usa gli strumenti solo se la richiesta lo richiede davvero.",
        ),
        tts_voice=env.get("LARI_TTS_VOICE", "it-IT-ElsaNeural"),
        stt_model=env.get("LARI_STT_MODEL", "base"),
        stt_backend=env.get("LARI_STT_BACKEND", "whisper").strip(),
        vosk_model_dir=Path(env.get("LARI_VOSK_MODEL_DIR", str(BASE_DIR / "models/vosk-model-small-it-0.22"))),
        stt_lang=env.get("LARI_STT_LANG", "it").strip(),
        stt_gate=env.get("LARI_STT_GATE", "").strip(),
        wake_provider=env.get("LARI_WAKE_PROVIDER", "whisper"),
        wake_confirm=_bool(env, 'LARI_WAKE_CONFIRM', "1"),
        wake_sensitivity=_number(env, 'LARI_SENSITIVITY', '0.5', float, maximum=1),
        confirm_frames=_number(env, 'LARI_CONFIRM_FRAMES', '3', int, minimum=1),
        ambient_pause_s=_number(env, 'LARI_AMBIENT_PAUSE', '6', float),
        echo_mute_s=_number(env, 'LARI_ECHO_MUTE', '2.5', float),
        followup_s=_number(env, 'LARI_FOLLOWUP_S', '30', float),
        playback_ack_timeout_s=_number(env, 'LARI_PLAYBACK_ACK_TIMEOUT', '45', float, positive=True),
        stream_tts_queue_max=_number(env, 'LARI_STREAM_TTS_QUEUE_MAX', '8', int, minimum=1),
        stream_text_max_chars=_number(env, 'LARI_STREAM_TEXT_MAX_CHARS', '4000', int, minimum=1),
        stream_sentence_max_chars=_number(env, 'LARI_STREAM_SENTENCE_MAX_CHARS', '280', int, minimum=1),
        vad_min_rms=_number(env, 'LARI_VAD_MIN_RMS', '900', float),
        vad_noise_mult=_number(env, 'LARI_VAD_NOISE_MULT', '2.5', float),
        silence_end_s=_number(env, 'LARI_SILENCE_END', '2.5', float, positive=True),
        max_utterance_s=_number(env, 'LARI_MAX_UTTERANCE_S', '45', float, positive=True),
        min_speech_s=_number(env, 'LARI_MIN_SPEECH_S', '0.7', float),
        idle_abort_s=_number(env, 'LARI_IDLE_ABORT_S', '4.0', float, positive=True),
        agent_timeout_s=_number(env, 'LARI_AGENT_TIMEOUT', '180', float, positive=True),
        cli_timeout_s=_number(env, 'LARI_CLI_TIMEOUT', '70', float, positive=True),
        elevenlabs_api_key=env.get("ELEVENLABS_API_KEY", "").strip(),
        groq_api_key=env.get("GROQ_API_KEY", "").strip(),
        openai_api_key=env.get("OPENAI_API_KEY", "").strip(),
        groq_model=env.get("LARI_GROQ_MODEL", "whisper-large-v3"),
        realtime_usage_file=Path(env.get("LARI_REALTIME_USAGE_FILE", BASE_DIR / ".realtime_stt_usage.json")),
        realtime_daily_seconds=_number(env, "LARI_REALTIME_DAILY_SECONDS", "600.0", float),
        usage_ledger=Path(env.get("LARI_USAGE_LEDGER", BASE_DIR / "usage.json")),
        usage_eur_per_min=_number(env, "LARI_USAGE_EUR_PER_MIN", "", float) if rate else None,
        wake_phrase=phrase,
        wake_aliases=aliases,
        stt_keyterms=vocab,
        wake_re=wake_re,
        wake_config=build_wake_config(
            phrase, aliases=aliases.split(","), vocab=vocab.split(","),
            command_override=wake_re or None,
        ),
        hf_hub_offline=env.get("HF_HUB_OFFLINE", "1"),
    )


@cache
def get_settings() -> Settings:
    """Return the process-wide snapshot and apply the existing offline default."""
    settings = load_settings()
    os.environ.setdefault("HF_HUB_OFFLINE", settings.hf_hub_offline)
    return settings
