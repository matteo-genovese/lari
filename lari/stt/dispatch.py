"""Provider selection, decoding and strictly local realtime fallback."""
from __future__ import annotations
import logging
import numpy as np
from ..config import Settings
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
import asyncio
import time
from . import providers as stt_backends
from . import realtime as realtime_backend
from .local import transcribe
from .vosk import transcribe_vosk
from ..wake.detector import vosk_wake, resolve_command, resolve_vosk_command

def _transcribe_local_fallback(pcm: np.ndarray, settings: Settings) -> str:
    """Transcribe realtime failures without making a paid provider request."""
    try:
        return transcribe_vosk(pcm, settings.wake_config)
    except Exception:
        log.exception("Vosk locale non disponibile; uso faster-whisper locale")
        return transcribe(pcm, settings.wake_config, settings.stt_model, settings.stt_lang)


def stt_transcribe(pcm: np.ndarray, settings: Settings) -> str:
    """Raw text from the configured STT backend: cloud (groq/elevenlabs/openai)
    or local Whisper. A single dispatch point: _on_wake no longer has to pick
    by hand (it used to send everything but vosk to local Whisper)."""
    if settings.stt_backend == realtime_backend.REALTIME_BACKEND:
        # Realtime callers must never silently turn a provider failure into a
        # paid batch request.  The live path is opened only after local wake
        # confirmation; direct callers retain the same local fallback.
        return _transcribe_local_fallback(pcm, settings)
    if settings.stt_backend in stt_backends.BACKENDS:
        return stt_backends.transcribe(pcm, settings.stt_backend, settings=settings)
    return transcribe(pcm, settings.wake_config, settings.stt_model, settings.stt_lang)


async def _transcribe_realtime_or_batch(pcm: np.ndarray, realtime, turn: int,
                                        start_failed: bool = False, *, settings: Settings) -> tuple[str, bool]:
    """Return ``(transcript, used_batch)`` without a paid fallback.

    The second tuple value is retained for compatibility with older callers;
    it is always false now.  Realtime failure/cap/provider errors fall back to
    the local Vosk recognizer (then local faster-whisper), never Scribe batch.
    """
    try:
        if start_failed or realtime is None:
            raise realtime_backend.RealtimeUnavailable("realtime unavailable")
        return await realtime.finish(), False
    except realtime_backend.RealtimeUnavailable:
        log.warning(
            "turno %d: ElevenLabs realtime STT non disponibile; fallback STT locale",
            turn,
        )
        return await asyncio.to_thread(_transcribe_local_fallback, pcm, settings), False


async def transcribe_realtime_or_batch(pcm: np.ndarray, realtime, turn: int,
                                       start_failed: bool = False, *, settings: Settings) -> str:
    """Return one transcript: committed realtime text or local fallback."""
    text, _used_batch = await _transcribe_realtime_or_batch(
        pcm, realtime, turn, start_failed=start_failed, settings=settings
    )
    return text


def decode_utterance(pcm: np.ndarray, followup: bool, backend: str | None = None, *, settings: Settings) -> str | None:
    """Transcribe and filter the wake; None = speech not addressed to the bridge."""
    backend = backend or settings.stt_backend
    if backend == "vosk":
        # The grammar only looks at the start: it does not distort free transcription.
        wake = False if followup else vosk_wake(pcm, settings.wake_config)
        if not wake and not followup:
            return None
        text = transcribe_vosk(pcm, settings.wake_config)
        return resolve_vosk_command(text, wake, followup, settings.wake_config) if text else None
    if backend in stt_backends.BACKENDS:
        # Cloud (A/B/C): same rules as whisper — the wake is regexed against the
        # text, the follow-up window is evaluated at the start of the turn.
        text = stt_backends.transcribe(pcm, backend, settings=settings)
        return resolve_command(text, float("inf") if followup else 0.0,
                               time.monotonic(), settings.wake_config) if text else None
    if backend == realtime_backend.REALTIME_BACKEND:
        # Realtime is a streaming transport, not a reason to use ElevenLabs
        # batch when called synchronously.
        text = _transcribe_local_fallback(pcm, settings)
        return resolve_vosk_command(text, wake=not followup, followup=followup, cfg=settings.wake_config) if text else None
    if backend == "whisper":
        text = transcribe(pcm, settings.wake_config, settings.stt_model, settings.stt_lang)
        return resolve_command(text, float("inf") if followup else 0.0, time.monotonic(), settings.wake_config) if text else None
    raise ValueError(f"STT backend non supportato: {backend}")



async def open_stream(on_partial, *, settings: Settings):
    """Open the configured live transport after the local wake gates."""
    if settings.stt_backend != realtime_backend.REALTIME_BACKEND:
        return None, False
    try:
        return await realtime_backend.RealtimeScribe.connect(on_partial=on_partial, settings=settings), False
    except Exception:
        return None, True


async def transcribe_turn(pcm, stream, turn, followup, start_failed=False, *, settings: Settings):
    if settings.stt_backend == realtime_backend.REALTIME_BACKEND:
        text, _ = await _transcribe_realtime_or_batch(
            pcm, stream, turn, start_failed=start_failed, settings=settings)
    elif settings.stt_backend == "vosk":
        text = await asyncio.to_thread(decode_utterance, pcm, followup, settings=settings)
    else:
        text = await asyncio.to_thread(stt_transcribe, pcm, settings)
    if text is None or (settings.stt_backend != "vosk" and not text):
        return None
    return text


def command_for_turn(text, followup, local_wake_confirmed, settings: Settings):
    if settings.stt_backend == "vosk":
        return text
    if settings.stt_backend == realtime_backend.REALTIME_BACKEND:
        if followup:
            return text.strip()
        if local_wake_confirmed:
            return resolve_vosk_command(text, wake=True, followup=False, cfg=settings.wake_config)
        return resolve_command(text, 0.0, time.monotonic(), settings.wake_config)
    return resolve_command(text, float("inf") if followup else 0.0, time.monotonic(), settings.wake_config)
