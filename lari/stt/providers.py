"""Cloud STT backends for desk-buddy: A) ElevenLabs Scribe v2, B) Groq Whisper
Large V3 Turbo, C) OpenAI Whisper API (the cloud path of OpenWhispr).

Privacy: these send the utterance audio to the provider. Keys come from the
environment (never logged): ELEVENLABS_API_KEY, GROQ_API_KEY, OPENAI_API_KEY.

Interface: transcribe(pcm_int16_16k_mono, backend) -> text.
"""
from __future__ import annotations

import io
import logging
import wave

import numpy as np

from ..config import Settings, get_settings

log = logging.getLogger(__name__)

BACKENDS = ("elevenlabs", "groq", "openai")
KEY_ENV = {
    "elevenlabs": "ELEVENLABS_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}
# Wake-derived ASR bias terms: keyterms, style prompt and the realtime URL all
# derive from the configured wake phrase plus the optional LARI_STT_KEYTERMS
# vocabulary (names and places the ASR misrenders).
_WAKE = get_settings().wake_config
# ElevenLabs batch keyterms (<= 5 words each, <= 1000 total). Scribe v2 with
# keyterms costs +20% over the base rate.
KEYTERMS = list(_WAKE.batch_keyterms)
# Groq/OpenAI accept only a style prompt (max 224 tokens for Groq).
STYLE_PROMPT = _WAKE.style_prompt


def _to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _post_multipart(url: str, headers: dict, data_tuples: list,
                    file_bytes: bytes, filename: str, mime: str,
                    timeout: float = 30.0) -> dict:
    import httpx
    # httpx requires data as a Mapping (a list of tuples is treated as raw
    # content and the multipart encoding blows up). Repeated fields
    # (keyterms) become list values that _iter_fields expands into same-named fields.
    form: dict = {}
    for key, value in data_tuples:
        if key in form:
            existing = form[key]
            form[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            form[key] = value
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            url, headers=headers, data=form,
            files={"file": (filename, file_bytes, mime)},
        )
        response.raise_for_status()
        return response.json()


def transcribe(pcm: np.ndarray, backend: str, *, settings: Settings | None = None) -> str:
    pcm = np.asarray(pcm, dtype=np.int16)
    if backend not in BACKENDS:
        raise ValueError(f"backend STT cloud sconosciuto: {backend}")
    key_env = KEY_ENV[backend]
    settings = settings or get_settings()
    key = getattr(settings, key_env.lower())
    if not key:
        raise RuntimeError(f"{key_env} non impostata: aggiungila in desk-buddy/.env")

    if backend == "elevenlabs":
        # Bare 16 kHz mono s16le PCM: file_format=pcm_s16le_16 has lower latency.
        call = dict(
            url="https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": key},
            data_tuples=(
                [("model_id", "scribe_v2"),
                 ("language_code", "it"),
                 ("file_format", "pcm_s16le_16")]
                + [("keyterms", term) for term in KEYTERMS]
            ),
            file_bytes=pcm.tobytes(),
            filename="audio.pcm",
            mime="application/octet-stream",
        )
    elif backend == "groq":
        call = dict(
            url="https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data_tuples=[("model", settings.groq_model),
                         ("language", "it"),
                         ("prompt", STYLE_PROMPT),
                         ("response_format", "json")],
            file_bytes=_to_wav(pcm),
            filename="audio.wav",
            mime="audio/wav",
        )
    else:  # openai
        call = dict(
            url="https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data_tuples=[("model", "whisper-1"),
                         ("language", "it"),
                         ("prompt", STYLE_PROMPT)],
            file_bytes=_to_wav(pcm),
            filename="audio.wav",
            mime="audio/wav",
        )

    try:
        payload = _post_multipart(**call)
    except Exception as exc:  # network, quota, format: the caller decides the fallback
        raise RuntimeError(f"STT {backend} fallito: {exc}") from exc
    text = (payload.get("text") or "").strip()
    if not text:
        raise RuntimeError(f"STT {backend} ha restituito testo vuoto")
    return text
