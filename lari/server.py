"""Lari bridge: satellite microphone -> local wake gate -> STT -> Hermes -> TTS.

The browser satellite captures mono 16 kHz PCM and streams it to the bridge
via a token-protected WebSocket. Keep the installed wake phrase and backend
settings independent of the device and project branding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from . import stt_backends
from . import wake_config

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root; runtime paths never depend on cwd
HERMES_ROOT = Path(os.environ.get("BUDDY_HERMES_ROOT") or (Path.home() / ".hermes" / "hermes-agent")).expanduser()
sys.path.insert(0, str(HERMES_ROOT))


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
            "BUDDY_HERMES_ROOT"
        ) from exc
    return _build_engine

# ─── configuration ──────────────────────────────────────────────────────────
TOKEN = os.environ.get("BUDDY_TOKEN", "").strip()
PORT = int(os.environ.get("BUDDY_PORT", "8643"))
HERMES_API = os.environ.get("BUDDY_HERMES_API", "http://127.0.0.1:8642")
HERMES_KEY = os.environ.get("BUDDY_HERMES_KEY", "")
SESSION_KEY = os.environ.get("BUDDY_SESSION_KEY", "desk-buddy")
HERMES_PROVIDER = os.environ.get("BUDDY_HERMES_PROVIDER", "deepseek")
HERMES_MODEL = os.environ.get("BUDDY_HERMES_MODEL", "deepseek-flash")
# ─── voice agent ──────────────────────────────────────────────────────
# "deepseek" = DeepSeek flash via direct API (fast, NO tools: weather and
# news are not real-time). "hermes" = the full Hermes agent (real tools,
# much slower: 18k tokens of system prompt per utterance).
AGENT_BACKEND = os.environ.get("BUDDY_AGENT_BACKEND", "deepseek").strip()

from . import usage  # noqa: E402  (project module)
USAGE_LEDGER = usage.UsageLedger()

def _deepseek_from_hermes_env() -> tuple[str, str]:
    """DeepSeek key/base URL also from ~/.hermes/.env.

    The systemd unit loads only the bridge's .env; the key lives in the
    Hermes config.
    """
    key = base = ""
    try:
        for line in (Path.home() / ".hermes" / ".env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip("\"'")
            elif name.strip() == "DEEPSEEK_BASE_URL":
                base = value.strip().strip("\"'")
    except OSError:
        pass
    return key, base

_ds_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
_ds_base = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
if not _ds_key or not _ds_base:
    _hk, _hb = _deepseek_from_hermes_env()
    _ds_key = _ds_key or _hk
    _ds_base = _ds_base or _hb
DEEPSEEK_KEY = _ds_key
DEEPSEEK_API = _ds_base.rstrip("/") or "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("BUDDY_DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_TIMEOUT_S = float(os.environ.get("BUDDY_DEEPSEEK_TIMEOUT", "60"))
# Voice = short replies meant to be spoken. Without this, a bogus
# transcription makes the agent dig through logs (125 s in the worst measured case).
VOICE_SYSTEM = os.environ.get(
    "BUDDY_VOICE_SYSTEM",
    "Sei l'assistente vocale di un assistente personale. Le battute ti arrivano da un "
    "microfono, quindi possono essere trascritte in modo imperfetto: se il senso è "
    "intuibile rispondi comunque con la tua interpretazione migliore («ehi ora sono» va "
    "letto come «che ora sono»), chiedendo di ripetere SOLO se è davvero incomprensibile "
    "e in tal caso in una sola riga e senza strumenti. Rispondi SEMPRE in italiano, con "
    "frasi brevi e naturali (massimo 30 secondi di lettura), pensate per essere ascoltate "
    "a voce alta. Non fare liste, non usare markdown, non citare file o percorsi. "
    "Usa gli strumenti solo se la richiesta lo richiede davvero.",
)


class HermesContinuationError(RuntimeError):
    """A continued Hermes turn failed without switching transcripts."""


class HermesStreamTurnError(RuntimeError):
    """A streamed turn failed; the turn must never be submitted again."""

    def __init__(self, message: str, had_audio: bool = False):
        super().__init__(message)
        self.had_audio = had_audio


class HermesReply(str):
    """String-compatible Hermes reply carrying the response session id."""

    def __new__(
        cls,
        text: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        reply = super().__new__(cls, text)
        reply.session_id = session_id
        reply.run_id = run_id
        return reply


TTS_VOICE = os.environ.get("BUDDY_TTS_VOICE", "it-IT-ElsaNeural")
STT_MODEL = os.environ.get("BUDDY_STT_MODEL", "base")
STT_BACKEND = os.environ.get("BUDDY_STT_BACKEND", "whisper").strip()
VOSK_MODEL_DIR = Path(os.environ.get("BUDDY_VOSK_MODEL_DIR", str(BASE_DIR / "models/vosk-model-small-it-0.22")))
STT_LANG = os.environ.get("BUDDY_STT_LANG", "it").strip()  # "" = auto-detect
# Fast gate for the wake: on bad audio the large model takes 50-60 s and
# blocks everything. With a small model the rejection arrives in seconds.
# Empty ("") = a single pass with the large model.
# A tiny model used as veto also rejects genuine wake words.
# A single pass with small: fewer false negatives and no double transcription.
STT_GATE = os.environ.get("BUDDY_STT_GATE", "").strip()

SAMPLE_RATE = 16000
FRAME = 1280              # 80 ms @ 16 kHz, frame size recommended by openWakeWord
WAKE_PROVIDER = os.environ.get("BUDDY_WAKE_PROVIDER", "whisper")
# whisper: any speech starts the recording, then the phrase is searched in the text.
# sherpa/openwakeword: dedicated hotword engine (English; misses the IT pronunciation).
# The wake phrase is one setting: command regex, Vosk grammar, junk cleanup and
# ASR keyterms all derive from it in wake_config.py.  The regex is
# start-anchored so background mentions ("ho parlato con ...") never trigger.
WAKE_CONFIG = wake_config.from_env()
WAKE_RE = WAKE_CONFIG.command_re

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

WAKE_PHRASE = WAKE_CONFIG.phrase
# Second local gate on the wake candidate, before any provider connection.
# It may veto only a confident mismatch (see confirm_candidate): every doubt
# passes, so a real wake is never lost to an ASR mishearing.  Set
# BUDDY_WAKE_CONFIRM=0 to disable.
WAKE_CONFIRM = os.environ.get("BUDDY_WAKE_CONFIRM", "1").strip() != "0"
# sherpa threshold = 0.05 + 0.4*sens; 0.5 -> 0.25 (upstream-recommended value)
WAKE_SENSITIVITY = float(os.environ.get("BUDDY_SENSITIVITY", "0.5"))
CONFIRM_FRAMES = int(os.environ.get("BUDDY_CONFIRM_FRAMES", "3"))
COOLDOWN_S = 2.0          # same constraint as Hermes between two wakes
AMBIENT_PAUSE_S = float(os.environ.get("BUDDY_AMBIENT_PAUSE", "6"))  # pause after speech not addressed to us
# The phone plays the reply through the same speaker the microphone uses:
# without this mute, the system ends up transcribing itself.
ECHO_MUTE_S = float(os.environ.get("BUDDY_ECHO_MUTE", "2.5"))
FOLLOWUP_S = float(os.environ.get("BUDDY_FOLLOWUP_S", "30"))  # after audio playback
PLAYBACK_ACK_TIMEOUT_S = float(os.environ.get("BUDDY_PLAYBACK_ACK_TIMEOUT", "45"))

# Streaming voice responses are deliberately bounded.  The queue is small so a
# slow browser/TTS provider applies back-pressure to SSE instead of allowing an
# unbounded response to accumulate in memory.
STREAM_TTS_QUEUE_MAX = int(os.environ.get("BUDDY_STREAM_TTS_QUEUE_MAX", "8"))
STREAM_TEXT_MAX_CHARS = int(os.environ.get("BUDDY_STREAM_TEXT_MAX_CHARS", "4000"))
STREAM_SENTENCE_MAX_CHARS = int(os.environ.get("BUDDY_STREAM_SENTENCE_MAX_CHARS", "280"))

# VAD: adaptive threshold. Minimum base threshold + multiple of the noise floor.
VAD_MIN_RMS = float(os.environ.get("BUDDY_VAD_MIN_RMS", "900"))
VAD_NOISE_MULT = float(os.environ.get("BUDDY_VAD_NOISE_MULT", "2.5"))
# More tolerant of pauses while a request is being phrased; the VAD still
# closes on silence, without waiting for the hard limit.
SILENCE_END_S = float(os.environ.get("BUDDY_SILENCE_END", "2.5"))
# Only a parachute against continuous noise / a stuck VAD, not a target utterance length.
MAX_UTTERANCE_S = float(os.environ.get("BUDDY_MAX_UTTERANCE_S", "45"))
MIN_SPEECH_S = float(os.environ.get("BUDDY_MIN_SPEECH_S", "0.7"))
IDLE_ABORT_S = float(os.environ.get("BUDDY_IDLE_ABORT_S", "4.0"))

AGENT_TIMEOUT_S = float(os.environ.get("BUDDY_AGENT_TIMEOUT", "180"))
CLI_TIMEOUT_S = float(os.environ.get("BUDDY_CLI_TIMEOUT", "70"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # no revision check at every STT model load

log = logging.getLogger("desk-buddy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "server.log")],
)

_stt: dict = {}
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


_vosk_model = None
_vosk_lock = threading.Lock()

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

def confirm_candidate(pcm: np.ndarray, cfg: wake_config.WakeConfig | None = None) -> bool:
    """Second local gate on the 2.5 s candidate only, before Realtime opens.

    Returns True (pass) unless both local recognizers confidently transcribe
    clear non-wake speech.  A false negative costs more than the credits it
    saves, so every doubt passes.  The slow model runs only when the fast
    Vosk transcription is already clean, keeping true positives cheap.
    """
    cfg = cfg or WAKE_CONFIG
    prefix = pcm[:int(2.5 * SAMPLE_RATE)]
    t0 = time.monotonic()
    try:
        free = transcribe_vosk(prefix)
    except Exception:
        log.exception("second gate: Vosk libero non disponibile")
        free = ""
    if not cfg.confidently_clean(free):
        log.info("second gate %.2fs: dubbio su %r -> pass",
                 time.monotonic() - t0, free[:60])
        return True
    try:
        heard = transcribe(prefix)
    except Exception:
        log.exception("second gate: faster-whisper non disponibile")
        return True
    passed = not cfg.confidently_clean(heard)
    log.info("second gate %.2fs: free=%r whisper=%r -> %s",
             time.monotonic() - t0, free[:60], heard[:60],
             "pass" if passed else "veto")
    return passed

def _transcribe_local_fallback(pcm: np.ndarray) -> str:
    """Transcribe realtime failures without making a paid provider request."""
    try:
        return transcribe_vosk(pcm)
    except Exception:
        log.exception("Vosk locale non disponibile; uso faster-whisper locale")
        return transcribe(pcm)


def stt_transcribe(pcm: np.ndarray) -> str:
    """Raw text from the configured STT backend: cloud (groq/elevenlabs/openai)
    or local Whisper. A single dispatch point: _on_wake no longer has to pick
    by hand (it used to send everything but vosk to local Whisper)."""
    if STT_BACKEND == stt_backends.REALTIME_BACKEND:
        # Realtime callers must never silently turn a provider failure into a
        # paid batch request.  The live path is opened only after local wake
        # confirmation; direct callers retain the same local fallback.
        return _transcribe_local_fallback(pcm)
    if STT_BACKEND in stt_backends.BACKENDS:
        return stt_backends.transcribe(pcm, STT_BACKEND)
    return transcribe(pcm)


async def _transcribe_realtime_or_batch(pcm: np.ndarray, realtime, turn: int,
                                        start_failed: bool = False) -> tuple[str, bool]:
    """Return ``(transcript, used_batch)`` without a paid fallback.

    The second tuple value is retained for compatibility with older callers;
    it is always false now.  Realtime failure/cap/provider errors fall back to
    the local Vosk recognizer (then local faster-whisper), never Scribe batch.
    """
    try:
        if start_failed or realtime is None:
            raise stt_backends.RealtimeUnavailable("realtime unavailable")
        return await realtime.finish(), False
    except stt_backends.RealtimeUnavailable:
        log.warning(
            "turno %d: ElevenLabs realtime STT non disponibile; fallback STT locale",
            turn,
        )
        return await asyncio.to_thread(_transcribe_local_fallback, pcm), False


async def transcribe_realtime_or_batch(pcm: np.ndarray, realtime, turn: int,
                                       start_failed: bool = False) -> str:
    """Return one transcript: committed realtime text or local fallback."""
    text, _used_batch = await _transcribe_realtime_or_batch(
        pcm, realtime, turn, start_failed=start_failed
    )
    return text


def decode_utterance(pcm: np.ndarray, followup: bool, backend: str | None = None) -> str | None:
    """Transcribe and filter the wake; None = speech not addressed to the bridge."""
    backend = backend or STT_BACKEND
    if backend == "vosk":
        # The grammar only looks at the start: it does not distort free transcription.
        wake = False if followup else vosk_wake(pcm)
        if not wake and not followup:
            return None
        text = transcribe_vosk(pcm)
        return resolve_vosk_command(text, wake, followup) if text else None
    if backend in stt_backends.BACKENDS:
        # Cloud (A/B/C): same rules as whisper — the wake is regexed against the
        # text, the follow-up window is evaluated at the start of the turn.
        text = stt_backends.transcribe(pcm, backend)
        return resolve_command(text, float("inf") if followup else 0.0,
                               time.monotonic()) if text else None
    if backend == stt_backends.REALTIME_BACKEND:
        # Realtime is a streaming transport, not a reason to use ElevenLabs
        # batch when called synchronously.
        text = _transcribe_local_fallback(pcm)
        return resolve_vosk_command(text, wake=not followup, followup=followup) if text else None
    if backend == "whisper":
        text = transcribe(pcm)
        return resolve_command(text, float("inf") if followup else 0.0, time.monotonic()) if text else None
    raise ValueError(f"STT backend non supportato: {backend}")

def save_turn_audio(pcm: np.ndarray, directory: Path | None = None,
                    name: str | None = None, keep: int = 5) -> Path:
    """Keep a few turn WAVs for reproducible diagnostics (local only)."""
    import wave
    directory = directory or BASE_DIR / "calibration" / "turns"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if name is None:
        name = f"turn_{time.time_ns()}.wav"
    path = directory / name
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.astype(np.int16, copy=False).tobytes())
    for old in sorted(directory.glob("turn_*.wav"))[:-keep]:
        old.unlink()
    return path


def tts(text: str) -> bytes:
    """Text -> mp3 via edge-tts."""
    import asyncio as _a
    import edge_tts

    async def _run():
        buf = bytearray()
        comm = edge_tts.Communicate(text, TTS_VOICE)
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    return _a.run(_run())


class SpeakableSentenceBuffer:
    """Collect SSE deltas without ever cutting a word in half."""

    _END_RE = re.compile(r"[.!?]+(?:[\"'’”»\)\]]+)?(?=\s|$)|[;:]+(?=\s|$)|\n+")

    def __init__(self, max_chars: int = STREAM_SENTENCE_MAX_CHARS,
                 total_limit: int = STREAM_TEXT_MAX_CHARS):
        self.max_chars = max(1, max_chars)
        self.total_limit = max(1, total_limit)
        self.buffer = ""
        self.total_chars = 0

    def _take(self, end: int) -> str:
        sentence = self.buffer[:end].strip()
        self.buffer = self.buffer[end:].lstrip()
        return sentence

    def _split_long_prefix(self) -> str | None:
        if len(self.buffer) <= self.max_chars:
            return None
        # Only split at whitespace.  If one token is unusually long, retain it
        # until punctuation/final flush rather than producing partial-word audio.
        cut = self.buffer.rfind(" ", 0, self.max_chars + 1)
        if cut <= 0:
            return None
        return self._take(cut)

    def feed(self, delta: str) -> list[str]:
        if not isinstance(delta, str) or not delta:
            return []
        self.total_chars += len(delta)
        if self.total_chars > self.total_limit:
            raise RuntimeError("risposta Hermes troppo lunga")
        self.buffer += delta
        sentences: list[str] = []
        while self.buffer:
            match = self._END_RE.search(self.buffer)
            if match:
                sentence = self._take(match.end())
                if sentence:
                    sentences.append(sentence)
                continue
            sentence = self._split_long_prefix()
            if sentence:
                sentences.append(sentence)
                continue
            break
        return sentences

    def flush(self) -> list[str]:
        sentences: list[str] = []
        while self.buffer:
            sentence = self._split_long_prefix()
            if sentence:
                sentences.append(sentence)
                continue
            sentence = self.buffer.strip()
            self.buffer = ""
            if sentence:
                sentences.append(sentence)
        return sentences


async def ask_hermes(text: str, session_id: str | None = None) -> HermesReply:
    """Send the utterance to the agent, continuing ``session_id`` when present.

    The return value remains string-compatible for existing callers and carries the
    response's ``X-Hermes-Session-Id`` as ``.session_id``.
    """
    import httpx

    payload = {
        "model": HERMES_MODEL,
        "provider": HERMES_PROVIDER,
        "model_options": {"reasoning": {"enabled": False}},
        "messages": [
            {"role": "system", "content": VOICE_SYSTEM},
            {"role": "user", "content": text},
        ],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if HERMES_KEY:
        headers["Authorization"] = f"Bearer {HERMES_KEY}"
    headers["X-Hermes-Session-Key"] = SESSION_KEY
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=AGENT_TIMEOUT_S) as client:
            r = await client.post(f"{HERMES_API}/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            reply = (data["choices"][0]["message"]["content"] or "").strip()
            return HermesReply(reply, r.headers.get("X-Hermes-Session-Id"))
    except Exception as exc:  # API server down -> CLI fallback
        if session_id:
            # A CLI continuation has different state semantics. Keep the explicit
            # transcript id private to this WebSocket and surface a concise error.
            log.warning("continuità Hermes fallita (%s)", type(exc).__name__)
            raise HermesContinuationError("continuazione Hermes non disponibile") from None
        log.warning("API server non raggiungibile (%s), fallback hermes chat -q", type(exc).__name__)
        loop = asyncio.get_running_loop()
        return HermesReply(await loop.run_in_executor(None, _ask_cli, text))


async def stream_hermes(
    text: str,
    session_id: str | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    on_approval: Callable[[dict], Awaitable[None]] | None = None,
) -> HermesReply:
    """Stream one opt-in Hermes turn over SSE.

    Only ``delta.content`` is speech. Reasoning, tool/status events, and approval
    metadata are kept out of the returned text. The response session/run ids are
    published only after the terminal ``[DONE]`` frame, so callers can retain
    their previous transcript on any failed or incomplete stream.
    """
    import httpx

    payload = {
        "model": HERMES_MODEL,
        "provider": HERMES_PROVIDER,
        "model_options": {"reasoning": {"enabled": False}},
        "messages": [
            {"role": "system", "content": VOICE_SYSTEM},
            {"role": "user", "content": text},
        ],
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Hermes-Session-Key": SESSION_KEY,
    }
    if HERMES_KEY:
        headers["Authorization"] = f"Bearer {HERMES_KEY}"
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id

    content: list[str] = []
    response_session_id: str | None = None
    run_id: str | None = None
    done = False
    saw_stop_finish = False

    def find_run_id(value: object) -> str | None:
        if isinstance(value, dict):
            candidate = value.get("run_id")
            if isinstance(candidate, str) and candidate:
                return candidate
            for nested in value.values():
                found = find_run_id(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = find_run_id(nested)
                if found:
                    return found
        return None

    async def handle_event(event_name: str | None, data: str) -> None:
        nonlocal done, run_id, saw_stop_finish
        if not data:
            return
        if data == "[DONE]":
            done = True
            return

        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("stream Hermes non valido: JSON SSE corrotto") from exc

        event_run_id = find_run_id(event)
        if event_run_id:
            run_id = event_run_id

        if event_name == "approval.request":
            if on_approval is not None:
                await on_approval(event)
            return
        if event_name in {"hermes.tool.progress", "hermes.status"}:
            return

        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            return
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or finish_reason.lower() != "stop":
                if finish_reason == "error":
                    raise RuntimeError("Hermes stream terminato con errore")
                raise RuntimeError(
                    f"Hermes stream terminato con finish_reason={finish_reason!r}"
                )
            saw_stop_finish = True

        delta = choice.get("delta")
        delta_content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(delta_content, str):
            content.append(delta_content)
            if on_delta is not None:
                await on_delta(delta_content)

    try:
        async with httpx.AsyncClient(timeout=AGENT_TIMEOUT_S) as client:
            async with client.stream(
                "POST",
                f"{HERMES_API}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                response_session_id = response.headers.get("X-Hermes-Session-Id")

                event_name: str | None = None
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        if data_lines:
                            await handle_event(event_name, "\n".join(data_lines))
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    field, separator, value = line.partition(":")
                    if separator and value.startswith(" "):
                        value = value[1:]
                    if field == "event":
                        event_name = value
                    elif field == "data":
                        data_lines.append(value)

                # A final event without a blank line is still parsed, but cannot
                # satisfy the terminal [DONE] requirement by itself.
                if data_lines:
                    await handle_event(event_name, "\n".join(data_lines))
    except Exception:
        # Deliberately no retry: Hermes tool calls/approvals may already have had
        # side effects before a transport or stream-level failure was observed.
        raise

    if not done:
        raise RuntimeError("stream Hermes incompleto: manca [DONE]")
    if not saw_stop_finish:
        raise RuntimeError("stream Hermes incompleto: manca finish_reason='stop'")
    return HermesReply("".join(content).strip(), response_session_id, run_id)


def _ask_cli(text: str) -> str:
    """Fallback without the API server. stdin=DEVNULL: an approval prompt must
    fail immediately instead of hanging while waiting for a tty that does not exist."""
    import subprocess as sp

    t0 = time.time()
    try:
        p = sp.run(
            [str(HERMES_ROOT / "venv/bin/hermes"), "chat", "-q", text, "-Q",
             "-m", HERMES_MODEL, "--provider", HERMES_PROVIDER, "--reasoning", "none",
             "--continue", SESSION_KEY, "--create-if-missing"],
            capture_output=True, text=True, timeout=CLI_TIMEOUT_S, cwd=str(HERMES_ROOT),
            stdin=sp.DEVNULL,
        )
        out = (p.stdout or "").strip()
        dt = time.time() - t0
        log.info("fallback CLI: %.1fs exit=%s out=%r", dt, p.returncode, out[:120])
        return out or f"(nessuna risposta, exit={p.returncode}: {(p.stderr or '').strip()[:200]})"
    except sp.TimeoutExpired:
        log.error("fallback CLI: timeout dopo %.0fs", CLI_TIMEOUT_S)
        return "(il backend è lento: fai girare l'API server riavviando il gateway)"
    except Exception as exc:
        return f"(errore agente: {exc})"


async def _post_chat(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    """POST JSON to a chat/completions endpoint (the single HTTP point for the
    mocked test)."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def ask_deepseek(text: str) -> str:
    """DeepSeek flash via direct API: ~1 s, no tools. The model has no
    real-time data (weather, news): it says so instead of making it up."""
    if not DEEPSEEK_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY non impostata")
    system = (VOICE_SYSTEM +
              " Non hai strumenti e non hai dati in tempo reale (meteo, notizie, "
              "corsi): se la richiesta li richiede, dillo in una riga invece di "
              "inventare la risposta.")
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type": "application/json"}
    data = await _post_chat(f"{DEEPSEEK_API}/chat/completions", headers,
                            payload, DEEPSEEK_TIMEOUT_S)
    return (data["choices"][0]["message"]["content"] or "").strip()

async def ask(text: str, session_id: str | None = None) -> str:
    """Voice agent dispatch: DeepSeek flash by default (fast), with automatic
    fallback to the full Hermes agent if it fails or backend=hermes."""
    if AGENT_BACKEND == "deepseek":
        try:
            return await ask_deepseek(text)
        except Exception as exc:
            log.warning("DeepSeek non disponibile (%s), fallback agente Hermes", exc)
    if session_id is None:
        return await ask_hermes(text)
    return await ask_hermes(text, session_id=session_id)

# ─── wake engine ─────────────────────────────────────────────────────────────
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


class Session:
    """State of one satellite connection (one device = one session)."""

    def __init__(self, ws: WebSocket, send_json):
        self.ws = ws
        self.send_json = send_json
        self.engine = None
        self.state = "listening"           # the worker starts listening immediately
        self.frames = bytearray()          # frame-alignment residue
        self.last_wake = 0.0
        self.noise_floor = 500.0           # EMA of the noise floor
        self.last_silent = time.time()     # last truly silent frame
        self.recv_queue: queue.Queue = queue.Queue(maxsize=400)
        self.stop = threading.Event()
        self.turn = 0
        # rotating calibration buffer: the last ~12 s of microphone, to see
        # what the detector really hears when it does not fire
        self.recent = bytearray()
        self.recent_lock = threading.Lock()
        self.last_tts = 0.0
        self.conversation_until = 0.0
        self.awaiting_playback = False
        self.awaiting_playback_turn: int | None = None
        self._playback_waiter: asyncio.Future | None = None
        self.playback_status: str | None = None
        self._interrupted_turn: int | None = None
        self._interrupted_followup_task: asyncio.Task | None = None
        self.active_turn: int | None = None
        self._turn_task: asyncio.Task | None = None
        self.hermes_session_id: str | None = None
        self._partial_turn: int | None = None

    async def _send_partial(self, text: str, turn: int):
        self._partial_turn = turn
        await self.send_json({
            "type": "partial_transcript", "text": text, "turn": turn,
        })

    async def _clear_partial(self, turn: int):
        """Remove the browser's in-progress transcript after a discarded turn."""
        if self._partial_turn is None:
            return
        partial_turn = self._partial_turn
        self._partial_turn = None
        await self.send_json({
            "type": "partial_transcript", "text": "", "turn": partial_turn,
        })

    async def _ask_hermes(self, text: str) -> str:
        """Ask Hermes and update only this WebSocket's transcript id."""
        if self.hermes_session_id is None:
            # Keep the no-argument call compatible with existing test doubles and
            # first-turn callers.
            result = await ask_hermes(text)
        else:
            result = await ask_hermes(text, session_id=self.hermes_session_id)
        returned_id = getattr(result, "session_id", None)
        if returned_id:
            self.hermes_session_id = returned_id
        return str(result)

    def _begin_playback(self, turn: int):
        self.awaiting_playback = True
        self.awaiting_playback_turn = turn
        self._playback_waiter = asyncio.get_running_loop().create_future()
        self.playback_status = "pending"

    @staticmethod
    def _same_turn(left, right) -> bool:
        return left is not None and right is not None and str(left) == str(right)

    def mark_playback_done(self, now: float | None = None, turn: int | None = None,
                           status: str = "completed") -> bool:
        """Accept only the current turn's completion ACK.

        A missing turn remains accepted for the pre-segmented legacy client.  A
        failed segmented playback releases the worker without opening follow-up.
        """
        if not self.awaiting_playback:
            return False
        if turn is not None and not self._same_turn(turn, self.awaiting_playback_turn):
            return False
        if status not in {"completed", "failed"}:
            return False
        self.awaiting_playback = False
        self.awaiting_playback_turn = None
        completed = status == "completed"
        self.playback_status = status
        if completed:
            self.conversation_until = (time.monotonic() if now is None else now) + FOLLOWUP_S
            # The browser adds 700 ms of mute at the end of playback.
            self.last_tts = time.time() - ECHO_MUTE_S + 0.7
        else:
            self.conversation_until = 0.0
        waiter = self._playback_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(status)
        return True

    async def _wait_for_playback(self, turn: int) -> str:
        if not self.awaiting_playback or not self._same_turn(turn, self.awaiting_playback_turn):
            return "none"
        waiter = self._playback_waiter
        if waiter is None:
            return "none"
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), PLAYBACK_ACK_TIMEOUT_S)
        except asyncio.TimeoutError:
            if self.awaiting_playback and self._same_turn(turn, self.awaiting_playback_turn):
                self.awaiting_playback = False
                self.conversation_until = 0.0
                if not waiter.done():
                    waiter.cancel()
            return "timeout"
        finally:
            if self._playback_waiter is waiter:
                self._playback_waiter = None

    def _mark_playback_interrupted(self, turn: int) -> None:
        """Stop playback and mark a valid interruption for delayed follow-up."""
        if self.awaiting_playback and self._same_turn(turn, self.awaiting_playback_turn):
            self.awaiting_playback = False
            self.awaiting_playback_turn = None
            waiter = self._playback_waiter
            if waiter is not None and not waiter.done():
                waiter.set_result("interrupted")
        self.conversation_until = 0.0
        self._interrupted_turn = turn
        self.playback_status = "interrupted"
        # The browser has already stopped its current MP3, but the speaker can
        # still be ringing.  Keep the wake detector muted until that tail ends.
        self.last_tts = time.time()

    async def _open_interrupted_followup(self, turn: int) -> None:
        """Open the continuation window only after the speaker echo tail."""
        try:
            await asyncio.sleep(ECHO_MUTE_S)
            if (
                self._interrupted_turn != turn
                or self.playback_status != "interrupted"
                or self.turn != turn
                or self.active_turn is not None
            ):
                return
            self.conversation_until = time.monotonic() + FOLLOWUP_S
            log.info("turno %d: follow-up post-interruzione attivo per %.0fs", turn, FOLLOWUP_S)
            await self.send_json({
                "type": "followup", "seconds": FOLLOWUP_S, "interrupted": True,
            })
        except asyncio.CancelledError:
            raise
        finally:
            if self._interrupted_followup_task is asyncio.current_task():
                self._interrupted_followup_task = None

    async def interrupt_current_turn(self, turn) -> bool:
        """Cancel one active turn after validating its exact current id.

        Cancellation closes an in-flight Hermes SSE response and the bounded TTS
        worker.  It is deliberately not converted into a retry: Hermes tools or
        approvals may already have caused side effects.
        """
        task = self._turn_task
        if (
            task is None
            or task.done()
            or not self._same_turn(turn, self.active_turn)
        ):
            return False

        current_turn = self.active_turn
        self._mark_playback_interrupted(current_turn)
        task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("errore durante l'interruzione del turno %s", current_turn)

        await self.send_json({
            "type": "interrupt_ack", "turn": current_turn,
            "status": "interrupted", "echo_tail_ms": int(ECHO_MUTE_S * 1000),
        })
        await self.set_state("listening", turn=current_turn, interrupted=True)
        if self._interrupted_followup_task is not None:
            self._interrupted_followup_task.cancel()
        self._interrupted_followup_task = asyncio.create_task(
            self._open_interrupted_followup(current_turn)
        )
        return True

    async def _send_segmented_audio(self, turn: int, sequence: int, audio: bytes,
                                    started: bool) -> bool:
        if self.playback_status == "interrupted" and self._same_turn(turn, self.active_turn):
            return False
        if not started:
            self._begin_playback(turn)
            await self.set_state("speaking", turn=turn)
            await self.send_json({"type": "audio_start", "turn": turn})
        if self.playback_status == "interrupted" and self._same_turn(turn, self.active_turn):
            return False
        await self.send_json({"type": "audio_chunk", "turn": turn, "seq": sequence})
        if self.ws is None:
            raise RuntimeError("websocket audio non disponibile")
        if self.playback_status == "interrupted" and self._same_turn(turn, self.active_turn):
            return False
        await self.ws.send_bytes(audio)
        self.last_tts = time.time()
        return True

    async def _stream_hermes_speak(self, text: str, turn: int) -> tuple[str, bool, bool]:
        """Stream Hermes deltas into one ordered, bounded TTS worker.

        Returns ``(reply, had_audio, tts_failed)``.  Any stream failure is
        raised as ``HermesStreamTurnError`` after already-queued audio is drained;
        it is intentionally never retried because tools may have run.
        """
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max(1, STREAM_TTS_QUEUE_MAX))
        sentence_buffer = SpeakableSentenceBuffer()
        worker_error: Exception | None = None
        stream_error: Exception | None = None
        started = False
        sequence = 0

        async def on_delta(delta: str):
            for sentence in sentence_buffer.feed(delta):
                await sentence_queue.put(sentence)

        async def on_approval(event: dict):
            # Keep approvals observable without allowing their metadata into
            # spoken text.  The browser can ignore this optional event.
            await self.send_json({"type": "approval", "turn": turn, "approval": event})

        async def tts_worker():
            nonlocal worker_error, started, sequence
            while True:
                sentence = await sentence_queue.get()
                try:
                    if sentence is None:
                        return
                    if worker_error is not None:
                        continue
                    try:
                        audio = await asyncio.to_thread(tts, sentence)
                        if not audio:
                            continue
                        started = await self._send_segmented_audio(
                            turn, sequence, audio, started,
                        )
                        sequence += 1
                    except Exception as exc:
                        worker_error = exc
                finally:
                    sentence_queue.task_done()

        worker = asyncio.create_task(tts_worker())
        try:
            try:
                session_id = self.hermes_session_id
                result = await stream_hermes(
                    text,
                    session_id=session_id,
                    on_delta=on_delta,
                    on_approval=on_approval,
                )
            except Exception as exc:
                stream_error = exc
            else:
                for sentence in sentence_buffer.flush():
                    await sentence_queue.put(sentence)
            await sentence_queue.join()
            await sentence_queue.put(None)
            await worker
        finally:
            if not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        if started:
            await self.send_json({"type": "audio_end", "turn": turn})
        if stream_error is not None:
            raise HermesStreamTurnError("stream Hermes non disponibile", started) from None

        # HermesReply publishes its session id only after [DONE].  Do not touch
        # the WebSocket transcript id before this point, including when TTS
        # fails after the SSE itself completed.
        returned_id = getattr(result, "session_id", None)
        if returned_id:
            self.hermes_session_id = returned_id
        if worker_error is not None:
            # A TTS failure is not an excuse to submit Hermes again.  The final
            # text is still returned once, while the caller reports the audio
            # failure and waits for any already-sent segments to finish.
            return str(result), started, True
        return str(result), started, False

    async def set_state(self, state: str, **extra):
        self.state = state
        await self.send_json({"type": "state", "state": state, **extra})

    # —— wake: runs in a dedicated thread; the event loop must never block ——
    def _remember_audio(self, chunk: bytes) -> None:
        with self.recent_lock:
            self.recent += chunk
            limit = 12 * SAMPLE_RATE * 2
            if len(self.recent) > limit:
                del self.recent[: len(self.recent) - limit]

    def _candidate_from_queue(self, prelude: bytes, wait_for_gate: bool) -> bytes:
        """Return a contiguous candidate, bounded before any cloud STT call."""
        candidate = bytearray(prelude)
        if not wait_for_gate:
            return bytes(candidate)
        target = int(2.5 * SAMPLE_RATE * 2)
        while len(candidate) < target and not self.stop.is_set():
            try:
                chunk = self.recv_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._remember_audio(chunk)
            candidate += chunk
        return bytes(candidate[:target])

    def _launch_local_candidate(self, candidate: bytes) -> bool:
        """Apply local wake/follow-up policy and schedule exactly one turn."""
        if not candidate:
            return False
        followup = time.monotonic() < self.conversation_until and not self.awaiting_playback
        if not followup:
            pcm = np.frombuffer(candidate, dtype=np.int16)
            if not vosk_wake(pcm):
                log.info("candidato parlato scartato dal gate Vosk locale")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
            if WAKE_CONFIRM and not confirm_candidate(pcm):
                log.info("candidato scartato dal secondo gate locale")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
        if time.time() - self.last_tts < ECHO_MUTE_S:
            return False
        self.last_wake = time.time()
        self.state = "waking"
        asyncio.run_coroutine_threadsafe(
            self._on_wake(
                initial_pcm=candidate,
                local_wake_confirmed=not followup,
            ),
            _loop,
        )
        return True

    def wake_worker(self):
        self.engine = None
        if WAKE_PROVIDER != "whisper":
            try:
                self.engine = make_engine()
            except Exception:
                log.exception("impossibile creare l'engine wake")
                asyncio.run_coroutine_threadsafe(
                    self.send_json({"type": "fatal", "error": "wake engine startup failed"}), _loop
                )
                return
        else:
            log.info("wake provider=whisper: VAD energia + gate Vosk locale")
        buf = bytearray()
        speech_streak = 0
        stat_t = time.time()
        stat_frames = 0
        stat_peak = 0.0
        while not self.stop.is_set():
            st = self.state
            if time.time() - stat_t >= 5.0:
                log.info("audio: %d frame/5s, rms peak %.0f, noise floor %.0f, state=%s",
                         stat_frames, stat_peak, self.noise_floor, st)
                stat_t, stat_frames, stat_peak = time.time(), 0, 0.0
            if st in ("waking", "recording"):
                time.sleep(0.05)
                continue
            if st != "listening":
                try:
                    chunk = self.recv_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    return
                buf.clear()
                speech_streak = 0
                continue
            try:
                chunk = self.recv_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                return
            stat_frames += 1
            self._remember_audio(chunk)
            buf += chunk
            while len(buf) >= FRAME * 2:
                frame = bytes(buf[: FRAME * 2])
                del buf[: FRAME * 2]
                arr = np.frombuffer(frame, dtype=np.int16)
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                stat_peak = max(stat_peak, rms)
                thr = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
                if rms < thr:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 1.0)
                    self.last_silent = time.time()
                elif time.time() - self.last_silent > 3.0:
                    self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                if WAKE_PROVIDER == "whisper":
                    speech_streak = speech_streak + 1 if rms >= thr else 0
                    need = 3 if self.noise_floor < 2000 else 8
                    hit = speech_streak >= need
                else:
                    try:
                        hit = self.engine.process(arr)
                    except Exception:
                        log.exception("errore engine")
                        hit = False
                now = time.time()
                if hit and now - self.last_tts < ECHO_MUTE_S:
                    continue
                if hit and now - self.last_wake >= COOLDOWN_S and rms > 200:
                    with self.recent_lock:
                        prelude = bytes(self.recent[-int(1.5 * SAMPLE_RATE * 2):])
                    followup = time.monotonic() < self.conversation_until and not self.awaiting_playback
                    candidate = self._candidate_from_queue(prelude, wait_for_gate=not followup)
                    if self._launch_local_candidate(candidate) or followup:
                        buf.clear()
                        speech_streak = 0
                        break
                    buf.clear()
                    speech_streak = 0

    async def _on_wake(self, initial_pcm: bytes = b"",
                       local_wake_confirmed: bool = False):
        current_task = asyncio.current_task()
        self.turn += 1
        turn = self.turn
        self._turn_paid_s = 0.0
        self.active_turn = turn
        self._turn_task = current_task

        def clear_active_turn(done_task):
            if self._turn_task is done_task:
                self._turn_task = None
            if self.active_turn == turn:
                self.active_turn = None

        if current_task is not None:
            current_task.add_done_callback(clear_active_turn)
        t_turn = time.time()
        # The window must be evaluated when speech starts, NOT after STT:
        # On slow CPUs transcription can take many seconds.
        followup_at_start = time.monotonic() < self.conversation_until and not self.awaiting_playback
        await self.set_state("waking", turn=turn, followup=followup_at_start)
        await self._clear_partial(turn)
        # pre-roll: the wake phrase has already passed while the trigger decides,
        # so restart from the last 1.5 s already held in the rotating buffer
        prelude = bytes(initial_pcm)
        if not prelude and WAKE_PROVIDER == "whisper":
            with self.recent_lock:
                prelude = bytes(self.recent[-int(1.5 * SAMPLE_RATE * 2):])
        # pause only if a reply was just played (TTS echo)
        if time.time() - self.last_tts < 3.0:
            await asyncio.sleep(0.25)
        await self.set_state("recording", turn=turn, followup=followup_at_start)
        realtime = None
        realtime_start_failed = False
        if STT_BACKEND == stt_backends.REALTIME_BACKEND:
            try:
                realtime = await stt_backends.RealtimeScribe.connect(
                    on_partial=lambda text: self._send_partial(text, turn),
                )
            except Exception:
                # _transcribe_realtime_or_batch falls back locally; this flag
                # avoids pretending that a provider connection existed.
                realtime_start_failed = True
                await self.send_json({"type": "stt_status", "mode": "local", "turn": turn})
        try:
            pcm = await self._record_utterance(prelude, realtime=realtime)
        except Exception:
            if realtime is not None:
                await realtime.close()
            log.exception("errore registrazione")
            await self._clear_partial(turn)
            await self.set_state("listening")
            return
        if pcm is None or len(pcm) < int(MIN_SPEECH_S * SAMPLE_RATE):
            if realtime is not None:
                await realtime.close()
            await self._clear_partial(turn)
            await self.set_state("listening", note="niente da trascrivere")
            return

        try:
            saved = save_turn_audio(pcm)
            log.info("turno %d: audio per diagnostica %s", turn, saved.name)
        except Exception:
            log.exception("impossibile salvare il WAV diagnostico")
        await self.set_state("transcribing", turn=turn)
        realtime_result_used_batch = False
        try:
            if STT_BACKEND == stt_backends.REALTIME_BACKEND:
                text, realtime_result_used_batch = await _transcribe_realtime_or_batch(
                    pcm, realtime, turn, start_failed=realtime_start_failed
                )
            elif STT_BACKEND == "vosk":
                text = await asyncio.to_thread(decode_utterance, pcm, followup_at_start)
            else:
                text = await asyncio.to_thread(stt_transcribe, pcm)
        except Exception as exc:
            log.exception("STT fallito")
            await self._clear_partial(turn)
            await self.set_state("listening", error=f"stt: {exc}")
            return
        if text is None or (STT_BACKEND != "vosk" and not text):
            await self._clear_partial(turn)
            await self.set_state("listening")
            return
        if WAKE_PROVIDER == "whisper":
            if STT_BACKEND == "vosk":
                cmd = text
            elif STT_BACKEND == stt_backends.REALTIME_BACKEND:
                if followup_at_start:
                    cmd = text.strip()
                elif local_wake_confirmed:
                    # The trusted local grammar is the gate. Realtime may
                    # omit or garble the wake token, so do not ask a paid
                    # provider to verify it a second time.
                    cmd = resolve_vosk_command(text, wake=True, followup=False)
                else:
                    # Kept for direct/unit callers that invoke _on_wake()
                    # without the production candidate path.
                    cmd = resolve_command(text, 0.0, time.monotonic())
            else:
                cmd = resolve_command(
                    text, float("inf") if followup_at_start else 0.0,
                    time.monotonic()
                )
            if cmd is None:
                log.info("turno %d scartato (fuori dalla conversazione): %r", turn, text[:80])
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                await self._clear_partial(turn)
                await self.set_state("listening", note="non era per me")
                return
            if not cmd.strip():
                log.info("turno %d: sveglia senza comando", turn)
                await self._clear_partial(turn)
                await self._speak("Sì?", turn)
                await self.set_state("listening", turn=turn)
                return
            text = cmd.strip()
            self.conversation_until = 0.0  # next turn only after playback
            log.info("comando %s: %r", "follow-up" if followup_at_start else "wake", text[:120])
            self._partial_turn = None
            await self.send_json({"type": "transcript", "text": text, "turn": turn, "command": True})
        else:
            self._partial_turn = None
            await self.send_json({"type": "transcript", "text": text, "turn": turn})

        await self.set_state("thinking", turn=turn)
        if AGENT_BACKEND == "hermes":
            stream_had_audio = False
            try:
                reply, stream_had_audio, tts_failed = await self._stream_hermes_speak(text, turn)
            except HermesStreamTurnError as exc:
                # The stream may already have run tools or emitted audio.  Never
                # reissue this turn through ask_hermes; surface only a safe error.
                log.warning("turno %d: %s", turn, exc)
                reply = "Non riesco a completare la risposta."
                tts_failed = False
                stream_had_audio = exc.had_audio
            except Exception:
                log.exception("stream Hermes fallito")
                reply = "Non riesco a completare la risposta."
                tts_failed = False
            await self.send_json({"type": "reply", "text": reply, "turn": turn})
            if stream_had_audio:
                playback = await self._wait_for_playback(turn)
                if playback == "completed":
                    await self.set_state("listening", turn=turn)
                elif playback == "failed":
                    await self.set_state("listening", turn=turn, error="playback fallito")
                else:
                    await self.set_state("listening", turn=turn, error="playback timeout")
            elif tts_failed:
                await self.set_state("listening", turn=turn, error="tts non disponibile")
            elif reply:
                # A successful stream with no usable MP3 is safe to handle with
                # the old single-response TTS path: Hermes is not called again.
                await self._speak(reply, turn)
                if self.awaiting_playback:
                    playback = await self._wait_for_playback(turn)
                    if playback == "completed":
                        await self.set_state("listening", turn=turn)
                    elif playback == "failed":
                        await self.set_state("listening", turn=turn, error="playback fallito")
                    else:
                        await self.set_state("listening", turn=turn, error="playback timeout")
                else:
                    await self.set_state("listening", turn=turn)
            else:
                await self.set_state("listening", turn=turn)
            self._record_usage()
            log.info("turno %d completato in %.1fs", turn, time.time() - t_turn)
            return

        # Non-streaming backends retain the legacy single MP3 protocol.
        try:
            reply = await self._ask_hermes(text)
        except HermesContinuationError:
            log.warning("turno %d: continuità Hermes non disponibile", turn)
            reply = "Non riesco a continuare questa conversazione."
        except Exception as exc:
            log.exception("agente fallito")
            reply = f"Non sono riuscito a contattare l'agente: {exc}"
        await self.send_json({"type": "reply", "text": reply, "turn": turn})
        if not reply:
            await self.set_state("listening", turn=turn)
            return

        await self._speak(reply, turn)
        if self.awaiting_playback:
            playback = await self._wait_for_playback(turn)
            if playback == "completed":
                await self.set_state("listening", turn=turn)
            elif playback == "failed":
                await self.set_state("listening", turn=turn, error="playback fallito")
            else:
                await self.set_state("listening", turn=turn, error="playback timeout")
        else:
            await self.set_state("listening", turn=turn)
        log.info("turno %d completato in %.1fs", turn, time.time() - t_turn)

    async def _speak(self, text: str, turn: int):
        """TTS -> audio to the client; on error log and return to listening."""
        await self.set_state("speaking", turn=turn)
        try:
            audio = await asyncio.to_thread(tts, text)
        except Exception as exc:
            log.exception("TTS fallito")
            await self.set_state("listening", error=f"tts: {exc}")
            return
        if audio:
            self._begin_playback(turn)
            await self.send_json({"type": "audio", "fmt": "mp3", "bytes": len(audio), "turn": turn})
            if self.ws is None:
                await self.set_state("listening", error="websocket audio non disponibile")
                self.awaiting_playback = False
                return
            await self.ws.send_bytes(audio)
            self.last_tts = time.time()

    def _record_usage(self) -> None:
        """Best-effort accounting for the monthly usage report."""
        import datetime
        try:
            paid = float(getattr(self, "_turn_paid_s", 0.0) or 0.0)
            USAGE_LEDGER.record(
                datetime.date.today().isoformat(),
                turns=1,
                realtime_s=paid,
                local_turns=0 if paid else 1,
            )
        except Exception:
            log.exception("impossibile registrare l'usage")
        finally:
            self._turn_paid_s = 0.0

    async def _record_utterance(self, prelude: bytes = b"", realtime=None):
        """VAD: collect PCM while speech lasts, then close after SILENCE_END_S of silence.

        Timing is computed on received samples (not the wall clock), so the
        behavior stays identical with the live stream and in tests.
        `prelude` is the audio that precedes the trigger (it keeps the wake
        phrase, spoken before the system reacts, from being lost).
        """
        collected = bytearray(prelude)
        voiced_s = 0.0
        idle_s = 0.0
        started = False
        last_data = time.time()
        t_end_max = time.time() + MAX_UTTERANCE_S
        reported_stt_mode = None

        async def forward_audio(chunk: bytes):
            nonlocal reported_stt_mode
            assert realtime is not None
            sent = await realtime.send_audio(chunk)
            if sent:
                self._turn_paid_s = getattr(self, "_turn_paid_s", 0.0) + len(chunk) / (SAMPLE_RATE * 2)
            mode = "realtime" if sent else "local"
            if mode != reported_stt_mode:
                reported_stt_mode = mode
                await self.send_json({"type": "stt_status", "mode": mode, "turn": self.turn})

        if realtime is not None and prelude:
            # The local wake detector fired before recording began. Sending this
            # confirmed pre-roll after opening the provider preserves the wake
            # gate without ever keeping an idle provider connection alive.
            await forward_audio(prelude)
        if prelude:
            # A candidate is already buffered before the recorder starts. Feed
            # it through the same local VAD accounting so a quiet gap at the
            # queue boundary cannot discard the first spoken words.
            seed = np.frombuffer(prelude, dtype=np.int16)
            seed_threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
            for offset in range(0, len(seed), 1600):
                frame = seed[offset:offset + 1600]
                if len(frame) == 0:
                    continue
                dt = len(frame) / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                if rms >= seed_threshold:
                    started = True
                    voiced_s += dt
                    idle_s = 0.0
                elif started:
                    idle_s += dt
        while time.time() < t_end_max:
            try:
                chunk = await asyncio.to_thread(self.recv_queue.get, True, 0.2)
            except queue.Empty:
                # stream interrupted (client stalled): close instead of hanging
                if time.time() - last_data > 3.0:
                    break
                continue
            if chunk is None or len(chunk) < 2:
                break
            last_data = time.time()
            collected += chunk
            arr = np.frombuffer(chunk, dtype=np.int16)
            dt = len(arr) / SAMPLE_RATE
            if realtime is not None:
                # Realtime STT needs one contiguous stream.  Local VAD still
                # decides when the turn ends, but must not punch quiet holes in
                # the audio sent to the provider.
                await forward_audio(chunk)
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
            # The floor adapts ONLY before the first speech: during an
            # utterance the quietest syllables would be learned as ambient noise,
            # the threshold would rise above the speaker's own peaks and the
            # recorder would hang (real failure: 22 s collected, sentence
            # spoken twice in the transcript).
            if not started:
                if rms < threshold:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 1.0)
                    self.last_silent = time.time()
                elif time.time() - self.last_silent > 3.0:
                    self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                    self.last_silent = time.time()
                    threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
                    log.info("VAD: floor adattato al rumore ambiente -> soglia %.0f", threshold)
            if rms >= threshold:
                if not started:
                    log.info("VAD: parlato rilevato (rms=%.0f soglia=%.0f)", rms, threshold)
                started = True
                idle_s = 0.0
                voiced_s += dt
            elif started:
                idle_s += dt
                # Close anyway after double the expected silence: the wake or the
                # follow-up already confirmed spoken intent; the voiced quorum
                # must never be able to hang the recorder.
                if idle_s >= SILENCE_END_S and (
                        voiced_s >= MIN_SPEECH_S or idle_s >= SILENCE_END_S + 2.0):
                    break
            else:
                idle_s += dt
                if idle_s >= IDLE_ABORT_S:
                    log.info("VAD: abort, nessun parlato entro %.1fs", IDLE_ABORT_S)
                    return None
            if idle_s >= SILENCE_END_S and (
                    voiced_s >= MIN_SPEECH_S or idle_s >= SILENCE_END_S + 2.0):
                break
        pcm = np.frombuffer(bytes(collected), dtype=np.int16)
        log.info("VAD: raccolti %.2fs (parlato %.2fs, started=%s)", len(pcm) / SAMPLE_RATE, voiced_s, started)
        # Never trim the head: the pre-roll contains the wake, often quieter
        # than the command. The previous energy trim removed the wake phrase,
        # leaving only "what's the weather tomorrow...", which the regex then
        # rejected. Whisper already has its own VAD for leading silence.
        start = 0
        thr = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT) * 0.5
        win = 1600
        end = len(pcm)
        for i in range(len(pcm) - win, start, -win):
            if float(np.sqrt(np.mean(pcm[i:i + win].astype(np.float32) ** 2))) >= thr:
                end = min(len(pcm), i + 2 * win)
                break
        log.info("VAD: utterance tagliata %.2fs -> %.2fs",
                 len(pcm) / SAMPLE_RATE, (end - start) / SAMPLE_RATE)
        return pcm[start:end]


app = FastAPI(title="Lari")
_loop: asyncio.AbstractEventLoop | None = None
_sessions: set[Session] = set()


@app.get("/")
async def root():
    return Response(content="lari: open /<token>/", media_type="text/plain")


@app.get("/{token}/debug/last.wav")
async def debug_last(token: str):
    """The last ~12 s of microphone audio, to calibrate threshold/phrase offline."""
    if not TOKEN or token != TOKEN:
        return Response(content="token errato", status_code=403)
    for s in _sessions:
        with s.recent_lock:
            data = bytes(s.recent)
        if data:
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                path = fh.name
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(data)
            return FileResponse(path, media_type="audio/wav", filename="last.wav")
    return Response(content="nessuna sessione attiva", status_code=404)


@app.get("/{token}/")
async def page(token: str):
    if not TOKEN or token != TOKEN:
        return Response(content="token errato", status_code=403)
    return FileResponse(
        BASE_DIR / "static" / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, must-revalidate"},   # no stale HTML from cache
    )


@app.get("/{token}/assets/{asset_name}")
async def mascot_asset(token: str, asset_name: str):
    """Serve only bundled mascot and branding assets to clients with the installation token."""
    if not TOKEN or token != TOKEN:
        return Response(content="Forbidden", status_code=403)
    if asset_name not in {
        "lare-concept.svg", "lare-idle.svg", "lare-listening.svg",
        "lare-thinking.svg", "lare-speaking.svg", "lare-error.svg",
        "logo.jpg", "favicon.ico", "apple-touch-icon.png", "og.png",
        "icon-192.png", "icon-512.png",
    }:
        return Response(content="Not found", status_code=404)
    path = BASE_DIR / "static" / "assets" / asset_name
    if not path.is_file():
        return Response(content="Not found", status_code=404)
    media_type = {
        ".svg": "image/svg+xml", ".ico": "image/x-icon",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    }[path.suffix]
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/{token}/manifest.webmanifest")
async def pwa_manifest(token: str):
    """Install manifest scoped to this installation's URL space."""
    if not TOKEN or token != TOKEN:
        return Response(content="Forbidden", status_code=403)
    icons = "/%s/assets" % token
    payload = {
        "name": "Lari \u2014 Lare",
        "short_name": "Lari",
        "start_url": "/%s/" % token,
        "scope": "/%s/" % token,
        "display": "fullscreen",
        "background_color": "#010000",
        "theme_color": "#010000",
        "icons": [
            {"src": "%s/icon-192.png" % icons, "sizes": "192x192", "type": "image/png"},
            {"src": "%s/icon-512.png" % icons, "sizes": "512x512", "type": "image/png"},
        ],
    }
    return Response(content=json.dumps(payload), media_type="application/manifest+json")


@app.get("/{token}/sw.js")
async def pwa_service_worker(token: str):
    """Service worker (scope: the app's own URL space)."""
    if not TOKEN or token != TOKEN:
        return Response(content="Forbidden", status_code=403)
    return FileResponse(BASE_DIR / "static" / "sw.js",
                        media_type="application/javascript")


@app.get("/{token}/usage")
async def usage_summary(token: str):
    """Monthly usage summary: turns, paid realtime seconds, local turns."""
    if not TOKEN or token != TOKEN:
        return Response(content="Forbidden", status_code=403)
    rate_env = os.environ.get("BUDDY_USAGE_EUR_PER_MIN", "").strip()
    rate = float(rate_env) if rate_env else None
    summary = USAGE_LEDGER.month_summary(time.strftime("%Y-%m"), eur_per_min=rate)
    return Response(content=json.dumps(summary), media_type="application/json")


@app.websocket("/{token}/ws")
async def ws_endpoint(token: str, websocket: WebSocket):
    if not TOKEN or token != TOKEN:
        await websocket.close(code=4403)
        return
    await websocket.accept()
    log.info("client connesso da %s", websocket.client)
    session = Session(websocket, _make_sender(websocket))
    _sessions.add(session)
    worker = threading.Thread(target=session.wake_worker, daemon=True, name="wake-worker")
    worker.start()
    await session.send_json({
        "type": "state", "state": "listening",
        "phrase": WAKE_CONFIG.display, "provider": WAKE_PROVIDER,
        "voice": TTS_VOICE, "sensitivity": WAKE_SENSITIVITY,
        "confirm_frames": CONFIRM_FRAMES,
    })
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                if session.recv_queue.qsize() < session.recv_queue.maxsize:
                    session.recv_queue.put_nowait(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "ping":
                    await session.send_json({"type": "pong", "state": session.state})
                elif data.get("type") == "playback_done":
                    status = data.get("status", "completed")
                    accepted = session.mark_playback_done(
                        turn=data.get("turn"), status=status,
                    )
                    if accepted and status == "completed":
                        log.info("turno %d: follow-up attivo per %.0fs", session.turn, FOLLOWUP_S)
                        await session.send_json({"type": "followup", "seconds": FOLLOWUP_S})
                elif data.get("type") == "interrupt":
                    accepted = await session.interrupt_current_turn(data.get("turn"))
                    if not accepted:
                        await session.send_json({
                            "type": "interrupt_rejected", "turn": data.get("turn"),
                        })
                elif data.get("type") == "diag":
                    log.info("diag phone: ctx=%s rate=%s mic=%s frames=%s vis=%s raw=%s",
                             data.get("ctx"), data.get("rate"), data.get("mic"),
                             data.get("frames"), data.get("vis"), data.get("raw"))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket error")
    finally:
        session.stop.set()
        if session._interrupted_followup_task is not None:
            session._interrupted_followup_task.cancel()
        try:
            session.recv_queue.put_nowait(None)
        except Exception:
            pass
        # calibration: save the last ~12 s of microphone so it can be
        # re-analyzed offline after each user test
        try:
            with session.recent_lock:
                data = bytes(session.recent)
            if len(data) >= 2 * SAMPLE_RATE * 2:
                calib = BASE_DIR / "calibration"
                calib.mkdir(exist_ok=True)
                import wave as _wave
                dest = calib / f"mic_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                with _wave.open(str(dest), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                    w.writeframes(data)
                for old in sorted(calib.glob("mic_*.wav"))[:-8]:
                    old.unlink()
                log.info("calibrazione salvata: %s (%.1fs)", dest.name, len(data) / 2 / SAMPLE_RATE)
        except Exception:
            log.exception("salvataggio calibrazione fallito")
        log.info("sessione chiusa (turni: %d, ultimo stato: %s)", session.turn, session.state)
        _sessions.discard(session)
        if session.engine:
            try:
                session.engine.close()
            except Exception:
                pass
        log.info("sessione chiusa")


def _make_sender(websocket: WebSocket):
    async def send(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    return send


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()
    if not TOKEN:
        log.warning("BUDDY_TOKEN non imposto: il server rifiuta tutto")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
