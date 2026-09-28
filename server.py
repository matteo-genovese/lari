"""Lari voice bridge: satellite microphone -> local wake gate -> STT -> Hermes -> TTS.

The phone browser (or a future dedicated device) acts as a lightweight
microphone and speaker: it streams mono 16 kHz PCM over WebSocket.

Per-turn pipeline:
  PCM -> local wake phrase confirmation
      -> adaptive-energy VAD
      -> configured local or hosted STT
      -> Hermes API with a persistent transcript session
      -> edge-tts audio streamed back to the client

Access is protected by a per-installation URL token. Do not expose the bridge
without TLS and an appropriate private-network or reverse-proxy boundary.
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

import stt_backends


def setting(name: str, default=None):
    """Read current LARI settings while accepting legacy BUDDY_* names."""
    return os.environ.get(f"LARI_{name}", os.environ.get(f"BUDDY_{name}", default))


BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_HERMES_ROOT = Path.home() / ".hermes" / "hermes-agent"
HERMES_ROOT = Path(str(setting("HERMES_ROOT", "")).strip() or _DEFAULT_HERMES_ROOT).expanduser()
if HERMES_ROOT.exists():
    sys.path.insert(0, str(HERMES_ROOT))

try:
    from tools.wake_word import _build_engine  # noqa: E402
except ImportError:
    _build_engine = None

# Configuration
TOKEN = str(setting("TOKEN", "")).strip()
PORT = int(setting("PORT", "8643"))
HERMES_API = setting("HERMES_API", "http://127.0.0.1:8642")
HERMES_KEY = setting("HERMES_KEY", "")
SESSION_KEY = setting("SESSION_KEY", "lari")
HERMES_PROVIDER = setting("HERMES_PROVIDER", "")
HERMES_MODEL = setting("HERMES_MODEL", "")
AGENT_BACKEND = str(setting("AGENT_BACKEND", "hermes")).strip()

def _deepseek_from_hermes_env() -> tuple[str, str]:
    """Load optional DeepSeek credentials and endpoint from Hermes' environment."""
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
DEEPSEEK_MODEL = setting("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_TIMEOUT_S = float(setting("DEEPSEEK_TIMEOUT", "60"))
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


SAMPLE_RATE = 16000
FRAME = 1280              # Recommended 80 ms frame at 16 kHz for the wake engine

def wake_command(text: str) -> str | None:
    """Return the command following the wake phrase, or None when it is absent."""
    match = WAKE_RE.search(text)
    if not match:
        return None
    return text[match.end():].lstrip(" ,.!?;:-\u2019'\u201c\u201d")
def resolve_command(text: str, conversation_until: float, now: float) -> str | None:
    """Require the wake phrase on the first turn, then allow follow-ups in-window."""
    command = wake_command(text)
    if command is not None:
        return command
    if now < conversation_until and text.strip():
        return text.strip()
    return None

LANGUAGE = str(setting("LANGUAGE", "it")).strip().lower()
if LANGUAGE not in {"it", "en"}:
    raise ValueError("LARI_LANGUAGE must be 'it' or 'en'")


def localized(en: str, it: str) -> str:
    """Choose a user-facing message for the configured interface language."""
    return en if LANGUAGE == "en" else it


_default_wake_phrase = "Ehi Lari" if LANGUAGE == "it" else "Hey Lari"
WAKE_PHRASE = str(setting("WAKE_PHRASE", _default_wake_phrase)).strip() or _default_wake_phrase
# Additional pronunciations and ASR spellings are installation settings, not brand defaults.
WAKE_ALIASES = tuple(part.strip() for part in str(setting("WAKE_ALIASES", "")).split(",") if part.strip())
_wake_phrases = (WAKE_PHRASE, *WAKE_ALIASES)
if not WAKE_PHRASE.split():
    raise ValueError("LARI_WAKE_PHRASE must contain at least one word")
WAKE_RE = re.compile(
    r"^\s*(?:" + "|".join(
        r"[\s,.!?:;-]+".join(map(re.escape, phrase.split())) + r"\b"
        for phrase in sorted(_wake_phrases, key=len, reverse=True)
    ) + r")",
    re.IGNORECASE,
)
_default_voice_system = (
    "You are a voice assistant. Spoken requests may be transcribed imperfectly. Reply concisely in English without markdown or lists. Use tools only when needed."
    if LANGUAGE == "en"
    else "Sei un assistente vocale. Le richieste possono essere trascritte in modo imperfetto. Rispondi in modo conciso in italiano, senza markdown o liste. Usa gli strumenti solo quando servono."
)
VOICE_SYSTEM = setting("VOICE_SYSTEM", _default_voice_system)
TTS_VOICE = str(setting("TTS_VOICE", "")).strip() or ("it-IT-ElsaNeural" if LANGUAGE == "it" else "en-US-JennyNeural")
STT_MODEL = setting("STT_MODEL", "base")
STT_BACKEND = str(setting("STT_BACKEND", "whisper")).strip()
VOSK_MODEL_DIR = Path(str(setting("VOSK_MODEL_DIR", "")).strip() or str(BASE_DIR / f"models/vosk-model-small-{LANGUAGE}")).expanduser()
STT_LANG = str(setting("STT_LANG", "")).strip() or LANGUAGE
STT_GATE = str(setting("STT_GATE", "")).strip()
WAKE_PROVIDER = setting("WAKE_PROVIDER", "whisper")
WAKE_SENSITIVITY = float(setting("SENSITIVITY", "0.5"))
CONFIRM_FRAMES = int(setting("CONFIRM_FRAMES", "3"))
COOLDOWN_S = float(setting("COOLDOWN_S", "2"))
AMBIENT_PAUSE_S = float(setting("AMBIENT_PAUSE", "6"))
ECHO_MUTE_S = float(setting("ECHO_MUTE", "2.5"))
FOLLOWUP_S = float(setting("FOLLOWUP_S", "30"))
PLAYBACK_ACK_TIMEOUT_S = float(setting("PLAYBACK_ACK_TIMEOUT", "45"))
STREAM_TTS_QUEUE_MAX = int(setting("STREAM_TTS_QUEUE_MAX", "8"))
STREAM_TEXT_MAX_CHARS = int(setting("STREAM_TEXT_MAX_CHARS", "4000"))
STREAM_SENTENCE_MAX_CHARS = int(setting("STREAM_SENTENCE_MAX_CHARS", "280"))
VAD_MIN_RMS = float(setting("VAD_MIN_RMS", "900"))
VAD_NOISE_MULT = float(setting("VAD_NOISE_MULT", "2.5"))
SILENCE_END_S = float(setting("SILENCE_END", "2.5"))
MAX_UTTERANCE_S = float(setting("MAX_UTTERANCE_S", "45"))
MIN_SPEECH_S = float(setting("MIN_SPEECH_S", "0.7"))
IDLE_ABORT_S = float(setting("IDLE_ABORT_S", "4.0"))
AGENT_TIMEOUT_S = float(setting("AGENT_TIMEOUT", "180"))
CLI_TIMEOUT_S = float(setting("CLI_TIMEOUT", "70"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

log = logging.getLogger("lari")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "server.log")],
)

_stt: dict = {}
_stt_lock = threading.Lock()


def get_stt(model: str | None = None):
    """Lazily load and cache faster-whisper models."""
    with _stt_lock:
        name = model or STT_MODEL
        if name not in _stt:
            from faster_whisper import WhisperModel
            log.info("Loading STT model=%s", name)
            _stt[name] = WhisperModel(name, device="cpu", compute_type="int8")
        return _stt[name]


def transcribe(pcm: np.ndarray, model: str | None = None) -> str:
    """Transcribe mono 16 kHz int16 PCM and return the recognized text."""
    audio = pcm.astype(np.float32) / 32768.0
    # Pin the language: automatic detection can be unreliable on short, noisy clips.
    # An empty value enables automatic language detection.
    lang = STT_LANG or None
    segments, info = get_stt(model).transcribe(
        audio,
        language=lang,
        beam_size=1,                        # Beam size 5 was measured to be 3x slower with little accuracy improvement.
        vad_filter=True,
        vad_parameters={"threshold": 0.6,   # Trim short noise fragments more aggressively.
                        "min_silence_duration_ms": 400},
        hallucination_silence_threshold=2.0,  # Reject phrases hallucinated from trailing silence
        condition_on_previous_text=False,   # Prevent the model from repeating its own prior output
        # Keep the prompt short: long prompts can be hallucinated in place of the audio,
        # especially on noisy recordings.
        initial_prompt=f"{WAKE_PHRASE}.",
    )
    text = " ".join(s.text for s in segments).strip()
    log.info("STT lang=%s prob=%.2f -> %r", info.language, info.language_probability, text)
    # Hallucination guard: Whisper can invent short, recurring phrases from near-silence
    # or noise. Discard text without any credible speech instead of sending it to the agent.
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ']{3,}", text)]
    black = {"im sorry", "i'm sorry", "sorry", "thanks for watching", "subscrib", "musica"}
    low = text.lower().strip(" .!?")
    if len(words) < 1 or low in black:
        log.info("Discarded hallucinated STT text: %r", text[:60])
        return ""
    return text


_vosk_model = None
_vosk_lock = threading.Lock()

def get_vosk():
    """Load the language-specific Vosk model once; create a recognizer per call."""
    global _vosk_model
    with _vosk_lock:
        if _vosk_model is None:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            _vosk_model = Model(str(VOSK_MODEL_DIR))
        return _vosk_model

def vosk_wake(pcm: np.ndarray) -> bool:
    """Check the wake phrase in the first 2.5 seconds without constraining free-form transcription."""
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(
        get_vosk(), SAMPLE_RATE,
        json.dumps([*(phrase.casefold() for phrase in _wake_phrases), "[unk]"]),
    )
    rec.AcceptWaveform(pcm[:int(2.5 * SAMPLE_RATE)].astype(np.int16, copy=False).tobytes())
    heard = json.loads(rec.FinalResult()).get("text", "")
    log.info("Local wake candidate: %r", heard)
    return bool(WAKE_RE.search(heard))

def transcribe_vosk(pcm: np.ndarray) -> str:
    """Transcribe locally on CPU, processing the utterance in chunks."""
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

def resolve_vosk_command(text: str, wake: bool, followup: bool) -> str | None:
    if not wake and not followup:
        return None
    if wake:
        command = wake_command(text)
        if command is not None:
            return command
        return text.strip() or None
    return text.strip() or None

def _transcribe_local_fallback(pcm: np.ndarray) -> str:
    """Transcribe realtime failures without making a paid provider request."""
    try:
        return transcribe_vosk(pcm)
    except Exception:
        log.exception("Local Vosk backend unavailable; falling back to local faster-whisper")
        return transcribe(pcm)


def stt_transcribe(pcm: np.ndarray) -> str:
    """Return raw text from the configured cloud or local STT backend.

    This is the single dispatch point; the wake handler must not select a
    backend separately.
    """
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
            "turn %d: ElevenLabs Realtime STT unavailable; falling back to local STT",
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
    """Transcribe and validate the wake phrase; None indicates non-addressed speech."""
    backend = backend or STT_BACKEND
    if backend == "vosk":
        # The wake grammar only checks the beginning and does not constrain free-form transcription.
        wake = False if followup else vosk_wake(pcm)
        if not wake and not followup:
            return None
        text = transcribe_vosk(pcm)
        return resolve_vosk_command(text, wake, followup) if text else None
    if backend in stt_backends.BACKENDS:
        # Hosted STT uses the same wake-prefix validation as local Whisper. Evaluate the
        # follow-up window at the beginning of the turn.
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
    raise ValueError(f"Unsupported STT backend: {backend}")

def save_turn_audio(pcm: np.ndarray, directory: Path | None = None,
                    name: str | None = None, keep: int = 5) -> Path:
    """Keep a small number of local turn recordings for reproducible diagnostics."""
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
    """Convert text to MP3 with edge-tts."""
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
            raise RuntimeError("Hermes response exceeded the configured length limit")
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
    """Invia la battuta all'agente, continuando ``session_id`` quando presente.

    The return value remains string-compatible for existing callers and carries the
    response's ``X-Hermes-Session-Id`` as ``.session_id``.
    """
    import httpx

    payload = {
        "model": HERMES_MODEL or DEEPSEEK_MODEL,
        "provider": HERMES_PROVIDER or "deepseek",
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
    except Exception as exc:  # API server unavailable -> fall back to the CLI
        if session_id:
            # A CLI continuation has different state semantics. Keep the explicit
            # transcript id private to this WebSocket and surface a concise error.
            log.warning("Hermes continuation failed (%s)", type(exc).__name__)
            raise HermesContinuationError("Hermes continuation is unavailable") from None
        log.warning("Hermes API server unreachable (%s); falling back to hermes chat -q", type(exc).__name__)
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
        "model": HERMES_MODEL or DEEPSEEK_MODEL,
        "provider": HERMES_PROVIDER or "deepseek",
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
            raise RuntimeError("Hermes stream contains invalid SSE JSON") from exc

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
                    raise RuntimeError("Hermes stream ended with an error")
                raise RuntimeError(
                    f"Hermes stream ended with finish_reason={finish_reason!r}"
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
        raise RuntimeError("Hermes stream incomplete: missing [DONE]")
    if not saw_stop_finish:
        raise RuntimeError("Hermes stream incomplete: missing finish_reason='stop'")
    return HermesReply("".join(content).strip(), response_session_id, run_id)


def _ask_cli(text: str) -> str:
    """Fallback without the API server. DEVNULL stdin makes approval prompts
    fail quickly instead of hanging while waiting for a nonexistent TTY.
    """
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
        if out:
            return out
        detail = (p.stderr or "").strip()[:200]
        if LANGUAGE == "en":
            return f"(No response; exit={p.returncode}: {detail})"
        return f"(Nessuna risposta; exit={p.returncode}: {detail})"
    except sp.TimeoutExpired:
        log.error("Fallback CLI timed out after %.0fs", CLI_TIMEOUT_S)
        if LANGUAGE == "en":
            return "(The backend is slow; start the API server by restarting the gateway.)"
        return "(Il backend è lento: avvia l'API server riavviando il gateway.)"
    except Exception as exc:
        return f"(Agent error: {exc})" if LANGUAGE == "en" else f"(Errore agente: {exc})"


async def _post_chat(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    """POST JSON to a chat-completions endpoint (the single mockable HTTP boundary)."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def ask_deepseek(text: str) -> str:
    """Call DeepSeek directly; this backend is fast but does not provide tools.
    It must disclose missing live data instead of inventing current facts."""
    if not DEEPSEEK_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    limitations = (
        " You have no tools or live data (weather, news, schedules). If a request "
        "needs them, say so briefly instead of guessing."
        if LANGUAGE == "en"
        else " Non hai strumenti o dati in tempo reale (meteo, notizie, orari). "
        "Se una richiesta li richiede, dillo brevemente invece di inventare una risposta."
    )
    system = VOICE_SYSTEM + limitations
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
    """Dispatch a voice request to the configured agent backend.
    Fall back to the full Hermes agent if the direct backend is unavailable."""
    if AGENT_BACKEND == "deepseek":
        try:
            return await ask_deepseek(text)
        except Exception as exc:
            log.warning("DeepSeek unavailable (%s); falling back to the Hermes agent", exc)
    if session_id is None:
        return await ask_hermes(text)
    return await ask_hermes(text, session_id=session_id)

# ─── wake engine ─────────────────────────────────────────────────────────────
def make_engine():
    if _build_engine is None:
        raise RuntimeError("This wake provider requires Hermes wake-word modules on the Python path")
    cfg = {
        "provider": WAKE_PROVIDER,
        "phrase": WAKE_PHRASE,
        "sensitivity": WAKE_SENSITIVITY,
        "confirmation_frames": CONFIRM_FRAMES,
        "profile_routing": False,   # Match the wake phrase without Hermes profile routing
        "openwakeword": {"model": "hey_hermes"},
    }
    return _build_engine(cfg)


class Session:
    """State for one satellite connection (one device maps to one session)."""

    def __init__(self, ws: WebSocket, send_json):
        self.ws = ws
        self.send_json = send_json
        self.engine = None
        self.state = "listening"           # The worker starts in the listening state
        self.frames = bytearray()          # Partial bytes waiting for a complete frame
        self.last_wake = 0.0
        self.noise_floor = 500.0           # Exponential moving average of the noise floor
        self.last_silent = time.time()     # Most recent genuinely silent frame
        self.recv_queue: queue.Queue = queue.Queue(maxsize=400)
        self.stop = threading.Event()
        self.turn = 0
        # Rolling diagnostic buffer: retain the latest microphone audio so wake misses
        # can be inspected later.
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
            # The browser adds a 700 ms mute tail after playback.
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
            ):
                return
            self.conversation_until = time.monotonic() + FOLLOWUP_S
            log.info("turn %d: post-interruption follow-up enabled for %.0fs", turn, FOLLOWUP_S)
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
            log.exception("Error while interrupting turn %s", current_turn)

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
            raise RuntimeError("Audio WebSocket is unavailable")
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
            raise HermesStreamTurnError("Hermes stream unavailable", started) from None

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

    # Wake detection runs in a dedicated thread; never block the event loop.
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
                log.info("Speech candidate rejected by the local Vosk gate")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
            # Vosk's constrained grammar can hallucinate its allowed phrase on
            # ordinary speech. Require an independent unconstrained local ASR
            # prefix before opening any paid realtime connection.
            try:
                local_text = transcribe(pcm, model=STT_GATE or STT_MODEL)
            except Exception:
                log.exception("Local wake confirmation failed")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
            if not WAKE_RE.match(local_text):
                log.info("Vosk candidate rejected by local ASR confirmation: %r", local_text[:60])
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
            log.info("Wake provider=whisper: energy VAD with a local Vosk gate")
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
                        log.exception("Wake engine error")
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
        # Evaluate the follow-up window when speech starts, not after STT; local
        # transcription can take several seconds on modest hardware.
        followup_at_start = time.monotonic() < self.conversation_until and not self.awaiting_playback
        await self.set_state("waking", turn=turn)
        await self._clear_partial(turn)
        # Preserve pre-roll: the wake phrase may have been spoken before the trigger
        # is confirmed, so include recent audio from the rolling buffer.
        prelude = bytes(initial_pcm)
        if not prelude and WAKE_PROVIDER == "whisper":
            with self.recent_lock:
                prelude = bytes(self.recent[-int(1.5 * SAMPLE_RATE * 2):])
        # Apply the pause only after a response, to avoid reacting to TTS echo.
        if time.time() - self.last_tts < 3.0:
            await asyncio.sleep(0.25)
        await self.set_state("recording", turn=turn)
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
        try:
            pcm = await self._record_utterance(prelude, realtime=realtime)
        except Exception:
            if realtime is not None:
                await realtime.close()
            log.exception("Recording error")
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
            log.info("turn %d: diagnostic audio saved to %s", turn, saved.name)
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
            log.exception("STT failed")
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
                log.info("turn %d rejected (outside the conversation window): %r", turn, text[:80])
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                await self._clear_partial(turn)
                await self.set_state("listening", note=localized("That was not for me.", "Non era per me."))
                return
            if not cmd.strip():
                log.info("turn %d: wake phrase detected without a command", turn)
                await self._clear_partial(turn)
                await self._speak(localized("Yes?", "Sì?"), turn)
                await self.set_state("listening", turn=turn)
                return
            text = cmd.strip()
            self.conversation_until = 0.0  # Allow the next turn only after playback completes
            log.info("%s command: %r", "follow-up" if followup_at_start else "wake", text[:120])
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
                log.warning("turn %d: %s", turn, exc)
                reply = localized("I could not complete the response.", "Non riesco a completare la risposta.")
                tts_failed = False
                stream_had_audio = exc.had_audio
            except Exception:
                log.exception("Hermes stream failed")
                reply = localized("I could not complete the response.", "Non riesco a completare la risposta.")
                tts_failed = False
            await self.send_json({"type": "reply", "text": reply, "turn": turn})
            if stream_had_audio:
                playback = await self._wait_for_playback(turn)
                if playback == "completed":
                    await self.set_state("listening", turn=turn)
                elif playback == "failed":
                    await self.set_state("listening", turn=turn, error=localized("playback failed", "riproduzione non riuscita"))
                else:
                    await self.set_state("listening", turn=turn, error=localized("playback timed out", "timeout di riproduzione"))
            elif tts_failed:
                await self.set_state("listening", turn=turn, error=localized("TTS unavailable", "TTS non disponibile"))
            elif reply:
                # A successful stream with no usable MP3 is safe to handle with
                # the old single-response TTS path: Hermes is not called again.
                await self._speak(reply, turn)
                if self.awaiting_playback:
                    playback = await self._wait_for_playback(turn)
                    if playback == "completed":
                        await self.set_state("listening", turn=turn)
                    elif playback == "failed":
                        await self.set_state("listening", turn=turn, error=localized("playback failed", "riproduzione non riuscita"))
                    else:
                        await self.set_state("listening", turn=turn, error=localized("playback timed out", "timeout di riproduzione"))
                else:
                    await self.set_state("listening", turn=turn)
            else:
                await self.set_state("listening", turn=turn)
            log.info("turn %d completed in %.1fs", turn, time.time() - t_turn)
            return

        # Non-streaming backends retain the legacy single MP3 protocol.
        try:
            reply = await self._ask_hermes(text)
        except HermesContinuationError:
            log.warning("turn %d: Hermes continuation unavailable", turn)
            reply = localized("I could not continue this conversation.", "Non riesco a continuare questa conversazione.")
        except Exception as exc:
            log.exception("Agent failed")
            reply = localized(f"Could not contact the agent: {exc}", f"Non sono riuscito a contattare l'agente: {exc}")
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
                await self.set_state("listening", turn=turn, error=localized("playback failed", "riproduzione non riuscita"))
            else:
                await self.set_state("listening", turn=turn, error=localized("playback timed out", "timeout di riproduzione"))
        else:
            await self.set_state("listening", turn=turn)
        log.info("turn %d completed in %.1fs", turn, time.time() - t_turn)

    async def _speak(self, text: str, turn: int):
        """Synthesize speech, send audio to the client, then return to listening."""
        await self.set_state("speaking", turn=turn)
        try:
            audio = await asyncio.to_thread(tts, text)
        except Exception as exc:
            log.exception("TTS failed")
            await self.set_state("listening", error=f"tts: {exc}")
            return
        if audio:
            self._begin_playback(turn)
            await self.send_json({"type": "audio", "fmt": "mp3", "bytes": len(audio), "turn": turn})
            if self.ws is None:
                await self.set_state("listening", error="Audio WebSocket is unavailable")
                self.awaiting_playback = False
                return
            await self.ws.send_bytes(audio)
            self.last_tts = time.time()

    async def _record_utterance(self, prelude: bytes = b"", realtime=None):
        """Collect PCM while speech is detected, closing after SILENCE_END_S of silence.

        Timings use received samples rather than wall clock so behavior is stable
        for both live streams and tests. `prelude` contains audio captured before
        the trigger, preserving the wake phrase spoken before the system reacts.
        """
        collected = bytearray(prelude)
        voiced_s = 0.0
        idle_s = 0.0
        started = False
        last_data = time.time()
        t_end_max = time.time() + MAX_UTTERANCE_S
        if realtime is not None and prelude:
            # The local wake detector fired before recording began. Sending this
            # confirmed pre-roll after opening the provider preserves the wake
            # gate without ever keeping an idle provider connection alive.
            await realtime.send_audio(prelude)
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
                # Poll non-blockingly: on the service's Python runtime a
                # repeated Queue.get in asyncio's worker pool can retain the
                # queue condition between calls and stall the recorder.
                chunk = self.recv_queue.get_nowait()
            except queue.Empty:
                # Close an interrupted stream instead of leaving the recorder waiting.
                if time.time() - last_data > 3.0:
                    break
                await asyncio.sleep(0.02)
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
                await realtime.send_audio(chunk)
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
            # Keep adapting the noise floor while recording; otherwise a noisy environment
            # can leave the detection threshold too low.
            if rms < threshold:
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 1.0)
                self.last_silent = time.time()
            elif time.time() - self.last_silent > 3.0:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                self.last_silent = time.time()
                threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
                log.info("VAD: noise floor adapted to ambient sound -> threshold %.0f", threshold)
            if rms >= threshold:
                if not started:
                    log.info("VAD: speech detected (rms=%.0f threshold=%.0f)", rms, threshold)
                started = True
                idle_s = 0.0
                voiced_s += dt
            elif started:
                idle_s += dt
                if idle_s >= SILENCE_END_S and voiced_s >= MIN_SPEECH_S:
                    break
            else:
                idle_s += dt
                if idle_s >= IDLE_ABORT_S:
                    log.info("VAD: aborted; no speech detected within %.1fs", IDLE_ABORT_S)
                    return None
            if idle_s >= SILENCE_END_S and voiced_s >= MIN_SPEECH_S:
                break
        pcm = np.frombuffer(bytes(collected), dtype=np.int16)
        log.info("VAD: collected %.2fs (speech %.2fs, started=%s)", len(pcm) / SAMPLE_RATE, voiced_s, started)
        # Do not trim the beginning: pre-roll contains the wake phrase, which may be
        # quieter than the command. An earlier energy trim removed the wake and caused
        # the remaining command to be rejected. Whisper already has VAD for leading silence.
        start = 0
        thr = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT) * 0.5
        win = 1600
        end = len(pcm)
        for i in range(len(pcm) - win, start, -win):
            if float(np.sqrt(np.mean(pcm[i:i + win].astype(np.float32) ** 2))) >= thr:
                end = min(len(pcm), i + 2 * win)
                break
        log.info("VAD: utterance trimmed from %.2fs to %.2fs",
                 len(pcm) / SAMPLE_RATE, (end - start) / SAMPLE_RATE)
        return pcm[start:end]


app = FastAPI(title="Lari")
_loop: asyncio.AbstractEventLoop | None = None
_sessions: set[Session] = set()


@app.get("/")
async def root():
    return Response(content="Lari: open /<token>/", media_type="text/plain")


@app.get("/{token}/debug/last.wav")
async def debug_last(token: str):
    """Return the latest ~12 seconds of microphone audio for offline calibration."""
    if not TOKEN or token != TOKEN:
        return Response(content="Invalid token", status_code=403)
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
    return Response(content="No active session", status_code=404)


@app.get("/{token}/")
async def page(token: str):
    if not TOKEN or token != TOKEN:
        return Response(content="Invalid token", status_code=403)
    return FileResponse(
        BASE_DIR / "static" / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, must-revalidate"},   # Prevent stale HTML from being cached
    )


@app.get("/{token}/assets/{asset_name}")
async def asset(token: str, asset_name: str):
    if not TOKEN or token != TOKEN:
        return Response(content="Forbidden", status_code=403)
    if asset_name != "lare-concept.svg":
        return Response(content="Not found", status_code=404)
    path = BASE_DIR / "static" / "assets" / asset_name
    if not path.is_file():
        return Response(content="Not found", status_code=404)
    return FileResponse(
        path,
        media_type="image/svg+xml",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@app.websocket("/{token}/ws")
async def ws_endpoint(token: str, websocket: WebSocket):
    if not TOKEN or token != TOKEN:
        await websocket.close(code=4403)
        return
    await websocket.accept()
    log.info("Client connected from %s", websocket.client)
    session = Session(websocket, _make_sender(websocket))
    _sessions.add(session)
    worker = threading.Thread(target=session.wake_worker, daemon=True, name="wake-worker")
    worker.start()
    await session.send_json({
        "type": "state", "state": "listening",
        "phrase": WAKE_PHRASE, "provider": WAKE_PROVIDER,
        "language": LANGUAGE, "device": "Lare", "project": "Lari",
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
                        log.info("turn %d: follow-up enabled for %.0fs", session.turn, FOLLOWUP_S)
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
        # Diagnostics: retain recent microphone audio so a user test can be
        # inspected offline.
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
            log.exception("Calibration recording failed")
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
