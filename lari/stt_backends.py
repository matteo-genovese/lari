"""Cloud STT backends for desk-buddy: A) ElevenLabs Scribe v2, B) Groq Whisper
Large V3 Turbo, C) OpenAI Whisper API (the cloud path of OpenWhispr).

Privacy: these send the utterance audio to the provider. Keys come from the
environment (never logged): ELEVENLABS_API_KEY, GROQ_API_KEY, OPENAI_API_KEY.

Interface: transcribe(pcm_int16_16k_mono, backend) -> text.
"""
from __future__ import annotations

import io
import asyncio
import base64
import datetime as _datetime
import fcntl
import json
import logging
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from urllib.parse import urlencode
import wave
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import websockets

from . import wake_config

log = logging.getLogger(__name__)

BACKENDS = ("elevenlabs", "groq", "openai")
REALTIME_BACKEND = "elevenlabs_realtime"
KEY_ENV = {
    "elevenlabs": "ELEVENLABS_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}
# Wake-derived ASR bias terms: keyterms, style prompt and the realtime URL all
# derive from the configured wake phrase plus the optional BUDDY_STT_KEYTERMS
# vocabulary (names and places the ASR misrenders).
_WAKE = wake_config.from_env()
# ElevenLabs batch keyterms (<= 5 words each, <= 1000 total). Scribe v2 with
# keyterms costs +20% over the base rate.
KEYTERMS = list(_WAKE.batch_keyterms)
# Realtime accepts at most 20 characters per keyterm. These are encoded as
# repeated query parameters below, as required by the WebSocket API.
REALTIME_KEYTERMS = _WAKE.realtime_keyterms
# Groq/OpenAI accept only a style prompt (max 224 tokens for Groq).
STYLE_PROMPT = _WAKE.style_prompt

REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + urlencode(
    [
        ("model_id", "scribe_v2_realtime"),
        ("audio_format", "pcm_16000"),
        ("language_code", "it"),
        ("commit_strategy", "manual"),
        *[("keyterms", term) for term in REALTIME_KEYTERMS],
    ]
)
REALTIME_DAILY_SECONDS = 600.0
REALTIME_SESSION_TIMEOUT_S = 5.0
REALTIME_COMMIT_TIMEOUT_S = 10.0
_PCM_BYTES_PER_SECOND = 16000 * 2


class RealtimeUnavailable(RuntimeError):
    """Realtime STT cannot be used for this utterance."""


class _CorruptBudgetLedger(RuntimeError):
    """An existing realtime usage ledger is not safe to interpret."""


class DailyAudioBudget:
    """Atomic local ledger for realtime audio sent per day.

    This is an explicit local cap on audio sent to the realtime STT websocket.
    It is not an overall provider billing cap; the Desk Buddy realtime caller
    falls back locally rather than invoking paid batch transcription.
    """

    def __init__(self, path: str | Path | None = None,
                 daily_seconds: float | None = None):
        default_path = Path(__file__).resolve().parent.parent / ".realtime_stt_usage.json"
        self.path = Path(path or os.environ.get("BUDDY_REALTIME_USAGE_FILE", default_path))
        self.daily_seconds = float(
            os.environ.get("BUDDY_REALTIME_DAILY_SECONDS", REALTIME_DAILY_SECONDS)
            if daily_seconds is None else daily_seconds
        )
        self._lock = threading.Lock()

    @contextmanager
    def _process_lock(self, exclusive: bool = True):
        """Serialize ledger read/modify/write operations across processes."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _today() -> str:
        return _datetime.date.today().isoformat()

    def _read(self) -> tuple[str, float]:
        today = self._today()
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return today, 0.0
        except OSError as exc:
            raise _CorruptBudgetLedger("realtime usage ledger unreadable") from exc

        try:
            data = json.loads(raw)
            date = data["date"]
            seconds = data["seconds"]
            if (
                not isinstance(data, dict)
                or not isinstance(date, str)
                or not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(seconds)
                or seconds < 0.0
            ):
                raise ValueError
            _datetime.date.fromisoformat(date)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise _CorruptBudgetLedger("realtime usage ledger malformed") from exc

        if date == today:
            return date, float(seconds)
        return today, 0.0

    def _write(self, date: str, seconds: float) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"date": date, "seconds": round(seconds, 6)}, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def remaining(self) -> float:
        with self._lock:
            with self._process_lock(exclusive=False):
                try:
                    _, used = self._read()
                except _CorruptBudgetLedger:
                    return 0.0
                return max(0.0, self.daily_seconds - used)

    def reserve(self, seconds: float) -> bool:
        seconds = max(0.0, float(seconds))
        if seconds == 0.0:
            return True
        with self._lock:
            with self._process_lock():
                try:
                    date, used = self._read()
                except _CorruptBudgetLedger:
                    return False
                if used + seconds > self.daily_seconds + 1e-9:
                    return False
                self._write(date, used + seconds)
                return True

    def refund(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if seconds == 0.0:
            return
        with self._lock:
            with self._process_lock():
                try:
                    date, used = self._read()
                except _CorruptBudgetLedger:
                    # Never replace a corrupt ledger with a smaller value.
                    return
                self._write(date, max(0.0, used - seconds))


_daily_budget = DailyAudioBudget()


def realtime_daily_budget() -> DailyAudioBudget:
    """Return the process-wide ledger shared by all satellite sessions."""
    return _daily_budget


class RealtimeScribe:
    """One bounded ElevenLabs realtime session; never persistent or browser-side."""

    def __init__(self, websocket, on_partial: Callable[[str], Awaitable[None]] | None,
                 budget: DailyAudioBudget):
        self.websocket = websocket
        self.on_partial = on_partial
        self.budget = budget
        self._reader_task: asyncio.Task | None = None
        self._started = asyncio.Event()
        self._committed = asyncio.Event()
        self._committed_segments: list[str] = []
        self._error: BaseException | None = None
        self._closed = False
        self._disabled = False
        self._commit_sent = False

    @classmethod
    async def connect(cls, on_partial=None, budget: DailyAudioBudget | None = None):
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise RealtimeUnavailable("ELEVENLABS_API_KEY non impostata")
        budget = budget or realtime_daily_budget()
        if budget.remaining() <= 0.0:
            raise RealtimeUnavailable("daily realtime audio cap reached")
        try:
            websocket = await websockets.connect(
                REALTIME_URL,
                additional_headers={"xi-api-key": key},
            )
        except Exception as exc:
            raise RealtimeUnavailable("realtime websocket unavailable") from exc
        session = cls(websocket, on_partial, budget)
        session._reader_task = asyncio.create_task(session._read_events())
        try:
            await asyncio.wait_for(session._started.wait(), REALTIME_SESSION_TIMEOUT_S)
            if session._error:
                raise RealtimeUnavailable("realtime provider error")
        except RealtimeUnavailable:
            await session.close()
            raise
        except asyncio.TimeoutError as exc:
            await session.close()
            raise RealtimeUnavailable("realtime session start timeout") from exc
        except Exception as exc:
            await session.close()
            raise RealtimeUnavailable("realtime session start failed") from exc
        return session

    @property
    def failed(self) -> bool:
        return self._error is not None or self._disabled or self._closed

    async def _read_events(self):
        try:
            while not self._closed:
                raw = await self.websocket.recv()
                if raw is None:
                    if not self._closed:
                        self._error = RealtimeUnavailable("realtime websocket closed")
                    return
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                event = json.loads(raw)
                kind = event.get("message_type")
                if kind == "session_started":
                    self._started.set()
                elif kind == "partial_transcript":
                    text = (event.get("text") or "").strip()
                    if text and self.on_partial:
                        await self.on_partial(text)
                elif kind == "committed_transcript":
                    text = (event.get("text") or "").strip()
                    self._committed_segments.append(text)
                    self._committed.set()
                elif kind in {
                    "error", "auth_error", "quota_exceeded", "transcriber_error",
                    "input_error", "invalid_request", "commit_throttled",
                    "unaccepted_terms", "rate_limited", "queue_overflow",
                    "resource_exhausted", "session_time_limit_exceeded",
                    "chunk_size_exceeded", "insufficient_audio_activity",
                }:
                    self._error = RealtimeUnavailable("realtime provider error")
                    self._started.set()
                    self._committed.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._error = RealtimeUnavailable("realtime websocket failure")
            self._started.set()
            self._committed.set()

    async def send_audio(self, pcm: bytes) -> bool:
        """Send speech PCM and reserve exactly its duration in the local ledger."""
        if self.failed or len(pcm) < 2:
            return False
        seconds = len(pcm) / _PCM_BYTES_PER_SECOND
        try:
            allowed = self.budget.reserve(seconds)
        except Exception:
            self._error = RealtimeUnavailable("realtime usage ledger unavailable")
            return False
        if not allowed:
            self._disabled = True
            return False
        try:
            await self.websocket.send(json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode("ascii"),
                "commit": False,
                "sample_rate": 16000,
            }))
            return True
        except Exception:
            self.budget.refund(seconds)
            self._error = RealtimeUnavailable("realtime websocket send failure")
            return False

    async def finish(self) -> str:
        """Commit exactly once and return all committed segments for the turn."""
        try:
            if self.failed:
                raise RealtimeUnavailable("realtime session failed")
            if self._commit_sent:
                raise RealtimeUnavailable("realtime commit already sent")
            # Automatic commits can already be waiting in the websocket reader.
            # Let that task consume them before establishing the count that the
            # explicit final commit must advance.
            await asyncio.sleep(0)
            committed_before_finish = len(self._committed_segments)
            self._committed.clear()
            self._commit_sent = True
            await self.websocket.send(json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": 16000,
            }))
            while len(self._committed_segments) <= committed_before_finish:
                await asyncio.wait_for(self._committed.wait(), REALTIME_COMMIT_TIMEOUT_S)
                self._committed.clear()
            if self._error:
                raise self._error
            text = " ".join(segment for segment in self._committed_segments if segment).strip()
            if not text:
                raise RealtimeUnavailable("realtime committed transcript empty")
            return text
        except RealtimeUnavailable:
            raise
        except Exception as exc:
            raise RealtimeUnavailable("realtime commit failed") from exc
        finally:
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self.websocket.close()
        except Exception:
            pass
        if self._reader_task and self._reader_task is not asyncio.current_task():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass


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


def transcribe(pcm: np.ndarray, backend: str) -> str:
    pcm = np.asarray(pcm, dtype=np.int16)
    if backend not in BACKENDS:
        raise ValueError(f"backend STT cloud sconosciuto: {backend}")
    key_env = KEY_ENV[backend]
    key = os.environ.get(key_env, "").strip()
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
            data_tuples=[("model", os.environ.get("BUDDY_GROQ_MODEL", "whisper-large-v3")),
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
