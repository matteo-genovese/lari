"""Lari bridge: satellite microphone -> local wake gate -> STT -> Hermes -> TTS.

The browser satellite captures mono 16 kHz PCM and streams it to the bridge
via a token-protected WebSocket. Keep the installed wake phrase and backend
settings independent of the device and project branding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from . import hermes
from .stt import providers as stt_backends  # TEMP-P5
from .wake import config as wake_config  # TEMP-P5
from .config import get_settings


# TEMP-P5: old test/public names; implementations live in their subsystems.
from .audio import SessionAudio, save_turn_audio
from .wake import detector as _detector, confirm as _confirm
from .stt import local as _local, vosk as _vosk, dispatch as _dispatch
from . import audio as _audio
from .wake.detector import (_wake_engine_builder, wake_command, resolve_command,
                            get_vosk, vosk_wake, resolve_vosk_command, make_engine)
from .wake.confirm import confirm_candidate
from .stt.local import get_stt, transcribe
from .stt.vosk import transcribe_vosk
from .stt.dispatch import (_transcribe_local_fallback, stt_transcribe,
                          _transcribe_realtime_or_batch,
                          transcribe_realtime_or_batch, decode_utterance)

_SETTINGS = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root; runtime paths never depend on cwd
HERMES_ROOT = _SETTINGS.hermes_root


# ─── configuration ──────────────────────────────────────────────────────────
TOKEN = _SETTINGS.token
PORT = _SETTINGS.port
HERMES_API = _SETTINGS.hermes_api
HERMES_KEY = _SETTINGS.hermes_key
SESSION_KEY = _SETTINGS.session_key
HERMES_PROVIDER = _SETTINGS.hermes_provider
HERMES_MODEL = _SETTINGS.hermes_model

from . import usage  # noqa: E402  (project module)
USAGE_LEDGER = usage.UsageLedger()

# Voice = short replies meant to be spoken. Without this, a bogus
# transcription makes the agent dig through logs (125 s in the worst measured case).
VOICE_SYSTEM = _SETTINGS.voice_system


HermesReply = hermes.HermesReply  # TEMP-P5: alias del refactor, da rimuovere
HermesStreamTurnError = hermes.HermesStreamTurnError  # TEMP-P5: alias del refactor, da rimuovere
HermesContinuationError = hermes.HermesContinuationError  # TEMP-P5: alias del refactor, da rimuovere


TTS_VOICE = _SETTINGS.tts_voice
# TEMP-P5: STT configuration re-exports used by existing callers/tests.
STT_MODEL = _SETTINGS.stt_model
STT_BACKEND = _SETTINGS.stt_backend
VOSK_MODEL_DIR = _SETTINGS.vosk_model_dir
STT_LANG = _SETTINGS.stt_lang  # "" = auto-detect
# Fast gate for the wake: on bad audio the large model takes 50-60 s and
# blocks everything. With a small model the rejection arrives in seconds.
# Empty ("") = a single pass with the large model.
# A tiny model used as veto also rejects genuine wake words.
# A single pass with small: fewer false negatives and no double transcription.
STT_GATE = _SETTINGS.stt_gate

SAMPLE_RATE = 16000
FRAME = 1280              # 80 ms @ 16 kHz, frame size recommended by openWakeWord
WAKE_PROVIDER = _SETTINGS.wake_provider
# whisper: any speech starts the recording, then the phrase is searched in the text.
# sherpa/openwakeword: dedicated hotword engine (English; misses the IT pronunciation).
# The wake phrase is one setting: command regex, Vosk grammar, junk cleanup and
# ASR keyterms all derive from it in wake/config.py.  The regex anchors the
# wake as an address (transcript start or after a sentence boundary) so
# background mentions ("ho parlato con ...") never trigger.
WAKE_CONFIG = _SETTINGS.wake_config
WAKE_RE = WAKE_CONFIG.command_re


WAKE_PHRASE = WAKE_CONFIG.phrase
# Second local gate on the wake candidate, before any provider connection.
# It may veto only a confident mismatch (see confirm_candidate): every doubt
# passes, so a real wake is never lost to an ASR mishearing.  Set
# LARI_WAKE_CONFIRM=0 to disable.
WAKE_CONFIRM = _SETTINGS.wake_confirm
# sherpa threshold = 0.05 + 0.4*sens; 0.5 -> 0.25 (upstream-recommended value)
WAKE_SENSITIVITY = _SETTINGS.wake_sensitivity
CONFIRM_FRAMES = _SETTINGS.confirm_frames
COOLDOWN_S = 2.0          # same constraint as Hermes between two wakes
AMBIENT_PAUSE_S = _SETTINGS.ambient_pause_s  # pause after speech not addressed to us
# The phone plays the reply through the same speaker the microphone uses:
# without this mute, the system ends up transcribing itself.
ECHO_MUTE_S = _SETTINGS.echo_mute_s
FOLLOWUP_S = _SETTINGS.followup_s  # after audio playback
PLAYBACK_ACK_TIMEOUT_S = _SETTINGS.playback_ack_timeout_s

# Streaming voice responses are deliberately bounded.  The queue is small so a
# slow browser/TTS provider applies back-pressure to SSE instead of allowing an
# unbounded response to accumulate in memory.
STREAM_TTS_QUEUE_MAX = _SETTINGS.stream_tts_queue_max
STREAM_TEXT_MAX_CHARS = _SETTINGS.stream_text_max_chars
STREAM_SENTENCE_MAX_CHARS = _SETTINGS.stream_sentence_max_chars

# TEMP-P5: audio configuration re-exports used by existing callers/tests.
# VAD: adaptive threshold. Minimum base threshold + multiple of the noise floor.
VAD_MIN_RMS = _SETTINGS.vad_min_rms
VAD_NOISE_MULT = _SETTINGS.vad_noise_mult
# More tolerant of pauses while a request is being phrased; the VAD still
# closes on silence, without waiting for the hard limit.
SILENCE_END_S = _SETTINGS.silence_end_s
# Only a parachute against continuous noise / a stuck VAD, not a target utterance length.
MAX_UTTERANCE_S = _SETTINGS.max_utterance_s
MIN_SPEECH_S = _SETTINGS.min_speech_s
IDLE_ABORT_S = _SETTINGS.idle_abort_s

AGENT_TIMEOUT_S = _SETTINGS.agent_timeout_s
CLI_TIMEOUT_S = _SETTINGS.cli_timeout_s

log = logging.getLogger("lari")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "server.log")],
)


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


def _hermes_settings():
    # TEMP-P5: preserve patches of the legacy server configuration until P5.
    return replace(
        _SETTINGS, hermes_root=HERMES_ROOT, hermes_api=HERMES_API,
        hermes_key=HERMES_KEY, session_key=SESSION_KEY,
        hermes_provider=HERMES_PROVIDER, hermes_model=HERMES_MODEL,
        voice_system=VOICE_SYSTEM, agent_timeout_s=AGENT_TIMEOUT_S,
        cli_timeout_s=CLI_TIMEOUT_S,
    )


# TEMP-P5: alias del refactor, da rimuovere
async def ask_hermes(text: str, session_id: str | None = None) -> HermesReply:
    return await hermes.ask_hermes(
        text, _hermes_settings(), session_id=session_id, cli_fallback=_ask_cli,
    )


# TEMP-P5: alias del refactor, da rimuovere
async def stream_hermes(
    text: str,
    session_id: str | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    on_approval: Callable[[dict], Awaitable[None]] | None = None,
) -> HermesReply:
    return await hermes.stream_hermes(
        text, _hermes_settings(), session_id=session_id,
        on_delta=on_delta, on_approval=on_approval,
    )


# TEMP-P5: alias del refactor, da rimuovere
def _ask_cli(text: str) -> str:
    return hermes._ask_cli(text, _hermes_settings())


# ─── wake engine ─────────────────────────────────────────────────────────────


class Session(SessionAudio):
    """State of one satellite connection (one device = one session)."""

    def __init__(self, ws: WebSocket, send_json):
        self.ws = ws
        self.send_json = send_json
        self.engine = None
        self.state = "listening"           # the worker starts listening immediately
        self.last_wake = 0.0
        self._init_audio()
        self.recv_queue: queue.Queue = queue.Queue(maxsize=400)
        self.stop = threading.Event()
        self.turn = 0
        # rotating calibration buffer: the last ~12 s of microphone, to see
        # what the detector really hears when it does not fire
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

        # HermesReply publishes its session id only after completion. Do not touch
        # the WebSocket transcript id before this point, including when TTS
        # fails after the Hermes turn itself completed.
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
        if self.echo_muted(time.time(), ECHO_MUTE_S):
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
                thr = self.update_noise_floor(rms)
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
                if hit and self.echo_muted(now, ECHO_MUTE_S):
                    continue
                if hit and now - self.last_wake >= COOLDOWN_S and rms > 200:
                    prelude = self.pre_roll()
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
            prelude = self.pre_roll()
        # pause only if a reply was just played (TTS echo)
        if self.echo_muted(time.time(), 3.0):
            await asyncio.sleep(0.25)
        await self.set_state("recording", turn=turn, followup=followup_at_start)
        realtime, realtime_start_failed = await _dispatch.open_stream(
            on_partial=lambda text: self._send_partial(text, turn),
        )
        if realtime_start_failed:
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
        try:
            text = await _dispatch.transcribe_turn(
                pcm, realtime, turn, followup_at_start,
                start_failed=realtime_start_failed,
            )
        except Exception as exc:
            log.exception("STT fallito")
            await self._clear_partial(turn)
            await self.set_state("listening", error=f"stt: {exc}")
            return
        if text is None:
            await self._clear_partial(turn)
            await self.set_state("listening")
            return
        if WAKE_PROVIDER == "whisper":
            cmd = _dispatch.command_for_turn(text, followup_at_start, local_wake_confirmed)
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
                with wave.open(fh, "wb") as w:
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
    rate = _SETTINGS.usage_eur_per_min
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
            session.save_calibration()
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
        log.warning("LARI_TOKEN non imposto: il server rifiuta tutto")


# TEMP-P5: forward legacy monkeypatches to the owning modules. No subsystem
# imports server; remove this compatibility bridge with the old re-exports.
import types as _types
class _CompatibilityModule(_types.ModuleType):
    def __setattr__(self, name, value):
        for module in (_detector, _confirm, _local, _vosk, _dispatch, _audio):
            if name in _COMPAT_NAMES and hasattr(module, name):
                setattr(module, name, value)
        super().__setattr__(name, value)

_COMPAT_NAMES = {
    'BASE_DIR',
    'CONFIRM_FRAMES',
    'IDLE_ABORT_S',
    'MAX_UTTERANCE_S',
    'MIN_SPEECH_S',
    'SILENCE_END_S',
    'STT_BACKEND',
    'STT_LANG',
    'STT_MODEL',
    'VAD_MIN_RMS',
    'VAD_NOISE_MULT',
    'VOSK_MODEL_DIR',
    'WAKE_CONFIG',
    'WAKE_PHRASE',
    'WAKE_PROVIDER',
    'WAKE_SENSITIVITY',
    '_transcribe_local_fallback',
    '_transcribe_realtime_or_batch',
    '_wake_engine_builder',
    'confirm_candidate',
    'decode_utterance',
    'get_stt',
    'get_vosk',
    'make_engine',
    'resolve_command',
    'resolve_vosk_command',
    'save_turn_audio',
    'stt_transcribe',
    'transcribe',
    'transcribe_realtime_or_batch',
    'transcribe_vosk',
    'vosk_wake',
    'wake_command',
}

sys.modules[__name__].__class__ = _CompatibilityModule


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
