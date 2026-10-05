"""Lari bridge: satellite microphone -> local wake gate -> STT -> Hermes -> TTS.

The browser satellite captures mono 16 kHz PCM and streams it to the bridge
via a token-protected WebSocket. Keep the installed wake phrase and backend
settings independent of the device and project branding.
"""

from __future__ import annotations

import asyncio
import json
import logging
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




# TEMP-P5: public Session/voice names and monkeypatch compatibility.
from . import session as _session, tts as _tts
from .wake import runtime as _wake_runtime
from .session import Session
from .tts import tts, SpeakableSentenceBuffer, TTSUnavailable
_session.ask_hermes = ask_hermes
_session.stream_hermes = stream_hermes
_session.USAGE_LEDGER = USAGE_LEDGER


app = FastAPI(title="Lari")
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
    session.worker = worker
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
                await session.on_audio(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "ping":
                    await session.send_json({"type": "pong", "state": session.state})
                elif data.get("type") == "playback_done":
                    status = data.get("status", "completed")
                    await session.playback_completed(data.get("turn"), status=status)
                elif data.get("type") == "interrupt":
                    accepted = await session.interrupt(data.get("turn"))
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
        await session.disconnect()
        # calibration: save the last ~12 s of microphone so it can be
        # re-analyzed offline after each user test
        try:
            session.save_calibration()
        except Exception:
            log.exception("salvataggio calibrazione fallito")
        log.info("sessione chiusa (turni: %d, ultimo stato: %s)", session.turn, session.state)
        _sessions.discard(session)
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
    if not TOKEN:
        log.warning("LARI_TOKEN non imposto: il server rifiuta tutto")


# TEMP-P5: forward legacy monkeypatches to the owning modules. No subsystem
# imports server; remove this compatibility bridge with the old re-exports.
import types as _types
class _CompatibilityModule(_types.ModuleType):
    def __setattr__(self, name, value):
        for module in (_detector, _confirm, _local, _vosk, _dispatch, _audio, _session, _tts, _wake_runtime):
            if name in _COMPAT_NAMES and hasattr(module, name):
                setattr(module, name, value)
        super().__setattr__(name, value)

_COMPAT_NAMES = {
    'tts', 'stream_hermes', 'ask_hermes', 'USAGE_LEDGER',
    'TTS_VOICE', 'STREAM_TTS_QUEUE_MAX', 'STREAM_TEXT_MAX_CHARS',
    'STREAM_SENTENCE_MAX_CHARS', 'ECHO_MUTE_S', 'FOLLOWUP_S',
    'PLAYBACK_ACK_TIMEOUT_S', 'WAKE_CONFIRM', 'AMBIENT_PAUSE_S',
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
