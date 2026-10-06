"""Public satellite WebSocket protocol (JSON metadata + binary audio).

Inbound: PCM16 little-endian, 16 kHz mono binary frames; JSON ping,
playback_done (turn, status=completed|failed), interrupt (turn), ptt (phase=down|up), and diag
(ctx, rate, mic, frames, vis, raw). Unknown/malformed JSON is ignored.
Outbound: the fifteen constructors below. audio_chunk/audio metadata precedes
one binary MP3 frame; audio_end ends the segmented reading. Turn identifiers
scope playback and interruption. Optional fields are omitted, never invented.
States describe the interaction, independently of STT/SSE/TTS internals.
See README for the wire table; this module is the sole message factory.
"""
from __future__ import annotations

import json
from typing import Literal

State = Literal["idle", "listening", "waking", "recording", "transcribing",
                "thinking", "speaking", "error"]
STATES = frozenset({"idle", "listening", "waking", "recording",
                    "transcribing", "thinking", "speaking", "error"})
_MISSING = object()


def _message(kind: str, **fields) -> dict:
    return {"type": kind, **{key: value for key, value in fields.items()
                            if value is not _MISSING}}


def parse_input(raw: str) -> dict | None:
    """Recognize browser controls; transport disconnects belong to ASGI."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("type") not in (
            "ping", "playback_done", "interrupt", "ptt", "diag"):
        return None
    if data["type"] == "ptt" and data.get("phase") not in ("down", "up"):
        return None
    return data


def state(*, state: State, turn=_MISSING, followup=_MISSING, phrase=_MISSING,
          provider=_MISSING, voice=_MISSING, sensitivity=_MISSING,
          confirm_frames=_MISSING, interrupted=_MISSING, note=_MISSING,
          error=_MISSING, manual=_MISSING) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown semantic state: {state}")
    return _message(
        "state", state=state, turn=turn, followup=followup, phrase=phrase,
        provider=provider, voice=voice, sensitivity=sensitivity,
        confirm_frames=confirm_frames, interrupted=interrupted, note=note, error=error, manual=manual,
    )


def partial_transcript(*, text, turn) -> dict:
    return _message("partial_transcript", text=text, turn=turn)


def transcript(*, text, turn, command=_MISSING) -> dict:
    return _message("transcript", text=text, turn=turn, command=command)


def reply(*, text, turn) -> dict:
    return _message("reply", text=text, turn=turn)


def stt_status(*, mode, turn) -> dict:
    return _message("stt_status", mode=mode, turn=turn)


def audio_start(*, turn) -> dict:
    return _message("audio_start", turn=turn)


def audio_chunk(*, turn, seq) -> dict:
    return _message("audio_chunk", turn=turn, seq=seq)


def audio_end(*, turn) -> dict:
    return _message("audio_end", turn=turn)


def audio(*, fmt, bytes, turn) -> dict:
    return _message("audio", fmt=fmt, bytes=bytes, turn=turn)


def followup(*, seconds, interrupted=_MISSING) -> dict:
    return _message("followup", seconds=seconds, interrupted=interrupted)


def interrupt_ack(*, turn, status, echo_tail_ms) -> dict:
    return _message("interrupt_ack", turn=turn, status=status, echo_tail_ms=echo_tail_ms)


def interrupt_rejected(*, turn) -> dict:
    return _message("interrupt_rejected", turn=turn)


def pong(*, state) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown semantic state: {state}")
    return _message("pong", state=state)


def approval(*, turn, approval) -> dict:
    return _message("approval", turn=turn, approval=approval)


def fatal(*, error) -> dict:
    return _message("fatal", error=error)
