"""Hermes HTTP/SSE transport and CLI fallback for unavailable connections.

Empty model/provider settings use the gateway's configured default model/provider.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from functools import partial

from ..config import Settings
from .events import (
    HermesApprovalRequest, HermesReply, HermesSSEState, HermesTextDelta,
    parse_hermes_lines,
)

log = logging.getLogger("lari")


def _model_overrides(settings: Settings) -> dict:
    """Omit empty values so the gateway uses its configured default model/provider."""
    return {key: value for key, value in {
        "model": settings.hermes_model,
        "provider": settings.hermes_provider,
    }.items() if value}


class HermesContinuationError(RuntimeError):
    """A continued Hermes turn failed without switching transcripts."""


class HermesStreamTurnError(RuntimeError):
    """A streamed turn failed; the turn must never be submitted again."""

    def __init__(self, message: str, had_audio: bool = False):
        super().__init__(message)
        self.had_audio = had_audio


async def ask_hermes(
    text: str,
    settings: Settings,
    session_id: str | None = None,
    *,
    cli_fallback: Callable[[str], str] | None = None,
) -> HermesReply:
    """Send the utterance to the agent, continuing ``session_id`` when present.

    Empty model/provider settings use the gateway's configured defaults.
    The return value remains string-compatible for existing callers and carries the
    response's ``X-Hermes-Session-Id`` as ``.session_id``.
    """
    import httpx

    payload = {
        **_model_overrides(settings),
        "model_options": {"reasoning": {"enabled": False}},
        "messages": [
            {"role": "system", "content": settings.voice_system},
            {"role": "user", "content": text},
        ],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if settings.hermes_key:
        headers["Authorization"] = f"Bearer {settings.hermes_key}"
    headers["X-Hermes-Session-Key"] = settings.session_key
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=settings.agent_timeout_s) as client:
            r = await client.post(f"{settings.hermes_api}/v1/chat/completions", json=payload, headers=headers)
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
        return HermesReply(await loop.run_in_executor(None, cli_fallback or partial(_ask_cli, settings=settings), text))


async def stream_hermes(
    text: str,
    settings: Settings,
    session_id: str | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    on_approval: Callable[[dict], Awaitable[None]] | None = None,
) -> HermesReply:
    """Stream one opt-in Hermes turn over SSE.

    Empty model/provider settings use the gateway's configured defaults.
    Only ``delta.content`` is speech. Reasoning, tool/status events, and approval
    metadata are kept out of the returned text. The response session/run ids are
    published only after the terminal ``[DONE]`` frame, so callers can retain
    their previous transcript on any failed or incomplete stream.

    An initial turn may fall back to CLI only when connecting fails before the
    request can be sent. Once the stream opens, never submit the turn again.
    """
    import httpx

    payload = {
        **_model_overrides(settings),
        "model_options": {"reasoning": {"enabled": False}},
        "messages": [
            {"role": "system", "content": settings.voice_system},
            {"role": "user", "content": text},
        ],
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Hermes-Session-Key": settings.session_key,
    }
    if settings.hermes_key:
        headers["Authorization"] = f"Bearer {settings.hermes_key}"
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id

    state = HermesSSEState()
    response_session_id: str | None = None
    stream_opened = False

    async def dispatch(events):
        for event in events:
            if isinstance(event, HermesTextDelta) and on_delta is not None:
                await on_delta(event.text)
            elif isinstance(event, HermesApprovalRequest) and on_approval is not None:
                await on_approval(event.payload)

    try:
        async with httpx.AsyncClient(timeout=settings.agent_timeout_s) as client:
            async with client.stream(
                "POST",
                f"{settings.hermes_api}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                stream_opened = True
                response.raise_for_status()
                response_session_id = response.headers.get("X-Hermes-Session-Id")

                async for line in response.aiter_lines():
                    state, events = parse_hermes_lines((line,), state, final=False)
                    await dispatch(events)

                # Flush the final frame before checking terminal markers, so its
                # callbacks also run when the stream ultimately proves incomplete.
                state, events = parse_hermes_lines(("",), state, final=False)
                await dispatch(events)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        if stream_opened or session_id:
            # Hermes may already have executed tools/approvals, or this turn
            # needs the existing transcript. Never resubmit it via CLI.
            raise
        log.warning(
            "connessione API Hermes non stabilita (%s), fallback hermes chat -q",
            type(exc).__name__,
        )
        loop = asyncio.get_running_loop()
        reply = HermesReply(await loop.run_in_executor(None, partial(_ask_cli, settings=settings), text))
        if on_delta is not None:
            await on_delta(str(reply))
        return reply

    state, events = parse_hermes_lines((), state)
    completed = events[-1]
    return HermesReply(completed.text, response_session_id, completed.run_id)


def _ask_cli(text: str, settings: Settings) -> str:
    """Fallback without the API server; empty model/provider settings use the
    configured Hermes profile defaults, just as the gateway does.

    stdin=DEVNULL: an approval prompt must
    fail immediately instead of hanging while waiting for a tty that does not exist."""
    import subprocess as sp

    argv = [str(settings.hermes_root / "venv/bin/hermes"), "chat", "-q", text, "-Q"]
    if settings.hermes_model:
        argv.extend(["-m", settings.hermes_model])
    if settings.hermes_provider:
        argv.extend(["--provider", settings.hermes_provider])
    argv.extend(["--reasoning", "none", "--continue", settings.session_key,
                 "--create-if-missing"])
    t0 = time.time()
    try:
        p = sp.run(
            argv,
            capture_output=True, text=True, timeout=settings.cli_timeout_s, cwd=str(settings.hermes_root),
            stdin=sp.DEVNULL,
        )
        out = (p.stdout or "").strip()
        dt = time.time() - t0
        log.info("fallback CLI: %.1fs exit=%s out=%r", dt, p.returncode, out[:120])
        return out or f"(nessuna risposta, exit={p.returncode}: {(p.stderr or '').strip()[:200]})"
    except sp.TimeoutExpired:
        log.error("fallback CLI: timeout dopo %.0fs", settings.cli_timeout_s)
        return "(il backend è lento: fai girare l'API server riavviando il gateway)"
    except Exception as exc:
        return f"(errore agente: {exc})"
