"""HTTP/PWA routes and token-protected WebSocket input adapter."""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .config import get_settings
from .session import Session
from .usage import UsageLedger
from . import protocol

_SETTINGS = get_settings()
USAGE_LEDGER = UsageLedger(settings=_SETTINGS)
BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN = _SETTINGS.token
SAMPLE_RATE = 16000
log = logging.getLogger("lari")

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
            dict(src="%s/icon-192.png" % icons, sizes="192x192", type="image/png"),
            dict(src="%s/icon-512.png" % icons, sizes="512x512", type="image/png"),
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
    session = Session(websocket, _make_sender(websocket), settings=_SETTINGS, usage_ledger=USAGE_LEDGER)
    _sessions.add(session)
    try:
        await session.start()
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                await session.on_audio(msg["bytes"])
            elif "text" in msg and msg["text"]:
                data = protocol.parse_input(msg["text"])
                if data is None:
                    continue
                if data["type"] == "ping":
                    await session.send_json(protocol.pong(state=session.state))
                elif data["type"] == "playback_done":
                    await session.playback_completed(data.get("turn"), status=data.get("status", "completed"))
                elif data["type"] == "interrupt":
                    if not await session.interrupt(data.get("turn")):
                        await session.send_json(protocol.interrupt_rejected(turn=data.get("turn")))
                elif data["type"] == "diag":
                    log.info("diag phone: %s", data)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket error")
    finally:
        await session.disconnect()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_SETTINGS.port, log_level="info")
