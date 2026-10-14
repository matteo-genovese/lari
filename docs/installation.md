# Installation

## Requirements

Use Python 3.11/3.12 and Hermes with its API server enabled. Install the dependencies in `requirements.txt`. Local wake detection requires an extracted Vosk model containing `am/final.mdl`; the default path is `<repo>/models/vosk-model-small-it-0.22`. Local Whisper also needs its selected model available. `HF_HUB_OFFLINE` defaults to `1`; do not assume a fresh installation will download a missing Whisper model automatically.

Mobile browsers require HTTPS to access the microphone; localhost is exempt. Use a private network for the bridge. The optional Hermes wake engine needs a Hermes tree containing `tools/wake_word.py`; the optional CLI fallback expects `venv/bin/hermes` under `LARI_HERMES_ROOT`. Normal operation uses `LARI_HERMES_API`.

## Environment and models

`./setup.sh` creates `.venv`, installs dependencies, downloads and extracts the Italian Vosk model, and creates a private `.env` with a generated token if it does not exist. It preserves existing environment files. For manual installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Copy the generated value into `LARI_TOKEN` in the private `.env`. Download and extract the Italian model from `https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip` into `models/`, or set `LARI_VOSK_MODEL_DIR` to the extracted directory. Check its layout:

```bash
test -f models/vosk-model-small-it-0.22/am/final.mdl
```

Configure `LARI_HERMES_API` and, if required, `LARI_HERMES_KEY`. The template uses local Whisper (`LARI_STT_MODEL=small`); the code default without that setting is `base`. Select paid STT deliberately: `elevenlabs_realtime` needs `ELEVENLABS_API_KEY`; batch `elevenlabs`, `groq` and `openai` need their respective keys. See [configuration](configuration.md).

Start from the repository root, loading the private environment:

```bash
set -a
. ./.env
set +a
.venv/bin/uvicorn lari.server:app --host 127.0.0.1 --port "${LARI_PORT:-8643}"
```

Only source a trusted shell-compatible `.env`. `lari.server` does not load dotenv itself. `python -m lari.server` binds `0.0.0.0`; the explicit Uvicorn command above binds loopback.

## User systemd service

For a bridge-only installation, create `~/.config/systemd/user/lari-server.service` with the following template, replacing `<repo>` with the absolute repository path:

```ini
[Unit]
Description=Lari voice bridge
After=network.target

[Service]
WorkingDirectory=<repo>
EnvironmentFile=<repo>/.env
ExecStart=<repo>/.venv/bin/python -m uvicorn lari.server:app --host 127.0.0.1 --port 8643 --no-access-log
Restart=on-failure
UMask=0077

[Install]
WantedBy=default.target
```

The port in `ExecStart` must match your chosen `LARI_PORT`; this literal template uses `8643`. Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now lari-server.service
journalctl --user -u lari-server.service -f
```

The [desktop installer](desktop.md) generates its own server and desktop units; use that installer for the Qt host.

## Private HTTPS with Tailscale

With Tailscale installed and connected on the bridge machine, a loopback listener on port `8643` can be exposed with [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve):

```bash
tailscale serve --bg http://127.0.0.1:8643
```

Use the HTTPS address reported by Serve and append the private token path. To select a named satellite, append `?device=phone` to the browser page URL; see [device identity and status](protocol.md#device-identity-and-status). Do not publish authenticated URLs. Press **Start listening**, grant microphone access, and keep the page foregrounded and the phone awake. Before that action, a disconnected indicator is expected. Installing the page as a PWA preserves an offline shell, not offline speech processing. Its manifest launch URL omits the device query and uses `default`.

## WebSocket probe

After exporting `.env` as above, run this from the repository root to check the local token, initial state and ping/pong contract without sending audio:

```bash
.venv/bin/python - <<'PYCODE'
import asyncio
import json
import os
from urllib.parse import quote
from websockets.asyncio.client import connect

async def main():
    token = quote(os.environ["LARI_TOKEN"], safe="")
    port = os.environ.get("LARI_PORT", "8643")
    async with connect(f"ws://127.0.0.1:{port}/{token}/ws") as ws:
        initial = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert initial["type"] == "state", initial
        print(initial)
        await ws.send(json.dumps({"type": "ping"}))
        while True:
            reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if reply["type"] == "fatal":
                raise RuntimeError(reply["error"])
            if reply["type"] == "pong":
                print(reply)
                break

asyncio.run(main())
PYCODE
```

This checks transport, not microphone quality, model readiness or Hermes credentials. Verify wake → transcript → Hermes reply → audible playback on the actual satellite.

## Desktop and remote Hermes

Install `requirements-desktop.txt` into `.venv`, then run `.venv/bin/python scripts/install_desktop.py`; see [desktop](desktop.md). For a desktop-only client of another bridge, set private `LARI_DESKTOP_URL` and rerun the installer. That mode disables the local bridge service.

The remote bridge must reach Hermes itself. If Hermes runs on another machine and its API is loopback-only, establish the [SSH API tunnel](integrations.md#hermes-api-tunnel) on the bridge machine. The desktop does not forward its local Hermes API automatically.
