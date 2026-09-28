# Lari

**Lari** is a self-hosted voice bridge for [Hermes](https://github.com/NousResearch/hermes-agent). A phone browser acts as a microphone and speaker; Hermes remains the assistant brain. Each satellite is a **Lare**, represented by the five-state flame mascot. A dedicated hardware satellite is future work.

> **Current scope:** this is the proven Italian voice runtime with a *visual* rebrand. The working installation still uses `BUDDY_*` environment variables and a legacy wake gate tuned for “Hey Nic”. Renaming the mascot does **not** change its wake phrase. Do not change `BUDDY_WAKE_PHRASE` alone: the Vosk grammar, local ASR prefix regex and provider keyterms must agree, and another phrase needs real-voice calibration before use.

## How it works

```text
Phone browser ── 16 kHz PCM via WebSocket ──► local wake gate → VAD → STT
Phone speaker ◄──── segmented edge-tts audio ◄──── Hermes API (same conversation)
```

The browser displays **idle, listening, thinking, speaking and error** using individually cropped SVG assets; there are no concept-sheet frames. `waking` and `recording` use the listening illustration, while `transcribing` uses thinking. The wake phrase displayed in the UI comes from the initial WebSocket state frame—not from the brand or a static string. The bridge does not open a provider connection until the local wake gate confirms the utterance.

## Quick start

Requirements: Python 3.11/3.12, Hermes with the API server enabled, the Hermes source tree (for its wake engine), a separately downloaded Italian Vosk model, and HTTPS for mobile microphone access (localhost is exempt).

```bash
git clone https://github.com/matteo-genovese/lari.git
cd lari
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

- Download an Italian Vosk model; place its extracted folder at `models/vosk-model-small-it-0.22/` or set `BUDDY_VOSK_MODEL_DIR` to its absolute path. Verify `am/final.mdl` exists inside. Downloaded models are ignored by Git.
- Generate a unique URL token with `python -c 'import secrets; print(secrets.token_urlsafe(32))'` and put it in `BUDDY_TOKEN` in `.env`. Never commit that file or disclose the token.
- Configure `BUDDY_HERMES_API` and `BUDDY_HERMES_KEY` as required by your Hermes installation; `BUDDY_HERMES_ROOT` can point to a nonstandard source checkout.
- The example selects local Whisper STT. To use ElevenLabs Scribe Realtime instead, set `BUDDY_STT_BACKEND=elevenlabs_realtime` and supply `ELEVENLABS_API_KEY` in your **private** environment. Realtime is a paid service; do not enable it by copying an example inadvertently.

Run the bridge after loading your private environment:

```bash
set -a
. ./.env
set +a
.venv/bin/uvicorn server:app --host 127.0.0.1 --port "${BUDDY_PORT:-8643}"
```

Expose it **only** through a private network with HTTPS (for example, Tailscale Serve). Open `https://<your-private-host>/<your-token>/` in the phone browser and press **Avvia ascolto**. The initial red *disconnesso* indicator is expected before starting the microphone/WebSocket. Keep the page foregrounded and the phone awake. Never expose this token-protected bridge directly to the open internet: it can invoke Hermes tools.

## Runtime settings

- `BUDDY_TOKEN`: required secret in the URL path; rejects absent/incorrect tokens.
- `BUDDY_PORT`: listener port (default `8643`).
- `BUDDY_HERMES_API`, `BUDDY_HERMES_KEY`: Hermes API endpoint and optional key.
- `BUDDY_HERMES_ROOT`: Hermes source tree; defaults to `~/.hermes/hermes-agent`.
- `BUDDY_AGENT_BACKEND`: set `hermes` for the full agent and tools.
- `BUDDY_HERMES_PROVIDER`, `BUDDY_HERMES_MODEL`: per-satellite agent model without changing Telegram's model.
- `BUDDY_STT_BACKEND`: `whisper` (local), `vosk`, `elevenlabs_realtime` or other configured backend.
- `BUDDY_STT_MODEL`, `BUDDY_STT_LANG`: local fallback model and language.
- `BUDDY_VOSK_MODEL_DIR`: extracted Vosk model directory; must contain `am/final.mdl`.
- `BUDDY_WAKE_PHRASE`, `BUDDY_WAKE_RE`: legacy wake configuration; changing the phrase needs grammar/keyterm alignment and a real-voice test.
- `BUDDY_REALTIME_DAILY_SECONDS`: local limit on seconds sent to Realtime, **not** a hard account spending limit.

See `.env.example` for a minimal, nonsecret template. The UI brand is independent of the voice identity. Avoid putting installation-specific private URLs, recordings or downloaded models into repository files.

## Privacy, costs and testing

- Wake confirmation uses Vosk plus local faster-whisper. ElevenLabs Realtime opens **only after** the local gate. If Realtime is unavailable or reaches its local usage limit, transcription falls back to local STT rather than paid batch. Local models consume CPU; ElevenLabs incurs provider usage when selected.
- The microphone is muted while the phone plays the response and through the echo tail. The interrupt button is **not** voice barge-in.
- The token-protected diagnostic route `/<token>/debug/last.wav` can return captured microphone audio. Limit token access, do not publish recordings, and remove/restrict this route if remote diagnostics are unwanted.
- Tests run without provider credentials or private recordings:

```bash
.venv/bin/python -m unittest discover -q
.venv/bin/python -m py_compile server.py stt_backends.py
```

A green unit suite is not an on-phone wake test: validate wake → transcription → Hermes → audible TTS on the actual device before changing the wake configuration.

## License

No license has been selected yet. Public visibility does not grant permission to reuse the code.
