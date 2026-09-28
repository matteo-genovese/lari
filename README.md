# Lari

**Lari** is a self-hosted voice bridge for [Hermes](https://github.com/NousResearch/hermes-agent). A phone browser acts as a microphone and speaker; Hermes remains the assistant brain. Each satellite is a **Lare**, represented by the five-state flame mascot. A dedicated hardware satellite is future work.

> **Wake phrase:** one setting drives the whole pipeline. `BUDDY_WAKE_PHRASE` (default `ehi lari`, or `hey lari` for a non-Italian `BUDDY_STT_LANG`) rebuilds the command regex, the Vosk wake grammar, the junk cleanup and the provider keyterms automatically. The runtime keeps the legacy `BUDDY_*` names. A new phrase still deserves real-voice calibration: replay real recordings through the gate and add the observed ASR variants to `BUDDY_WAKE_ALIASES`.

## How it works

```text
Phone browser ── 16 kHz PCM via WebSocket ──► local wake gate → VAD → STT
Phone speaker ◄──── segmented edge-tts audio ◄──── Hermes API (same conversation)
```

The browser displays **idle, listening, thinking, speaking and error** using individually cropped SVG assets; there are no concept-sheet frames. `waking` and `recording` use the listening illustration, while `transcribing` uses thinking. A separate status badge distinguishes local pre-wake monitoring, the local no-wake follow-up window, pending post-wake STT, and audio actually sent to ElevenLabs. The paid-audio indication comes from the bridge **after a successful Realtime audio send**, not from a mascot state or an assumed provider connection; it is not an ElevenLabs balance or billing estimate. The wake phrase displayed in the UI comes from the initial WebSocket state frame—not from the brand or a static string. The bridge does not open a provider connection until the local wake gate confirms the utterance.

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
- `BUDDY_WAKE_PHRASE`: the wake phrase; command regex, Vosk grammar, junk cleanup and keyterms all derive from it (default `ehi lari`/`hey lari` by `BUDDY_STT_LANG`).
- `BUDDY_WAKE_ALIASES`: comma-separated extra accepted renderings (observed ASR variants); extends matching, grammar and cleanup.
- `BUDDY_STT_KEYTERMS`: comma-separated vocabulary (names, places) biased in cloud STT; the realtime path drops terms longer than 20 characters.
- `BUDDY_WAKE_RE`: full command-regex override for advanced calibration.
- `BUDDY_REALTIME_DAILY_SECONDS`: local limit on seconds sent to Realtime, **not** a hard account spending limit.

See `.env.example` for a minimal, nonsecret template. The UI brand is independent of the voice identity. Avoid putting installation-specific private URLs, recordings or downloaded models into repository files.

## Privacy, costs and testing

- Wake confirmation uses Vosk plus local faster-whisper. ElevenLabs Realtime opens **only after** the local gate. If Realtime is unavailable or reaches its local usage limit, transcription falls back to local STT rather than paid batch. Local models consume CPU; ElevenLabs incurs provider usage when selected. The Vosk gate is a deliberately loose candidate detector: its constrained grammar maps near-miss speech onto the wake phrase, so strict rejection happens when the command is extracted from the transcript.
- The microphone is muted while the phone plays the response and through the echo tail. The interrupt button is **not** voice barge-in.
- The token-protected diagnostic route `/<token>/debug/last.wav` can return captured microphone audio. Limit token access, do not publish recordings, and remove/restrict this route if remote diagnostics are unwanted.
- Tests run without provider credentials or private recordings:

```bash
.venv/bin/python -m unittest discover -q
.venv/bin/python -m py_compile server.py stt_backends.py
```

A green unit suite is not an on-phone wake test: validate wake → transcription → Hermes → audible TTS on the actual device before changing the wake configuration.

## License

[MIT](LICENSE) © 2026 Matteo Genovese.
