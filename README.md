<p align="center">
  <img src="static/assets/logo.jpg" alt="Lari — the Lare" width="240">
</p>

# Lari

**Lari** is a self-hosted voice bridge for [Hermes](https://github.com/NousResearch/hermes-agent). A phone browser acts as a microphone and speaker; Hermes remains the assistant brain. Each satellite is a **Lare**, represented by the five-state flame mascot. A dedicated hardware satellite is future work.

> **Wake phrase:** one setting drives the whole pipeline. `LARI_WAKE_PHRASE` (default `ehi lari`, or `hey lari` for a non-Italian `LARI_STT_LANG`) rebuilds the command regex, the Vosk wake grammar, the junk cleanup and the provider keyterms automatically. The environment-variable prefix was renamed to `LARI_*`. Existing installs must rename the keys in their private `.env`, keeping values unchanged. Test fixtures are synthetic: real-voice calibration data is per-installation and never committed. A new phrase still deserves real-voice calibration: replay real recordings through the gate and add the observed ASR variants to `LARI_WAKE_ALIASES`.

## The name

Roman households kept Lares (Latin singular Lar; Italian Lare, plural Lari): the spirits who watched over the hearth and the family — every home had its own. The project is named Lari after them. Each satellite is a Lare — the device's proper name: a small hearth-spirit in flame form that keeps you company and answers when you call.

In the project artwork the Lare rides on the shoulder of Hermes — the satellite and its brain.

## How it works

```text
Phone browser ── 16 kHz PCM via WebSocket ──► local wake gate → VAD → STT
Phone speaker ◄──── segmented edge-tts audio ◄──── Hermes API (same conversation)
```

The browser displays **idle, listening, thinking, speaking and error** using individually derived animated SVG assets; `scripts/derive_mascot_states.py` rebuilds them from the editable animated source `static/assets/lare-concept.svg` (which doubles as the state-cycle demo at `/<token>/assets/lare-concept.svg`). Each asset carries only its own state's motion. `waking` and `recording` use the listening illustration, while `transcribing` uses thinking. A separate status badge distinguishes local pre-wake monitoring, the local no-wake follow-up window, pending post-wake STT, and audio actually sent to ElevenLabs. The paid-audio indication comes from the bridge **after a successful Realtime audio send**, not from a mascot state or an assumed provider connection; it is not an ElevenLabs balance or billing estimate. The wake phrase displayed in the UI comes from the initial WebSocket state frame—not from the brand or a static string. The bridge does not open a provider connection until the local wake gate confirms the utterance.

## Quick start

Requirements: Python 3.11/3.12, Hermes with the API server enabled, the Hermes source tree (only for the optional openWakeWord engine), a separately downloaded Italian Vosk model, and HTTPS for mobile microphone access (localhost is exempt).

Run `./setup.sh` to automate everything below (virtualenv, dependencies, Italian Vosk model, a generated `LARI_TOKEN` in `.env`), or do it by hand:

```bash
git clone https://github.com/matteo-genovese/lari.git
cd lari
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

- Download an Italian Vosk model; place its extracted folder at `models/vosk-model-small-it-0.22/` or set `LARI_VOSK_MODEL_DIR` to its absolute path. Verify `am/final.mdl` exists inside. Downloaded models are ignored by Git.
- Generate a unique URL token with `python -c 'import secrets; print(secrets.token_urlsafe(32))'` and put it in `LARI_TOKEN` in `.env`. Never commit that file or disclose the token.
- Configure `LARI_HERMES_API` and `LARI_HERMES_KEY` as required by your Hermes installation; `LARI_HERMES_ROOT` can point to a nonstandard source checkout.
- The example selects local Whisper STT. To use ElevenLabs Scribe Realtime instead, set `LARI_STT_BACKEND=elevenlabs_realtime` and supply `ELEVENLABS_API_KEY` in your **private** environment. Realtime is a paid service; do not enable it by copying an example inadvertently.

Run the bridge after loading your private environment:

```bash
set -a
. ./.env
set +a
.venv/bin/uvicorn lari.server:app --host 127.0.0.1 --port "${LARI_PORT:-8643}"
```

Under systemd use `ExecStart=<repo>/.venv/bin/python -m lari.server` with `WorkingDirectory` set to the repository root.

Expose it **only** through a private network with HTTPS (for example, Tailscale Serve). Open `https://<your-private-host>/<your-token>/` in the phone browser and press **Avvia ascolto**. The initial red *disconnesso* indicator is expected before starting the microphone/WebSocket. Keep the page foregrounded and the phone awake. Never expose this token-protected bridge directly to the open internet: it can invoke Hermes tools.

## Runtime settings

- `LARI_TOKEN`: required secret in the URL path; rejects absent/incorrect tokens.
- `LARI_PORT`: listener port (default `8643`).
- `LARI_HERMES_API`, `LARI_HERMES_KEY`: Hermes API endpoint and optional key.
- `LARI_HERMES_ROOT`: Hermes source tree; defaults to `~/.hermes/hermes-agent`.
- `LARI_AGENT_BACKEND`: set `hermes` for the full agent and tools.
- `LARI_HERMES_PROVIDER`, `LARI_HERMES_MODEL`: per-satellite agent model without changing Telegram's model.
- `LARI_SESSION_KEY`: names the Hermes conversation carried by the satellite (default `lari`); changing its value starts a fresh memory.
- `LARI_STT_BACKEND`: `whisper` (local), `vosk`, `elevenlabs_realtime` or other configured backend.
- `LARI_STT_MODEL`, `LARI_STT_LANG`: local fallback model and language.
- `LARI_VOSK_MODEL_DIR`: extracted Vosk model directory; must contain `am/final.mdl`.
- `LARI_WAKE_PHRASE`: the wake phrase; command regex, Vosk grammar, junk cleanup and keyterms all derive from it (default `ehi lari`/`hey lari` by `LARI_STT_LANG`).
- `LARI_WAKE_ALIASES`: comma-separated extra accepted renderings (observed ASR variants); extends matching, grammar and cleanup.
- `LARI_WAKE_CONFIRM`: second local gate before the paid provider opens (default on). It vetoes only a confident mismatch — both local recognizers clearly transcribing non-wake speech — and passes every doubt, so an ASR mishearing never kills a real wake. The slow local model runs only on the veto path, never on real wakes. Set `0` to disable if a real wake is ever lost; add the lost rendering to `LARI_WAKE_ALIASES` instead when possible.
- `LARI_STT_KEYTERMS`: comma-separated vocabulary (names, places) biased in cloud STT; the realtime path drops terms longer than 20 characters.
- `LARI_WAKE_RE`: full command-regex override for advanced calibration.
- `LARI_REALTIME_DAILY_SECONDS`: local limit on seconds sent to Realtime, **not** a hard account spending limit.

See `.env.example` for a minimal, nonsecret template. The UI brand is independent of the voice identity. Avoid putting installation-specific private URLs, recordings or downloaded models into repository files.

## PWA & usage reporting

Add the page to your home screen (PWA): the manifest and service worker make it open fullscreen and keep the shell available offline, so a closed connection shows as disconnected instead of a blank tab. The monthly usage report is served at `/<token>/usage` (turns, paid realtime seconds, local turns) and summarized in the UI; set `LARI_USAGE_EUR_PER_MIN` in your private environment to attach a cost estimate from your own provider rate — no price is ever hardcoded.

## Privacy, costs and testing

- Wake confirmation uses Vosk plus local faster-whisper. ElevenLabs Realtime opens **only after** the local gate. If Realtime is unavailable or reaches its local usage limit, transcription falls back to local STT rather than paid batch. Local models consume CPU; ElevenLabs incurs provider usage when selected. The Vosk gate is a deliberately loose candidate detector: its constrained grammar maps near-miss speech onto the wake phrase, so strict rejection happens when the command is extracted from the transcript.
- The microphone is muted while the phone plays the response and through the echo tail. The interrupt button is **not** voice barge-in.
- The token-protected diagnostic route `/<token>/debug/last.wav` can return captured microphone audio. Limit token access, do not publish recordings, and remove/restrict this route if remote diagnostics are unwanted.
- Tests run without provider credentials or private recordings:

```bash
.venv/bin/python -m unittest discover -q
.venv/bin/python -m py_compile lari/server.py lari/stt_backends.py lari/wake_config.py
```

`scripts/bench_wake_gate.py` is the second gate's A/B benchmark: it replays real recordings through the candidate gate and the veto and reports the three decision numbers (false negatives on real wakes, added latency, candidate seconds kept away from the paid provider).

A green unit suite is not an on-phone wake test: validate wake → transcription → Hermes → audible TTS on the actual device before changing the wake configuration.

## License

[MIT](LICENSE) © 2026 Matteo Genovese.
