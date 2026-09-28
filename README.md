# Lari

> Lari — voice satellites for Hermes. Named after the Lares, the Roman household spirits who watched over the home.

Lari connects lightweight voice satellites to [Hermes](https://github.com/NousResearch/hermes-agent). A satellite captures speech and plays responses; Hermes remains the assistant brain. The current client is a browser-based prototype. ESP32 and other dedicated satellite targets are future work.

The project is **Lari**; one physical device is **a Lare**; the on-screen mascot is **the Lare / il Lare**, a small household spirit. Keep asset/state identifiers prefixed with `lare_` (for example `lare_idle`, `lare_listening`).

The mobile client displays the selected state from this SVG sprite sheet: `idle`, `listening`, `thinking`, `speaking`, and `error`. It crops one vector panel at a time to avoid bundling duplicate artwork. `static/assets/lare-concept.svg` is the sanitized source; the client maps `waking` and `recording` to `listening`, and `transcribing` to `thinking`. Speech currently uses a gentle CSS pulse; it is not amplitude-driven by TTS volume yet.

## How it works

```text
Phone browser / future satellite
  microphone → local wake gate → speech audio ── WebSocket ──► Lari bridge
  speaker    ◄── streamed TTS audio              Hermes API ◄──┤
                                                               ├ local VAD and STT
                                                               ├ optional STT provider
                                                               └ Hermes session continuity
```

The wake phrase is configured at setup, not baked into the product name or interface. `LARI_LANGUAGE=it` defaults to **Ehi Lari**; `LARI_LANGUAGE=en` defaults to **Hey Lari**. Set `LARI_WAKE_PHRASE` to override either default. The UI receives the selected phrase from the bridge and renders localized prompts.

## Quick start

Requirements: Python 3.11 or 3.12, a running Hermes API server, access to the Hermes source tree (defaults to `~/.hermes/hermes-agent`; override with `LARI_HERMES_ROOT`), a Vosk model for the selected language, and HTTPS (required by mobile browsers for microphone access, except on localhost).

```bash
git clone https://github.com/matteo-genovese/lari.git
cd lari
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` before starting the bridge:

- Set `LARI_TOKEN` to a long random access token. Generate one with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`, then copy the result into `.env`. The bridge refuses requests when the value is empty.
- Set `LARI_LANGUAGE` to `it` or `en`.
- Optionally set `LARI_WAKE_PHRASE`; if omitted, the language default is used.
- Point `LARI_VOSK_MODEL_DIR` at a Vosk model matching the selected language. Models are downloaded separately and are not committed.
- Configure the Hermes API URL and key if required by your Hermes deployment.

Load the environment and start the server:

```bash
set -a
. ./.env
set +a
.venv/bin/uvicorn server:app --host 0.0.0.0 --port "${LARI_PORT:-8643}"
```

Open `https://<your-host>/<LARI_TOKEN>/` on the satellite browser, grant microphone permission, then press **Start listening / Avvia ascolto**. The page must remain open and the device awake for the browser-based prototype to listen.

Never expose the bridge directly to the public Internet without TLS and a strong token. Use a private network or a properly secured reverse proxy.

## Configuration

All settings use the `LARI_` prefix. The bridge still accepts legacy `BUDDY_` names for existing installations, but new setups should use `LARI_`.

| Variable | Default | Purpose |
|---|---|---|
| `LARI_TOKEN` | unset | Required bearer-like URL token; generate a unique random value |
| `LARI_LANGUAGE` | `it` | UI, speech recognition, TTS and wake defaults: `it` or `en` |
| `LARI_WAKE_PHRASE` | `Ehi Lari` / `Hey Lari` | Optional setup-time override; otherwise derived from the language |
| `LARI_PORT` | `8643` | HTTP/WebSocket listener port |
| `LARI_HERMES_API` | `http://127.0.0.1:8642` | Hermes API server base URL |
| `LARI_HERMES_ROOT` | `~/.hermes/hermes-agent` | Path to the Hermes source tree used by the wake engine |
| `LARI_HERMES_KEY` | unset | API key when the Hermes endpoint requires authentication |
| `LARI_HERMES_PROVIDER` | `deepseek` | Request provider for this satellite; can be overridden independently |
| `LARI_HERMES_MODEL` | `LARI_DEEPSEEK_MODEL` | Per-satellite model override |
| `LARI_DEEPSEEK_MODEL` | `deepseek-flash` | Default model used by this satellite |
| `LARI_SESSION_KEY` | `lari` | Stable Hermes session scope for this satellite service |
| `LARI_AGENT_BACKEND` | `hermes` | Use the full Hermes agent by default |
| `LARI_STT_BACKEND` | `whisper` | Local recognition by default; `elevenlabs_realtime` is optional |
| `LARI_STT_MODEL` | `base` | faster-whisper model used locally |
| `LARI_STT_LANG` | `LARI_LANGUAGE` | Local ASR language override |
| `LARI_VOSK_MODEL_DIR` | `models/vosk-model-small-<language>` | Path to the separately downloaded Vosk model |
| `LARI_REALTIME_DAILY_SECONDS` | `600` | Local cap on audio sent to Realtime; over-cap and provider-error fallback is local, not paid batch |
| `LARI_FOLLOWUP_S` | `30` | Seconds for follow-up turns after a response |
| `LARI_ECHO_MUTE` | `2.5` | Keep the microphone muted through the speaker echo tail |
| `LARI_TTS_VOICE` | language-specific | edge-tts voice override |

See `.env.example` for the complete set of supported options. Provider API keys (for optional STT providers) are read from the process environment and must never be committed.

## Privacy and cost behavior

- Wake detection runs locally before an ElevenLabs Realtime connection is opened. A Vosk grammar candidate must also be confirmed by local ASR; ordinary ambient speech should not start provider streaming.
- The Realtime daily cap counts PCM seconds sent over the Realtime WebSocket only. It is not an account-wide billing cap. When the cap is reached or Realtime fails, Lari falls back to local Vosk/faster-whisper and does not call paid batch transcription as a hidden fallback.
- Wake audio is held briefly in memory. For diagnostics, the server retains up to five turn WAVs under the ignored `calibration/turns/` directory and exposes a rolling ~12-second capture at `/<token>/debug/last.wav`; both are private runtime data. The debug route requires the installation token. Remove or restrict the route if you do not want remote access to live diagnostic audio. Never commit recordings.
- The browser microphone is muted while TTS plays and through the echo tail. Spoken barge-in is not enabled; use the interrupt button, then speak after the UI returns to listening.
- Hermes session IDs are scoped to the satellite connection so follow-up turns continue the same transcript.

## Development

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python -m py_compile server.py stt_backends.py
```

The inline browser script can be checked with `node --check` after extracting the `<script>` body. Tests use mocked provider responses; live provider credentials and private voice recordings are not required for the unit suite.

## Project language and naming

Code, comments and documentation are written in English for the Hermes community. The interface and spoken defaults support Italian and English. Keep the project name **Lari**, the device name **Lare**, and the mascot name **the Lare / il Lare** distinct. Never put a selected wake phrase in a repository slug, asset name or static UI copy; read it from configuration.

## Topics

`hermes` · `voice-assistant` · `wake-word` · `esp32` · `mascot` · `self-hosted`

## License

No license has been selected yet. Until one is added, standard copyright applies; publication does not grant permission to reuse the code.
