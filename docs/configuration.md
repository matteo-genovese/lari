# Configuration

This reference lists every `LARI_*` environment variable read by `lari/config.py`. Defaults are code defaults, not overrides in `.env.example`. `empty` means the empty string unless stated otherwise. Runtime files shown under `<repo>` default to the repository root. Numeric settings are validated at load time and float values must be finite.

The bridge takes one immutable settings snapshot; restart it after environment changes. Load the private `.env` in the launching shell or service; `lari.server` does not load it automatically. The desktop loads `<repo>/.env` through python-dotenv. Keep secrets, authenticated URLs, recordings and installation-specific paths out of public files.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LARI_HERMES_ROOT` | `~/.hermes/hermes-agent` | Hermes tree for optional wake-engine import and CLI fallback; expands `~`. |
| `LARI_TOKEN` | `empty` | Secret path token; an empty value makes the bridge reject token-scoped requests. |
| `LARI_DESKTOP_URL` | `empty` | Complete private satellite page URL for remote desktop mode; see the desktop URL restrictions. |
| `LARI_PORT` | `8643` | Listener port for module startup and desktop installation; integer 1–65535. Uvicorn CLI needs its own port argument. |
| `LARI_HERMES_API` | `http://127.0.0.1:8642` | Hermes API base URL; requests append `/v1/chat/completions`. |
| `LARI_HERMES_KEY` | `empty` | Optional Hermes bearer API key. |
| `LARI_SESSION_KEY` | `lari` | Base Hermes conversation key; `default` uses it unchanged, other devices use `<base>-<device>`. |
| `LARI_HERMES_PROVIDER` | `empty` | Optional per-request provider override; empty follows the Hermes profile. |
| `LARI_HERMES_MODEL` | `empty` | Optional per-request model override; empty follows the Hermes profile. |
| `LARI_VOICE_SYSTEM` | `built-in Italian prompt (exact value below)` | System message for Hermes: tolerate imperfect transcription, give brief spoken Italian replies and use tools only when needed. |
| `LARI_TTS_VOICE` | `it-IT-ElsaNeural` | edge-tts voice identifier. |
| `LARI_STT_MODEL` | `base` | Local faster-whisper model, including second-gate confirmation/fallback; the template explicitly selects `small`. |
| `LARI_STT_BACKEND` | `whisper` | One of `whisper`, `vosk`, `elevenlabs_realtime`, `elevenlabs`, `groq`, `openai`; unsupported values fail validation. |
| `LARI_VOSK_MODEL_DIR` | `<repo>/models/vosk-model-small-it-0.22` | Extracted Vosk directory for wake and local transcription; must contain `am/final.mdl`. |
| `LARI_STT_LANG` | `it` | Local Whisper language; empty enables automatic detection. Also used to derive an unspecified wake phrase, with the caveat below. |
| `LARI_STT_GATE` | `empty` | Loaded into `Settings.stt_gate`; currently read but not used at runtime. |
| `LARI_WAKE_PROVIDER` | `whisper` | Default energy VAD plus local Vosk gate; other values are forwarded to the optional Hermes wake engine. |
| `LARI_WAKE_CONFIRM` | `1` | Enable second local confirmation gate; only `0` and `1` accepted. |
| `LARI_SENSITIVITY` | `0.5` | Sensitivity passed to the optional Hermes wake engine; finite number 0–1. |
| `LARI_CONFIRM_FRAMES` | `3` | Confirmation frames passed to the optional Hermes wake engine; integer ≥1. |
| `LARI_AMBIENT_PAUSE` | `6` | Seconds of candidate backoff after rejected ambient speech; ≥0. |
| `LARI_ECHO_MUTE` | `2.5` | Bridge echo suppression duration in seconds, including interruption tail; ≥0. |
| `LARI_FOLLOWUP_S` | `30` | Seconds of no-wake follow-up after successful playback; ≥0. |
| `LARI_PLAYBACK_ACK_TIMEOUT` | `45` | Seconds to wait for playback acknowledgement; >0. |
| `LARI_STREAM_TTS_QUEUE_MAX` | `8` | Maximum queued TTS sentences; integer ≥1. |
| `LARI_STREAM_TEXT_MAX_CHARS` | `4000` | Maximum reading text characters; integer ≥1. |
| `LARI_STREAM_SENTENCE_MAX_CHARS` | `280` | Sentence buffer split target in characters; integer ≥1. Splits at whitespace; punctuation-delimited sentences and unbroken tokens can exceed it. |
| `LARI_VAD_MIN_RMS` | `900` | Minimum energy threshold for speech detection in int16 RMS units; ≥0. |
| `LARI_VAD_NOISE_MULT` | `2.5` | Multiplier on adaptive noise floor for speech threshold; ≥0. |
| `LARI_SILENCE_END` | `2.5` | Seconds of silence used for utterance endpointing; >0. |
| `LARI_MAX_UTTERANCE_S` | `45` | Maximum utterance recording wall time in seconds; >0. |
| `LARI_MIN_SPEECH_S` | `0.7` | Minimum voiced duration for endpointing, in seconds; ≥0. Also rejects turns whose total collected audio is shorter than this duration. |
| `LARI_IDLE_ABORT_S` | `4.0` | Seconds without detected speech before aborting recording; >0. |
| `LARI_AGENT_TIMEOUT` | `180` | Hermes HTTP client timeout in seconds; >0. |
| `LARI_AGENT_STALL_S` | `120.0` | Maximum seconds without semantic SSE events; keepalive comments do not reset it; >0. |
| `LARI_CLI_TIMEOUT` | `70` | Optional Hermes CLI subprocess timeout in seconds; >0. |
| `LARI_GROQ_MODEL` | `whisper-large-v3` | Model sent to the Groq batch transcription endpoint. |
| `LARI_REALTIME_USAGE_FILE` | `<repo>/.realtime_stt_usage.json` | Persistent shared daily Realtime budget file; separate from historical usage. |
| `LARI_REALTIME_DAILY_SECONDS` | `600.0` | Local daily cap on seconds reserved for Realtime audio; ≥0. Not an account spending cap. |
| `LARI_USAGE_LEDGER` | `<repo>/usage.json` | Persistent historical daily usage ledger for the monthly report. |
| `LARI_USAGE_EUR_PER_MIN` | `empty → None` | Optional operator-supplied per-minute rate for estimates; finite value ≥0, no hardcoded price. |
| `LARI_WAKE_PHRASE` | `derived; see below` | Wake phrase used to build command matching, grammar, cleanup and keyterms; whitespace-only is treated as unset. |
| `LARI_WAKE_ALIASES` | `empty` | Comma-separated observed accepted wake renderings; extends matching, grammar and cleanup. |
| `LARI_WAKE_OBSERVED_JUNK` | `empty` | Comma-separated observed spurious transcript prefixes; cleanup only, never wake triggers. |
| `LARI_STT_KEYTERMS` | `empty` | Comma-separated extra provider vocabulary; Realtime terms longer than 20 characters are dropped. |
| `LARI_WAKE_RE` | `empty` | Advanced command-regex override; empty uses the phrase-derived matcher. |

## Derived wake default

If `LARI_WAKE_PHRASE` is empty/unset, `default_phrase()` gets the raw environment value of `LARI_STT_LANG`, with an empty fallback. An explicitly Italian value (case-insensitive prefix `it`) gives `ehi lari`; other or empty values give `hey lari`. Consequently, with both variables absent the effective phrase is `hey lari`, even though `Settings.stt_lang` separately defaults to `it`. The template sets both to Italian explicitly.

## Built-in voice system message

The exact default is retained in Italian because it is runtime configuration, not documentation prose:

```text
Sei l'assistente vocale di un assistente personale. Le battute ti arrivano da un microfono, quindi possono essere trascritte in modo imperfetto: se il senso è intuibile rispondi comunque con la tua interpretazione migliore («ehi ora sono» va letto come «che ora sono»), chiedendo di ripetere SOLO se è davvero incomprensibile e in tal caso in una sola riga e senza strumenti. Rispondi SEMPRE in italiano, con frasi brevi e naturali (massimo 30 secondi di lettura), pensate per essere ascoltate a voce alta. Non fare liste, non usare markdown, non citare file o percorsi. Usa gli strumenti solo se la richiesta lo richiede davvero.
```

## Other environment inputs

`config.py` also reads `ELEVENLABS_API_KEY`, `GROQ_API_KEY` and `OPENAI_API_KEY` (all empty by default), and `HF_HUB_OFFLINE` (default `1`). Local STT needs no provider key. Realtime failure/cap exhaustion uses local fallback, not paid batch.

`LARI_STT_LANG` controls local Whisper; batch ElevenLabs/Groq/OpenAI requests and the ElevenLabs Realtime URL currently specify Italian independently. The wake style prompt is also Italian. Changing the local language does not translate all provider requests or the voice system message.

For existing installations with older variable names, use the `LARI_*` names above while retaining private values. Settings not listed here should not be inferred from `.env` files from other installations.
