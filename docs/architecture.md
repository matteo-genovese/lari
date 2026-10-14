# Architecture

Lari is the voice layer. Hermes owns model selection, assistant execution, tools and conversation memory. Cloud speech providers supply STT; they are not alternative assistant backends. The browser/PWA is the reference satellite, and `lari/desktop.py` hosts that same page through Qt WebEngine.

| Module | Boundary |
| --- | --- |
| `lari/server.py` | Composition root and FastAPI transport adapter: HTTP/PWA routes, token checks, WebSocket receive loop, JSON and binary sends, connection cleanup. |
| `lari/device.py` | Shared device-name validation and device context for diagnostics. |
| `lari/protocol.py` | Transport-independent message constructors and incoming JSON control parsing. |
| `lari/session.py` | Per-connection semantic state, turn ownership, wake-to-reply orchestration, interruption, playback acknowledgements and follow-up. |
| `lari/wake/config.py` | Wake phrase, aliases, command matching, constrained grammar, cleanup and provider vocabulary. |
| `lari/wake/detector.py`, `confirm.py`, `runtime.py` | Local candidate detection, optional Hermes wake engine, conservative second gate and per-session worker thread. |
| `lari/audio.py` | PCM buffering, adaptive energy VAD, endpointing, pre-roll and private diagnostic WAVs. |
| `lari/stt/` | Local Whisper/Vosk, cloud batch providers, Realtime transport/budget and dispatch with local fallback. |
| `lari/hermes/` | Hermes HTTP/SSE events, session continuity, tool approvals and optional CLI fallback. |
| `lari/tts.py` | edge-tts synthesis, retry, sentence buffering and ordered MP3 delivery with bounded queues. |
| `lari/usage.py` | Persistent daily history and monthly usage summaries, separate from the Realtime daily cap. |

`Session` is transport-agnostic: its constructor receives `send_json` and `send_audio` callbacks, explicit `Settings`, and a ledger. Its entry points accept PCM and semantic controls rather than FastAPI objects. A different transport can drive these entry points; a WebSocket satellite already uses the public wire contract without changing the pipeline.

Each connection owns its `Session`, wake worker, receive queue, audio buffers, noise floor, turn counter, active jobs, playback waiter, follow-up deadline, Realtime connection and Hermes continuation ID. These mutable values are not shared across connections. Heavy Vosk models are cached by path and Whisper models by name; recognizer/decode state stays per call. The usage ledger and file-locked Realtime budget cover the bridge, rather than providing separate spending allowances per satellite.

At import, `server.py` calls cached `get_settings()` once, stores the immutable snapshot in `_SETTINGS`, constructs `USAGE_LEDGER` with it, and passes both explicitly to each `Session`. Runtime components receive settings instead of re-reading environment variables. `load_settings()` validates a supplied mapping without modifying it; `get_settings()` also applies the `HF_HUB_OFFLINE` default. Restart the process after changing its environment. The desktop process loads `.env` itself and creates its own snapshot.

The initial WebSocket state supplies the UI wake phrase and voice configuration. Semantic states describe the interaction; `stt_status` separately describes actual audio transmission. Realtime is opened only after a local wake is accepted or a manual turn starts. Its failures and local budget exhaustion fall back to local Vosk, then local Whisper if Vosk is unavailable, without a paid batch fallback.

The browser renders five mascot assets: idle, listening, thinking, speaking and error. Waking/recording use the listening illustration; transcribing uses thinking. `scripts/derive_mascot_states.py` derives the assets from `static/assets/lare-concept.svg`. The PWA service worker caches a small shell for offline display; the voice path still requires a network connection.

The device-specific Hermes session-key and status contract is described in [protocol](protocol.md#device-identity-and-status), as implemented by the reference page, `Session` and the server. Names select Hermes conversation keys; duplicate names are allowed and appear as separate status entries.
