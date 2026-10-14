# Protocol

The satellite connects to `/<token>/ws`. The path token must be configured and match; an invalid token is rejected before WebSocket acceptance (the adapter requests close code `4403`). Binary input is PCM16 little-endian at 16 kHz, mono, int16. Outgoing text frames are JSON objects with `type`, constructed by `lari/protocol.py`. MP3 metadata precedes its binary frame.

Semantic states are `idle`, `listening`, `waking`, `recording`, `transcribing`, `thinking`, `speaking`, `error`. They describe the interaction independently of STT, Hermes SSE or TTS internals. The initial `state` frame includes the wake/voice configuration.

## Satellite → bridge

| Frame | Fields / format | Meaning |
| --- | --- | --- |
| Binary | PCM16 little-endian, 16 kHz mono int16 | Captured microphone audio. |
| `ping` | `type` | Request a liveness response. |
| `playback_done` | `turn`; optional `status`, default `completed`, also accepts `failed` | Report actual playback completion or failure. |
| `interrupt` | `turn` | Cancel the matching active turn. |
| `ptt` | `phase`: `down` or `up` | Start a manual turn, then close its utterance. |
| `diag` | `ctx`, `rate`, `mic`, `frames`, `vis`, `raw` | Microphone diagnostics logged by the adapter. |

All controls also contain `type`. Unknown controls, malformed JSON and invalid `ptt` phases are ignored. Legacy `playback_done` without a turn is accepted; a missing interrupt turn targets the active turn. New clients should send explicit turn IDs so stale events cannot affect another turn. Failed playback does not open follow-up.

## Bridge → satellite

| Type | Fields | Meaning |
| --- | --- | --- |
| `state` | `state`; optional `turn`, `followup`, `manual`, `phrase`, `provider`, `voice`, `sensitivity`, `confirm_frames`, `interrupted`, `note`, `error` | Semantic state and optional context. |
| `partial_transcript` | `text`, `turn` | Provisional text; empty text clears it. |
| `transcript` | `text`, `turn`; optional `command` | Final user text, optionally marked as a command. |
| `reply` | `text`, `turn` | Hermes reply. |
| `stt_status` | `mode`: `local` or `realtime`, `turn` | Realtime-path send status: `realtime` after a successful send, `local` when unavailable or a send is refused. Not emitted for ordinary local or batch backends. |
| `audio_start` | `turn` | Start a segmented MP3 reading. |
| `audio_chunk` | `turn`, `seq` | Metadata for the next binary MP3 segment; sequence starts at zero. |
| `audio_end` | `turn` | No more segments; queued playback may still be running. |
| `audio` | `fmt`: `mp3`, `bytes`, `turn` | Metadata for the next single binary MP3 frame. |
| `followup` | `seconds`; optional `interrupted` | Local no-wake conversation window. |
| `interrupt_ack` | `turn`, `status`: `interrupted`, `echo_tail_ms` | Cancellation accepted; wait for the echo tail. |
| `interrupt_rejected` | `turn` | No matching active turn. |
| `pong` | `state` | Current semantic state. |
| `approval` | `turn`, `approval` | Hermes approval event for observability. |
| `fatal` | `error` | Wake engine startup failed. |

Optional fields are omitted when not supplied. Send `playback_done` after the audio is actually played, not at `audio_end`; completion opens follow-up. Hold-to-talk and tap-to-toggle both use the same `ptt` down/up pair. They bypass wake detection for that turn. The interrupt button is not voice barge-in. No inbound voice approval control exists.

## HTTP endpoints

These are GET routes in `lari/server.py`. Token-scoped routes return `403` for an absent configured token or a mismatched path token.

| Route | Response |
| --- | --- |
| `/` | Plain-text instruction to open the token-scoped page; no token required. |
| `/<token>/` | Reference HTML page with `Cache-Control: no-store, must-revalidate`. |
| `/<token>/assets/<asset_name>` | Allowlisted bundled mascot/branding assets; unknown or missing files return `404`. |
| `/<token>/manifest.webmanifest` | Token-scoped PWA manifest with fullscreen display. |
| `/<token>/sw.js` | Service worker for the app shell. |
| `/<token>/status` | Version-1 JSON connection inventory; see device identity and status below. |
| `/<token>/usage` | Current-month usage JSON: `month`, `turns`, `local_turns`, `realtime_s`, `realtime_min`, `days`, `est_eur`; corruption can add `corrupt` and `corrupt_copy`. |
| `/<token>/debug/last.wav` | Up to about 12 seconds of recent session PCM as WAV; `404` when no audio is available. Temporary response file is deleted after transmission. |

`est_eur` is null unless `LARI_USAGE_EUR_PER_MIN` is set. Paid-audio UI status follows a successful Realtime send; it is not an account balance or invoice. The diagnostic route contains private microphone audio and does not select a device when several connections exist.

## Device identity and status

The reference page (`static/index.html`) reads the first `device` query parameter with `URLSearchParams`, trims whitespace and lowercases it. The normalized name must match `[a-z0-9][a-z0-9-]{0,31}` in full. A valid name is URL-encoded and appended to the WebSocket URL: `?device=Phone` therefore connects to `/<token>/ws?device=phone`.

If the parameter is absent, the page omits it from the WebSocket URL and the server uses `default`, without a warning. A present but empty or invalid name also omits it, falling back to `default`, and adds the visible log note “Invalid or empty device name; using default.”

The WebSocket adapter (`lari/server.py`) validates the received name without trimming or lowercasing it. An omitted name defaults to `default`; an empty or invalid name requests close code `4403` before acceptance, as does an absent configured token or a wrong path token. Because rejection happens before acceptance, clients may observe an HTTP handshake rejection rather than a WebSocket close frame.

`Session` (`lari/session.py`) uses `LARI_SESSION_KEY` unchanged for device `default`. Other devices use `LARI_SESSION_KEY-<name>`: a base key `lari` gives `lari-phone`. The derived settings are used for Hermes HTTP/SSE and CLI calls. Different names therefore select different Hermes conversation keys. Names are not exclusive: connections using the same name share the derived key, while their in-process state and continuation IDs remain per connection.

`GET /<token>/status` returns HTTP `200` with this JSON shape:

```json
{"version":1,"phrase":"<phrase>","stt_backend":"<backend>","devices":[{"device":"phone","connected":true,"state":"listening","turn":0}]}
```

`phrase` is the configured wake display phrase (`Settings.wake_config.display`); `stt_backend` is the configured backend, not a per-turn fallback result. Each device object contains exactly `device`, `connected` (always `true`), `state` (the session's current semantic state) and `turn` (its integer turn counter, initially zero). There is one entry per active connection, including repeated names, sorted ascending by `(device, state, turn)`. Disconnected sessions are removed; no active connections gives `devices: []`. The response does not include tokens, API keys or Hermes session keys/IDs. The [bar integration](../integrations/bar/README.md) consumes this version-1 payload.

An empty configured `LARI_TOKEN` or a mismatched path token returns HTTP `403` with an empty body. There is no tokenless status endpoint: `/status` redirects with `307` to `/status/`, which is handled as the generic token-scoped page (with `status` as the token), not as the inventory route.

The PWA manifest's `start_url` is `/<token>/` without a device query. A manifest launch therefore uses `default`; it does not preserve a device selected in the original page URL.
