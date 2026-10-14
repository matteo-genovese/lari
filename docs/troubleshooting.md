# Troubleshooting

## Logs and transport checks

For the generated user services:

```bash
journalctl --user -u lari-server.service -u lari-desktop.service -f
systemctl --user status lari-server.service lari-desktop.service
lari-desktop --status
```

`lari-desktop --status` requires the installed launcher and reports the local host, not the bridge HTTP inventory. Server diagnostics include connected/disconnected sessions, microphone `diag`, frame counts, RMS peaks/noise floor, Vosk candidates, second-gate pass/veto, VAD duration, STT language/text, Hermes failures/stalls, approvals and TTS retries. Logs and saved WAVs can contain private speech; do not publish them unredacted.

Use the [WebSocket probe](installation.md#websocket-probe) to separate token/transport failures from audio failures. It sends no microphone audio. A successful pong does not verify the microphone, Hermes credentials or speech models. This checkout implements `GET /<token>/status`: check that exact path for the [connection inventory](protocol.md#device-identity-and-status). A wrong token or an empty configured token returns `403`; bare `/status` redirects to the token-scoped page route rather than returning the inventory.

## Known cases

| Symptom | Cause and check |
| --- | --- |
| edge-tts service returns `403` | Known reported case: an old edge-tts version. Upgrade it in the bridge venv with the command below and restart the bridge. No exact fixed version is established here. |
| Bluetooth microphone produces no useful audio | Known reported case: capture is routed to a silent input. Check frame/RMS diagnostics and OS input routing; verify the actual selected microphone instead of assuming capture success implies audible speech. |
| TTS is picked up as a new command | Speaker-to-microphone echo. Confirm playback muting and acknowledgements, retain the echo tail, and check that raw mode is off. See [calibration](calibration.md#follow-up-and-echo). |
| STT language changes on short clips | Automatic language detection on short/noisy audio can drift. Set `LARI_STT_LANG` for local Whisper; an empty value enables automatic detection. Cloud request language is currently fixed to Italian, so this setting is not a global provider-language switch. |
| Poor recognition in a reverberant room | Known reported case: the microphone receives reflected/noisy speech. Compare a private recording with RMS/VAD logs, improve placement or input selection, and calibrate in that room. No universal threshold fixes it. |
| A local wake is confirmed but ASR omits it and the turn is discarded | Historical reported failure from strict transcript-only matching. Current `command_for_turn()` uses `resolve_vosk_command()` after local confirmation, preserving nonempty command text even without the wake rendering. If the journal shows `scartato (fuori dalla conversazione)`, check whether confirmation was actually present and which backend/path ran. Empty/filtered STT can still produce no turn. |
| Hermes asks for tool approval | Approval is forwarded for observability, then the voice stream closes and the assistant asks to retry from Telegram. The voice channel cannot approve tools; this is not a microphone failure. |
| Whisper cannot load on a fresh installation | `HF_HUB_OFFLINE=1` by default and the selected model may not be cached. Make the model available before testing; setup downloads Vosk, not Whisper. |
| API cannot be reached or a reply stalls | Check `LARI_HERMES_API`, credentials and the bridge-side SSH tunnel. `LARI_AGENT_STALL_S` limits time without semantic events; SSE keepalives do not reset it. The optional CLI fallback requires its own Hermes tree. |
| Server rejects all token-scoped routes | `LARI_TOKEN` is empty or does not match. Confirm the service loads the private environment; `python -m lari.server` does not read `.env` by itself. |

```bash
.venv/bin/python -m pip install --upgrade 'edge-tts>=6.1,<8'
systemctl --user restart lari-server.service
```

The edge-tts range above matches `requirements.txt`; it does not assert a particular version fixes every `403`.

## Tests

From the repository root, with dependencies installed:

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests use synthetic fixtures and mocked providers rather than private recordings or provider credentials. Always validate one real wake → Hermes reply → audible playback before treating a deployment as working.
