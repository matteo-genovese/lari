# Calibration

Synthetic test fixtures exercise matching and control flow. They do not establish how a real microphone, voice or room behaves. Real recordings are private per-installation data and must never be committed; WAVs, `calibration/` and downloaded models are Git-ignored.

## Wake phrase and recordings

`LARI_WAKE_PHRASE` drives command matching, the Vosk constrained grammar, junk cleanup and provider keyterms. An empty/unset phrase is derived from the raw `LARI_STT_LANG` environment value; see the [default caveat](configuration.md). A new phrase requires replaying both genuine wakes and background speech from the real device.

The local candidate gate uses a constrained Vosk grammar, which can map near-miss speech onto the phrase. With `LARI_WAKE_CONFIRM=1`, the second gate vetoes only when free Vosk and local Whisper both confidently identify non-wake speech. Doubt and recognition failures pass. Whisper on this second gate runs only after free Vosk proposes a confident mismatch.

Add observed accepted ASR renderings to comma-separated `LARI_WAKE_ALIASES`; they extend matching, grammar and cleanup. `LARI_WAKE_OBSERVED_JUNK` is only for observed spurious transcript prefixes: it cleans rendering after a wake, never creates a wake trigger. Avoid adding guessed background words to either list. `LARI_WAKE_RE` is an advanced command-matching override, not a replacement for real calibration.

The bridge keeps about 12 seconds of recent audio for the token-protected `/<token>/debug/last.wav` route. It saves private calibration WAVs on disconnect when enough audio exists, keeps up to eight disconnect WAVs under `calibration/`, and up to five recent turn WAVs under `calibration/turns/`. These retention limits are shared across devices; the debug route returns the first session with buffered audio in an unordered set, without device selection. Do not publish diagnostics or authenticated download URLs.

After exporting the private environment, replay a private 16 kHz mono PCM16 WAV:

```bash
.venv/bin/python scripts/bench_wake_gate.py ~/lari-calibration/wake.wav
```

The script accepts positional WAV paths and prints candidate decisions, free/Whisper cleanliness, latency and vetoed seconds. It does not label ground truth or calculate a false-negative rate automatically: compare its verdicts with your own genuine-wake/background labels. An empty argument list does not run a calibration set.

## VAD and endpointing

The energy threshold is `max(LARI_VAD_MIN_RMS, noise_floor * LARI_VAD_NOISE_MULT)`, with defaults `900` and `2.5`. Watch journal RMS peaks and the noise floor before tuning. The noise floor adapts before speech and is held during the utterance so quieter syllables are not learned as ambient noise.

`LARI_SILENCE_END` defaults to `2.5` seconds; `LARI_MIN_SPEECH_S` to `0.7`, `LARI_IDLE_ABORT_S` to `4.0`, and `LARI_MAX_UTTERANCE_S` to `45`. Silence accounting follows received samples; maximum duration also bounds wall time. Recording also closes after more than three seconds without incoming audio. `LARI_MIN_SPEECH_S` is a voiced-duration threshold for endpointing; the separate turn-length check counts all collected samples, not voiced samples. Pre-roll preserves the quieter wake at the start. Raising thresholds can miss quiet speech; lowering them can admit room noise. Test both speech and silence on the actual device.

## Follow-up and echo

Successful playback acknowledgement opens `LARI_FOLLOWUP_S` (default `30` seconds) without a new wake. The window is evaluated when a turn begins, not after transcription. Failed playback does not open it. Interruption opens follow-up after the configured echo tail.

The browser stops forwarding microphone audio while playing the response and mutes for an additional 700 ms after normal playback. The bridge also suppresses wake candidates using `LARI_ECHO_MUTE` (default `2.5` seconds), and tells interrupted clients the echo-tail duration. `?raw=1` disables browser echo cancellation and noise suppression; use it only for controlled diagnostics. The interrupt control is not voice barge-in.

Test wake → transcription → Hermes → audible TTS, then follow-up and interruption. A passing unit suite cannot calibrate room echo or microphone routing.
