"""Microphone buffering, adaptive VAD, endpointing and private diagnostics."""
from __future__ import annotations
import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from .config import get_settings
_SETTINGS = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
log = logging.getLogger("lari")
VAD_MIN_RMS = _SETTINGS.vad_min_rms
VAD_NOISE_MULT = _SETTINGS.vad_noise_mult
SILENCE_END_S = _SETTINGS.silence_end_s
MAX_UTTERANCE_S = _SETTINGS.max_utterance_s
MIN_SPEECH_S = _SETTINGS.min_speech_s
IDLE_ABORT_S = _SETTINGS.idle_abort_s

def save_turn_audio(pcm: np.ndarray, directory: Path | None = None,
                    name: str | None = None, keep: int = 5) -> Path:
    """Keep a few turn WAVs for reproducible diagnostics (local only)."""
    import wave
    directory = directory or BASE_DIR / "calibration" / "turns"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if name is None:
        name = f"turn_{time.time_ns()}.wav"
    path = directory / name
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.astype(np.int16, copy=False).tobytes())
    for old in sorted(directory.glob("turn_*.wav"))[:-keep]:
        old.unlink()
    return path


class SessionAudio:
    """Audio state and algorithms shared by a satellite session."""

    def _init_audio(self):
        self.frames = bytearray()          # frame-alignment residue
        self.noise_floor = 500.0
        self.last_silent = time.time()
        self.recent = bytearray()
        self.recent_lock = threading.Lock()
        self.audio_id = uuid.uuid4().hex

    def echo_muted(self, now, seconds):
        return now - self.last_tts < seconds

    def pre_roll(self):
        with self.recent_lock:
            return bytes(self.recent[-int(1.5 * SAMPLE_RATE * 2):])

    def _remember_audio(self, chunk: bytes) -> None:
        with self.recent_lock:
            self.recent += chunk
            limit = 12 * SAMPLE_RATE * 2
            if len(self.recent) > limit:
                del self.recent[: len(self.recent) - limit]


    def _candidate_from_queue(self, prelude: bytes, wait_for_gate: bool) -> bytes:
        """Return a contiguous candidate, bounded before any cloud STT call."""
        candidate = bytearray(prelude)
        if not wait_for_gate:
            return bytes(candidate)
        target = int(2.5 * SAMPLE_RATE * 2)
        while len(candidate) < target and not self.stop.is_set():
            try:
                chunk = self.recv_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._remember_audio(chunk)
            candidate += chunk
        return bytes(candidate[:target])


    async def _record_utterance(self, prelude: bytes = b"", realtime=None):
        """VAD: collect PCM while speech lasts, then close after SILENCE_END_S of silence.

        Timing is computed on received samples (not the wall clock), so the
        behavior stays identical with the live stream and in tests.
        `prelude` is the audio that precedes the trigger (it keeps the wake
        phrase, spoken before the system reacts, from being lost).
        """
        collected = bytearray(prelude)
        voiced_s = 0.0
        idle_s = 0.0
        started = False
        last_data = time.time()
        t_end_max = time.time() + MAX_UTTERANCE_S
        reported_stt_mode = None

        async def forward_audio(chunk: bytes):
            nonlocal reported_stt_mode
            assert realtime is not None
            sent = await realtime.send_audio(chunk)
            if sent:
                self._turn_paid_s = getattr(self, "_turn_paid_s", 0.0) + len(chunk) / (SAMPLE_RATE * 2)
            mode = "realtime" if sent else "local"
            if mode != reported_stt_mode:
                reported_stt_mode = mode
                await self.send_json({"type": "stt_status", "mode": mode, "turn": self.turn})

        if realtime is not None and prelude:
            # The local wake detector fired before recording began. Sending this
            # confirmed pre-roll after opening the provider preserves the wake
            # gate without ever keeping an idle provider connection alive.
            await forward_audio(prelude)
        if prelude:
            # A candidate is already buffered before the recorder starts. Feed
            # it through the same local VAD accounting so a quiet gap at the
            # queue boundary cannot discard the first spoken words.
            seed = np.frombuffer(prelude, dtype=np.int16)
            seed_threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
            for offset in range(0, len(seed), 1600):
                frame = seed[offset:offset + 1600]
                if len(frame) == 0:
                    continue
                dt = len(frame) / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                if rms >= seed_threshold:
                    started = True
                    voiced_s += dt
                    idle_s = 0.0
                elif started:
                    idle_s += dt
        while time.time() < t_end_max:
            try:
                chunk = await asyncio.to_thread(self.recv_queue.get, True, 0.2)
            except queue.Empty:
                # stream interrupted (client stalled): close instead of hanging
                if time.time() - last_data > 3.0:
                    break
                continue
            if chunk is None or len(chunk) < 2:
                break
            last_data = time.time()
            collected += chunk
            arr = np.frombuffer(chunk, dtype=np.int16)
            dt = len(arr) / SAMPLE_RATE
            if realtime is not None:
                # Realtime STT needs one contiguous stream.  Local VAD still
                # decides when the turn ends, but must not punch quiet holes in
                # the audio sent to the provider.
                await forward_audio(chunk)
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
            # The floor adapts ONLY before the first speech: during an
            # utterance the quietest syllables would be learned as ambient noise,
            # the threshold would rise above the speaker's own peaks and the
            # recorder would hang (real failure: 22 s collected, sentence
            # spoken twice in the transcript).
            if not started:
                if rms < threshold:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 1.0)
                    self.last_silent = time.time()
                elif time.time() - self.last_silent > 3.0:
                    self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                    self.last_silent = time.time()
                    threshold = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
                    log.info("VAD: floor adattato al rumore ambiente -> soglia %.0f", threshold)
            if rms >= threshold:
                if not started:
                    log.info("VAD: parlato rilevato (rms=%.0f soglia=%.0f)", rms, threshold)
                started = True
                idle_s = 0.0
                voiced_s += dt
            elif started:
                idle_s += dt
                # Close anyway after double the expected silence: the wake or the
                # follow-up already confirmed spoken intent; the voiced quorum
                # must never be able to hang the recorder.
                if idle_s >= SILENCE_END_S and (
                        voiced_s >= MIN_SPEECH_S or idle_s >= SILENCE_END_S + 2.0):
                    break
            else:
                idle_s += dt
                if idle_s >= IDLE_ABORT_S:
                    log.info("VAD: abort, nessun parlato entro %.1fs", IDLE_ABORT_S)
                    return None
            if idle_s >= SILENCE_END_S and (
                    voiced_s >= MIN_SPEECH_S or idle_s >= SILENCE_END_S + 2.0):
                break
        pcm = np.frombuffer(bytes(collected), dtype=np.int16)
        log.info("VAD: raccolti %.2fs (parlato %.2fs, started=%s)", len(pcm) / SAMPLE_RATE, voiced_s, started)
        # Never trim the head: the pre-roll contains the wake, often quieter
        # than the command. The previous energy trim removed the wake phrase,
        # leaving only "what's the weather tomorrow...", which the regex then
        # rejected. Whisper already has its own VAD for leading silence.
        start = 0
        thr = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT) * 0.5
        win = 1600
        end = len(pcm)
        for i in range(len(pcm) - win, start, -win):
            if float(np.sqrt(np.mean(pcm[i:i + win].astype(np.float32) ** 2))) >= thr:
                end = min(len(pcm), i + 2 * win)
                break
        log.info("VAD: utterance tagliata %.2fs -> %.2fs",
                 len(pcm) / SAMPLE_RATE, (end - start) / SAMPLE_RATE)
        return pcm[start:end]


    def update_noise_floor(self, rms):
        thr = max(VAD_MIN_RMS, self.noise_floor * VAD_NOISE_MULT)
        if rms < thr:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 1.0)
            self.last_silent = time.time()
        elif time.time() - self.last_silent > 3.0:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
        return thr

    def save_calibration(self, directory=None):
        import wave
        with self.recent_lock:
            data = bytes(self.recent)
        if len(data) < 2 * SAMPLE_RATE * 2:
            return None
        calib = directory or BASE_DIR / "calibration"
        calib.mkdir(exist_ok=True)
        dest = calib / f"mic_{time.strftime('%Y%m%d_%H%M%S')}_{self.audio_id}.wav"
        fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output, wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(data)
        for old in sorted(calib.glob("mic_*.wav"))[:-8]:
            old.unlink()
        log.info("calibrazione salvata: %s (%.1fs)", dest.name, len(data) / 2 / SAMPLE_RATE)
        return dest
