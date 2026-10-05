"""Per-satellite wake frame processing; no application-global event loop."""
import asyncio
import logging
import queue
import time
import numpy as np
from ..config import get_settings
from .detector import make_engine, vosk_wake
from .confirm import confirm_candidate
_SETTINGS = get_settings()
WAKE_PROVIDER = _SETTINGS.wake_provider
ECHO_MUTE_S = _SETTINGS.echo_mute_s
COOLDOWN_S = 2.0
FRAME = 1280
WAKE_CONFIRM = _SETTINGS.wake_confirm
AMBIENT_PAUSE_S = _SETTINGS.ambient_pause_s
log = logging.getLogger("lari")


class WakeWorker:
    def wake_worker(self):
        self.engine = None
        if WAKE_PROVIDER != "whisper":
            try:
                self.engine = make_engine()
            except Exception:
                log.exception("impossibile creare l'engine wake")
                asyncio.run_coroutine_threadsafe(
                    self.send_json({"type": "fatal", "error": "wake engine startup failed"}), self.loop
                )
                return
        else:
            log.info("wake provider=whisper: VAD energia + gate Vosk locale")
        buf = bytearray()
        speech_streak = 0
        stat_t = time.time()
        stat_frames = 0
        stat_peak = 0.0
        while not self.stop.is_set():
            st = self.state
            if time.time() - stat_t >= 5.0:
                log.info("audio: %d frame/5s, rms peak %.0f, noise floor %.0f, state=%s",
                         stat_frames, stat_peak, self.noise_floor, st)
                stat_t, stat_frames, stat_peak = time.time(), 0, 0.0
            if st in ("waking", "recording"):
                time.sleep(0.05)
                continue
            if st != "listening":
                try:
                    chunk = self.recv_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    return
                buf.clear()
                speech_streak = 0
                continue
            try:
                chunk = self.recv_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                return
            stat_frames += 1
            self._remember_audio(chunk)
            buf += chunk
            while len(buf) >= FRAME * 2:
                frame = bytes(buf[: FRAME * 2])
                del buf[: FRAME * 2]
                arr = np.frombuffer(frame, dtype=np.int16)
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                stat_peak = max(stat_peak, rms)
                thr = self.update_noise_floor(rms)
                if WAKE_PROVIDER == "whisper":
                    speech_streak = speech_streak + 1 if rms >= thr else 0
                    need = 3 if self.noise_floor < 2000 else 8
                    hit = speech_streak >= need
                else:
                    try:
                        hit = self.engine.process(arr)
                    except Exception:
                        log.exception("errore engine")
                        hit = False
                now = time.time()
                if hit and self.echo_muted(now, ECHO_MUTE_S):
                    continue
                if hit and now - self.last_wake >= COOLDOWN_S and rms > 200:
                    prelude = self.pre_roll()
                    followup = time.monotonic() < self.conversation_until and not self.awaiting_playback
                    candidate = self._candidate_from_queue(prelude, wait_for_gate=not followup)
                    if self._launch_local_candidate(candidate) or followup:
                        buf.clear()
                        speech_streak = 0
                        break
                    buf.clear()
                    speech_streak = 0


    # —— wake: runs in a dedicated thread; the event loop must never block ——


    def _launch_local_candidate(self, candidate: bytes) -> bool:
        """Apply local wake/follow-up policy and schedule exactly one turn."""
        if not candidate:
            return False
        followup = time.monotonic() < self.conversation_until and not self.awaiting_playback
        if not followup:
            pcm = np.frombuffer(candidate, dtype=np.int16)
            if not vosk_wake(pcm):
                log.info("candidato parlato scartato dal gate Vosk locale")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
            if WAKE_CONFIRM and not confirm_candidate(pcm):
                log.info("candidato scartato dal secondo gate locale")
                self.last_wake = time.time() + AMBIENT_PAUSE_S - COOLDOWN_S
                return False
        if self.echo_muted(time.time(), ECHO_MUTE_S):
            return False
        self.last_wake = time.time()
        self.state = "waking"
        with self._schedule_lock:
            if self.stop.is_set():
                return False
            future = asyncio.run_coroutine_threadsafe(
                self._on_wake(initial_pcm=candidate, local_wake_confirmed=not followup),
                self.loop,
            )
            self._scheduled.add(future)
        future.add_done_callback(self._scheduled.discard)
        return True

