"""Per-satellite semantic state, turn ownership and playback orchestration."""
from __future__ import annotations
from . import protocol
import asyncio
import logging
import queue
import threading
from threading import Thread
import time
from .config import Settings
from .audio import SessionAudio, save_turn_audio
from .wake.runtime import WakeWorker
from .stt import dispatch as _dispatch
from .hermes import (
    ApprovalNotAvailable, HermesStreamStalled, HermesStreamTurnError,
    ask_hermes, stream_hermes,
)
from .tts import Reading, TTSUnavailable
from . import usage
log = logging.getLogger("lari")
SAMPLE_RATE = 16000
COOLDOWN_S = 2.0
MANUAL_PREROLL_S = 0.6
APPROVAL_NOT_AVAILABLE_REPLY = "Questa richiesta ha bisogno di un'approvazione che non posso darti a voce. Riprova da Telegram."
AGENT_STALLED_REPLY = "Ci sto mettendo troppo: non riesco a completare la risposta."

class Session(SessionAudio, WakeWorker):
    """One satellite owns turns, playback, realtime, follow-up and UI state.

    Immutable Settings may be shared; mutable connection state is never shared.
    """

    def __init__(self, send_json, send_audio, *, settings: Settings, usage_ledger: usage.UsageLedger | None = None):
        self._settings = settings
        self.usage_ledger = usage_ledger or usage.UsageLedger(settings=settings)
        self._send_audio = send_audio
        self._send_json = send_json
        self.engine = None
        self.state = "listening"           # the worker starts listening immediately
        self.last_wake = 0.0
        self._init_audio()
        self._manual_claim = threading.Event()
        self._manual_active = False
        self.recv_queue: queue.Queue = queue.Queue(maxsize=400)
        self.stop = threading.Event()
        self.turn = 0
        # rotating calibration buffer: the last ~12 s of microphone, to see
        # what the detector really hears when it does not fire
        self.last_tts = 0.0
        self.conversation_until = 0.0
        self.awaiting_playback = False
        self.awaiting_playback_turn: int | None = None
        self._playback_waiter: asyncio.Future | None = None
        self.playback_status: str | None = None
        self._interrupted_turn: int | None = None
        self._interrupted_followup_task: asyncio.Task | None = None
        self.active_turn: int | None = None
        self._turn_task: asyncio.Task | None = None
        self.hermes_session_id: str | None = None
        self._partial_turn: int | None = None
        self._reading: Reading | None = None
        self._realtime = None
        self.worker: threading.Thread | None = None
        self._scheduled = set()
        self._jobs = set()
        self._turn_start_lock = asyncio.Lock()
        self._schedule_lock = threading.Lock()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    async def start(self):
        """Start local wake monitoring and announce the satellite configuration."""
        self.worker = Thread(target=self.wake_worker, daemon=True, name="wake-worker")
        self.worker.start()
        await self.send_json(protocol.state(
            state="listening", phrase=self._settings.wake_config.display,
            provider=self._settings.wake_provider, voice=self._settings.tts_voice,
            sensitivity=self._settings.wake_sensitivity,
            confirm_frames=self._settings.confirm_frames,
        ))

    async def send_json(self, data):
        turn = data.get("turn")
        if self.stop.is_set() or (turn is not None and self.turn and
                                 not self._same_turn(turn, self.turn)):
            return
        await self._send_json(data)

    async def on_audio(self, pcm: bytes):
        if not self.stop.is_set():
            try:
                self.recv_queue.put_nowait(pcm)
            except queue.Full:
                pass

    async def playback_completed(self, turn_id, status="completed") -> bool:
        accepted = self.mark_playback_done(turn=turn_id, status=status)
        if accepted and status == "completed":
            await self.send_json(protocol.followup(seconds=self._settings.followup_s))
        return accepted

    async def interrupt(self, turn=None) -> bool:
        return await self.interrupt_current_turn(self.active_turn if turn is None else turn)

    async def disconnect(self):
        """Stop ingress, cancel owned work and await all transport cleanup."""
        self.stop.set()
        self.conversation_until = 0.0
        self.awaiting_playback = False
        self.awaiting_playback_turn = None
        waiter = self._playback_waiter
        if waiter is not None and not waiter.done():
            waiter.cancel()
        with self._schedule_lock:
            scheduled = list(self._scheduled)
            for future in scheduled:
                future.cancel()
        if scheduled:
            await asyncio.sleep(0)
        tasks = {task for task in (*self._jobs, self._turn_task, self._interrupted_followup_task)
                 if task is not None and task is not asyncio.current_task()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._reading is not None:
            await self._reading.cancel()
        if self._realtime is not None:
            await self._realtime.close()
            self._realtime = None
        if scheduled:
            await asyncio.gather(*(asyncio.wrap_future(f) for f in scheduled),
                                 return_exceptions=True)
        while not self.recv_queue.empty():
            self.recv_queue.get_nowait()
        self.recv_queue.put_nowait(None)
        if isinstance(self.worker, Thread):
            while self.worker.is_alive():
                await asyncio.sleep(0.01)
            self.worker.join()
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        try:
            self.save_calibration()
        except Exception:
            log.exception("salvataggio calibrazione fallito")

    async def _send_partial(self, text: str, turn: int):
        if not self._reading_valid(turn):
            return
        self._partial_turn = turn
        await self.send_json(protocol.partial_transcript(text=text, turn=turn))

    async def _clear_partial(self, turn: int):
        """Remove the browser's in-progress transcript after a discarded turn."""
        if self._partial_turn is None:
            return
        partial_turn = self._partial_turn
        self._partial_turn = None
        await self.send_json(protocol.partial_transcript(text="", turn=partial_turn))

    async def _ask_hermes(self, text: str) -> str:
        """Ask Hermes and update only this session's transcript id."""
        turn = self.turn
        result = await ask_hermes(
            text, settings=self._settings, session_id=self.hermes_session_id,
        )
        returned_id = getattr(result, "session_id", None)
        if returned_id and self.turn == turn and not self.stop.is_set():
            self.hermes_session_id = returned_id
        return str(result)

    def _begin_playback(self, turn: int):
        self.awaiting_playback = True
        self.awaiting_playback_turn = turn
        self._playback_waiter = asyncio.get_running_loop().create_future()
        self.playback_status = "pending"

    @staticmethod
    def _same_turn(left, right) -> bool:
        return left is not None and right is not None and str(left) == str(right)

    def mark_playback_done(self, now: float | None = None, turn: int | None = None,
                           status: str = "completed") -> bool:
        """Accept only the current turn's completion ACK.

        A missing turn remains accepted for the pre-segmented legacy client.  A
        failed segmented playback releases the worker without opening follow-up.
        """
        if self.stop.is_set() or not self.awaiting_playback:
            return False
        if turn is not None and self.turn and not self._same_turn(turn, self.turn):
            return False
        if turn is not None and not self._same_turn(turn, self.awaiting_playback_turn):
            return False
        if status not in {"completed", "failed"}:
            return False
        self.awaiting_playback = False
        self.awaiting_playback_turn = None
        completed = status == "completed"
        self.playback_status = status
        if completed:
            self.conversation_until = (time.monotonic() if now is None else now) + self._settings.followup_s
            # The browser adds 700 ms of mute at the end of playback.
            self.last_tts = time.time() - self._settings.echo_mute_s + 0.7
        else:
            self.conversation_until = 0.0
        waiter = self._playback_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(status)
        return True

    async def _wait_for_playback(self, turn: int) -> str:
        if not self.awaiting_playback or not self._same_turn(turn, self.awaiting_playback_turn):
            return "none"
        waiter = self._playback_waiter
        if waiter is None:
            return "none"
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), self._settings.playback_ack_timeout_s)
        except asyncio.TimeoutError:
            if self.awaiting_playback and self._same_turn(turn, self.awaiting_playback_turn):
                self.awaiting_playback = False
                self.conversation_until = 0.0
                if not waiter.done():
                    waiter.cancel()
            return "timeout"
        finally:
            if self._playback_waiter is waiter:
                self._playback_waiter = None

    def _mark_playback_interrupted(self, turn: int) -> None:
        """Stop playback and mark a valid interruption for delayed follow-up."""
        if self.awaiting_playback and self._same_turn(turn, self.awaiting_playback_turn):
            self.awaiting_playback = False
            self.awaiting_playback_turn = None
            waiter = self._playback_waiter
            if waiter is not None and not waiter.done():
                waiter.set_result("interrupted")
        self.conversation_until = 0.0
        self._interrupted_turn = turn
        self.playback_status = "interrupted"
        # The browser has already stopped its current MP3, but the speaker can
        # still be ringing.  Keep the wake detector muted until that tail ends.
        self.last_tts = time.time()

    async def _open_interrupted_followup(self, turn: int) -> None:
        """Open the continuation window only after the speaker echo tail."""
        try:
            await asyncio.sleep(self._settings.echo_mute_s)
            if (
                self._interrupted_turn != turn
                or self.playback_status != "interrupted"
                or self.turn != turn
                or self.active_turn is not None
            ):
                return
            self.conversation_until = time.monotonic() + self._settings.followup_s
            log.info("turno %d: follow-up post-interruzione attivo per %.0fs", turn, self._settings.followup_s)
            await self.send_json(protocol.followup(seconds=self._settings.followup_s, interrupted=True))
        except asyncio.CancelledError:
            raise
        finally:
            if self._interrupted_followup_task is asyncio.current_task():
                self._interrupted_followup_task = None

    async def interrupt_current_turn(self, turn) -> bool:
        """Cancel one active turn after validating its exact current id.

        Cancellation closes an in-flight Hermes SSE response and the bounded TTS
        worker.  It is deliberately not converted into a retry: Hermes tools or
        approvals may already have caused side effects.
        """
        task = self._turn_task
        if (
            task is None
            or task.done()
            or not self._same_turn(turn, self.active_turn)
        ):
            return False

        current_turn = self.active_turn
        self._mark_playback_interrupted(current_turn)
        task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("errore durante l'interruzione del turno %s", current_turn)

        await self.send_json(protocol.interrupt_ack(turn=current_turn, status="interrupted", echo_tail_ms=int(self._settings.echo_mute_s * 1000)))
        await self.set_state("listening", turn=current_turn, interrupted=True)
        if self._interrupted_followup_task is not None:
            self._interrupted_followup_task.cancel()
            await asyncio.gather(self._interrupted_followup_task, return_exceptions=True)
        self._interrupted_followup_task = asyncio.create_task(
            self._open_interrupted_followup(current_turn)
        )
        return True

    def _reading_valid(self, turn):
        return not self.stop.is_set() and (
            self.active_turn is None or self._same_turn(turn, self.active_turn)
        ) and self._interrupted_turn != turn and (
            self.turn == 0 or self._same_turn(turn, self.turn)
        )

    def _new_reading(self, turn):
        async def started():
            if self._reading_valid(turn):
                self._begin_playback(turn)
                await self.set_state("speaking", turn=turn)

        def audio_sent():
            self.last_tts = time.time()

        return Reading(turn, self.send_json,
                       self._send_audio,
                       started, audio_sent, lambda: self._reading_valid(turn), settings=self._settings)

    async def _stream_hermes_speak(self, text: str, turn: int) -> tuple[str, bool, bool]:
        """Read streamed response text; never reissue a partially executed turn."""
        job = asyncio.current_task()
        owns_job = job not in self._jobs
        self._jobs.add(job)
        reading = self._new_reading(turn)
        self._reading = reading
        reading.start()
        stream_error = None

        async def on_approval(event):
            log.warning("turno %d: approvazione non disponibile nel canale vocale", turn)
            if self._reading_valid(turn):
                await self.send_json(protocol.approval(turn=turn, approval=event))
            raise ApprovalNotAvailable()

        try:
            try:
                result = await stream_hermes(
                    text, settings=self._settings, session_id=self.hermes_session_id,
                    on_delta=reading.feed, on_approval=on_approval,
                )
            except Exception as exc:
                stream_error = exc
            await reading.finish(flush=stream_error is None)
        finally:
            await reading.cancel()
            if owns_job:
                self._jobs.discard(job)
            if self._reading is reading:
                self._reading = None
        if stream_error is not None:
            if isinstance(stream_error, (ApprovalNotAvailable, HermesStreamStalled)):
                raise stream_error
            raise HermesStreamTurnError("stream Hermes non disponibile", reading.started) from None
        if self._reading_valid(turn):
            returned_id = getattr(result, "session_id", None)
            if returned_id:
                self.hermes_session_id = returned_id
        return str(result), reading.started, reading.error is not None

    async def set_state(self, state: str, **extra):
        turn = extra.get("turn")
        if self.stop.is_set() or (turn is not None and self.turn and
                                 not self._same_turn(turn, self.turn)):
            return
        self.state = state
        await self.send_json(protocol.state(state=state, **extra))

    async def manual_turn_start(self) -> bool:
        if self.stop.is_set():
            log.info("PTT: pressione ignorata, sessione arrestata")
            return False
        if self._manual_active:
            log.info("PTT: pressione ignorata, turno manuale già attivo")
            return False
        if self.state != "listening":
            log.info("PTT: pressione ignorata, stato %s", self.state)
            return False
        # A short prelude bridges frames between the press and recorder start;
        # a full 1.5 s would only add silence to a paid realtime stream.
        prelude = b"" if self.echo_muted(time.time(), self._settings.echo_mute_s) else (
            self.pre_roll()[-int(MANUAL_PREROLL_S * SAMPLE_RATE * 2):]
        )
        self._manual_active = True
        self._utterance_close.clear()
        self._manual_claim.set()
        self.last_wake = time.time()
        self.state = "waking"
        asyncio.create_task(self._on_wake(
            initial_pcm=prelude, local_wake_confirmed=False, manual=True,
        ))
        log.info("PTT: avvio turno manuale")
        return True

    def manual_turn_end(self) -> bool:
        if not self._manual_active:
            return False
        self._utterance_close.set()
        return True

    async def _on_wake(self, initial_pcm: bytes = b"",
                       local_wake_confirmed: bool = False, manual: bool = False):
        if self.stop.is_set():
            if manual:
                self._manual_claim.clear()
                self._manual_active = False
            return
        current_task = asyncio.current_task()
        self._jobs.add(current_task)
        try:
            async with self._turn_start_lock:
                previous = self._turn_task
                if previous is not None and previous is not current_task and not previous.done():
                    previous.cancel()
                    await asyncio.gather(previous, return_exceptions=True)
                if self.stop.is_set():
                    return
                self._turn_task = current_task
            try:
                await self._run_turn(initial_pcm, local_wake_confirmed, manual=manual)
            finally:
                if self._realtime is not None:
                    await self._realtime.close()
                    self._realtime = None
                if self._turn_task is current_task:
                    self._turn_task = None
                    self.active_turn = None
        finally:
            if manual:
                self._manual_claim.clear()
                self._manual_active = False
            self._jobs.discard(current_task)

    async def _run_turn(self, initial_pcm, local_wake_confirmed, manual=False):
        if not manual:
            self._utterance_close.clear()
        manual_fields = {"manual": True} if manual else {}
        followup_at_start = time.monotonic() < self.conversation_until and not self.awaiting_playback
        if self._playback_waiter is not None and not self._playback_waiter.done():
            self._playback_waiter.cancel()
        self.awaiting_playback = False
        self.awaiting_playback_turn = None
        self.turn += 1
        turn = self.turn
        self._turn_paid_s = 0.0
        self.active_turn = turn
        t_turn = time.time()
        # The window must be evaluated when speech starts, NOT after STT:
        # On slow CPUs transcription can take many seconds.
        await self.set_state("waking", turn=turn, followup=followup_at_start, **manual_fields)
        await self._clear_partial(turn)
        # pre-roll: the wake phrase has already passed while the trigger decides,
        # so restart from the last 1.5 s already held in the rotating buffer
        prelude = bytes(initial_pcm)
        if not prelude and not manual and self._settings.wake_provider == "whisper":
            prelude = self.pre_roll()
        # pause only if a reply was just played (TTS echo)
        if self.echo_muted(time.time(), 3.0):
            await asyncio.sleep(0.25)
        await self.set_state("recording", turn=turn, followup=followup_at_start, **manual_fields)
        realtime, realtime_start_failed = await _dispatch.open_stream(
            on_partial=lambda text: self._send_partial(text, turn), settings=self._settings,
        )
        self._realtime = realtime
        if realtime_start_failed:
            await self.send_json(protocol.stt_status(mode="local", turn=turn))
        try:
            pcm = await self._record_utterance(prelude, realtime=realtime)
        except Exception:
            if realtime is not None:
                await realtime.close()
            log.exception("errore registrazione")
            await self._clear_partial(turn)
            await self.set_state("listening")
            return
        if pcm is None or len(pcm) < int(self._settings.min_speech_s * SAMPLE_RATE):
            if realtime is not None:
                await realtime.close()
            await self._clear_partial(turn)
            await self.set_state("listening", note="niente da trascrivere")
            return

        try:
            saved = save_turn_audio(pcm)
            log.info("turno %d: audio per diagnostica %s", turn, saved.name)
        except Exception:
            log.exception("impossibile salvare il WAV diagnostico")
        await self.set_state("transcribing", turn=turn)
        try:
            text = await _dispatch.transcribe_turn(
                pcm, realtime, turn, followup_at_start,
                start_failed=realtime_start_failed, settings=self._settings, manual=manual,
            )
        except Exception as exc:
            log.exception("STT fallito")
            await self._clear_partial(turn)
            await self.set_state("listening", error=f"stt: {exc}")
            return
        if text is None:
            await self._clear_partial(turn)
            await self.set_state("listening")
            return
        if self._settings.wake_provider == "whisper":
            cmd = _dispatch.command_for_turn(text, followup_at_start, local_wake_confirmed, self._settings, manual=manual)
            if cmd is None:
                log.info("turno %d scartato (fuori dalla conversazione): %r", turn, text[:80])
                self.last_wake = time.time() + self._settings.ambient_pause_s - COOLDOWN_S
                await self._clear_partial(turn)
                await self.set_state("listening", note="non era per me")
                return
            if not cmd.strip():
                if manual:
                    log.info("turno %d: push-to-talk senza contenuto", turn)
                    await self._clear_partial(turn)
                    await self.set_state("listening", note="niente da trascrivere")
                    return
                log.info("turno %d: sveglia senza comando", turn)
                await self._clear_partial(turn)
                await self._speak("Sì?", turn)
                await self.set_state("listening", turn=turn)
                return
            text = cmd.strip()
            self.conversation_until = 0.0  # next turn only after playback
            log.info("comando %s: %r", "follow-up" if followup_at_start else "wake", text[:120])
            self._partial_turn = None
            await self.send_json(protocol.transcript(text=text, turn=turn, command=True))
        else:
            self._partial_turn = None
            await self.send_json(protocol.transcript(text=text, turn=turn))

        await self.set_state("thinking", turn=turn)
        stream_had_audio = False
        try:
            reply, stream_had_audio, tts_failed = await self._stream_hermes_speak(text, turn)
        except ApprovalNotAvailable:
            reply = APPROVAL_NOT_AVAILABLE_REPLY
            tts_failed = False
        except HermesStreamStalled:
            log.warning("turno %d: stream Hermes in stallo", turn)
            reply = AGENT_STALLED_REPLY
            tts_failed = False
        except HermesStreamTurnError as exc:
            # The stream may already have run tools or emitted audio.  Never
            # reissue this turn through ask_hermes; surface only a safe error.
            log.warning("turno %d: %s", turn, exc)
            reply = "Non riesco a completare la risposta."
            tts_failed = False
            stream_had_audio = exc.had_audio
        except Exception:
            log.exception("stream Hermes fallito")
            reply = "Non riesco a completare la risposta."
            tts_failed = False
        await self.send_json(protocol.reply(text=reply, turn=turn))
        if stream_had_audio:
            playback = await self._wait_for_playback(turn)
            if playback == "completed":
                await self.set_state("listening", turn=turn)
            elif playback == "failed":
                await self.set_state("listening", turn=turn, error="playback fallito")
            else:
                await self.set_state("listening", turn=turn, error="playback timeout")
        elif tts_failed:
            await self.set_state("listening", turn=turn, error="tts non disponibile")
        elif reply:
            # A successful stream with no usable MP3 is safe to handle with
            # the old single-response TTS path: Hermes is not called again.
            await self._speak(reply, turn)
            if self.awaiting_playback:
                playback = await self._wait_for_playback(turn)
                if playback == "completed":
                    await self.set_state("listening", turn=turn)
                elif playback == "failed":
                    await self.set_state("listening", turn=turn, error="playback fallito")
                else:
                    await self.set_state("listening", turn=turn, error="playback timeout")
            else:
                await self.set_state("listening", turn=turn)
        else:
            await self.set_state("listening", turn=turn)
        self._record_usage()
        log.info("turno %d completato in %.1fs", turn, time.time() - t_turn)
        return

    async def _speak(self, text: str, turn: int):
        job = asyncio.current_task()
        owns_job = job not in self._jobs
        self._jobs.add(job)
        reading = self._new_reading(turn)
        self._reading = reading
        try:
            await reading.speak(text)
        except TTSUnavailable:
            if reading.started:
                self.mark_playback_done(turn=turn, status="failed")
            log.exception("TTS fallito")
            await self.set_state("listening", turn=turn, error="tts non disponibile")
        finally:
            await reading.cancel()
            if owns_job:
                self._jobs.discard(job)
            if self._reading is reading:
                self._reading = None

    def _record_usage(self) -> None:
        """Best-effort accounting for the monthly usage report."""
        import datetime
        try:
            paid = float(getattr(self, "_turn_paid_s", 0.0) or 0.0)
            self.usage_ledger.record(
                datetime.date.today().isoformat(),
                turns=1,
                realtime_s=paid,
                local_turns=0 if paid else 1,
            )
        except Exception:
            log.exception("impossibile registrare l'usage")
        finally:
            self._turn_paid_s = 0.0
