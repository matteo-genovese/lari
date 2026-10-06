"""Manual turns bypass wake confirmation and close promptly on release."""
import asyncio
import json
from pathlib import Path
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari import protocol
from lari.config import load_settings
from lari.session import MANUAL_PREROLL_S, SAMPLE_RATE, Session
from lari import session as sessions
from lari.stt import dispatch


class PushToTalkProtocolTests(unittest.TestCase):
    def test_phases_and_malformed_controls(self):
        for phase in ("down", "up"):
            frame = {"type": "ptt", "phase": phase}
            self.assertEqual(protocol.parse_input(json.dumps(frame)), frame)
        for phase in (None, "invalid", [], {}):
            self.assertIsNone(protocol.parse_input(json.dumps({"type": "ptt", "phase": phase})))
        self.assertIsNone(protocol.parse_input('{"type":"ptt"}'))
        self.assertEqual(protocol.state(state="waking", manual=True)["manual"], True)
        self.assertNotIn("manual", protocol.state(state="waking"))

    def test_manual_command_needs_no_wake(self):
        for backend in ("elevenlabs_realtime", "whisper"):
            with self.subTest(backend=backend):
                settings = load_settings({"LARI_STT_BACKEND": backend})
                text = "  Tell me the weather tomorrow  "
                self.assertEqual(dispatch.command_for_turn(text, False, False, settings, manual=True), text.strip())
                self.assertIsNone(dispatch.command_for_turn(text, False, False, settings))

    def test_manual_decoding_skips_vosk_wake(self):
        settings = load_settings({"LARI_STT_BACKEND": "vosk"})
        with patch.object(dispatch, "vosk_wake") as wake, \
             patch.object(dispatch, "transcribe_vosk", return_value="tell me the weather"):
            self.assertEqual(dispatch.decode_utterance(np.ones(16000), False, settings=settings, manual=True),
                             "tell me the weather")
        wake.assert_not_called()

    def test_english_ui_and_ptt_controls(self):
        html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()
        for text in ('lang="en"', 'id="pttBtn"', 'id="tapBtn"', 'Tap to talk', 'Tap to stop', 'Hold to talk', 'Release to send',
                     '{type:"ptt", phase:"down"}', '{type:"ptt", phase:"up"}',
                     '"pointercancel"', '"lostpointercapture"', 'setPointerCapture',
                     'touch-action:none', 'turnIsManual ? "MANUAL"'):
            self.assertIn(text, html)
        for text in ("Nessuna conversazione", "Avvia ascolto", "Interrompi risposta", "in attesa",
                     "disconnesso", "Frase di sveglia", "Risposta audio", "Errore", "LOCALE"):
            self.assertNotIn(text, html)


class ManualTurnTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, backend="elevenlabs_realtime"):
        settings = load_settings({"LARI_STT_BACKEND": backend, "LARI_WAKE_PROVIDER": "whisper"})
        return Session(AsyncMock(), AsyncMock(), settings=settings, usage_ledger=Mock())

    async def test_start_refusals_and_one_claimed_turn(self):
        session = self.make_session()
        session._on_wake = AsyncMock()
        session.state = "recording"
        self.assertFalse(await session.manual_turn_start())
        session.state = "listening"
        session.stop.set()
        self.assertFalse(await session.manual_turn_start())
        session.stop.clear()
        session.recent = bytearray(np.full(24000, 5000, dtype=np.int16).tobytes())
        session._utterance_close.set()
        self.assertTrue(await session.manual_turn_start())
        self.assertEqual(session.state, "waking")
        self.assertTrue(session._manual_claim.is_set())
        self.assertFalse(session._utterance_close.is_set())
        session.state = "listening"
        self.assertFalse(await session.manual_turn_start())
        self.assertTrue(session.manual_turn_end())
        self.assertTrue(session.manual_turn_end())
        self.assertTrue(session._manual_active)
        await asyncio.sleep(0)
        session._on_wake.assert_awaited_once_with(
            initial_pcm=bytes(session.recent[-int(MANUAL_PREROLL_S * SAMPLE_RATE * 2):]),
            local_wake_confirmed=False, manual=True,
        )

    async def test_echo_tail_has_no_prelude_and_early_release_survives(self):
        session = self.make_session()
        session.last_tts = time.time()
        session.recent = bytearray(b"\x01\x00" * 24000)
        session._run_turn = AsyncMock()
        self.assertTrue(await session.manual_turn_start())
        self.assertTrue(session.manual_turn_end())
        await asyncio.sleep(0)
        session._run_turn.assert_awaited_once_with(b"", False, manual=True)
        self.assertTrue(session._utterance_close.is_set())
        self.assertFalse(session._manual_active)
        self.assertFalse(session._manual_claim.is_set())
        self.assertFalse(session.manual_turn_end())

    async def test_manual_claim_is_released_on_failure_and_stop(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                session = self.make_session()
                session._manual_active = True
                session._manual_claim.set()
                session._run_turn = AsyncMock(side_effect=RuntimeError("test"))
                if stopped:
                    session.stop.set()
                    await session._on_wake(manual=True)
                else:
                    with self.assertRaises(RuntimeError):
                        await session._on_wake(manual=True)
                self.assertFalse(session._manual_active)
                self.assertFalse(session._manual_claim.is_set())

    async def test_manual_turn_emits_wakeless_transcript(self):
        for backend in ("elevenlabs_realtime", "whisper"):
            with self.subTest(backend=backend):
                session = self.make_session(backend)
                session._record_utterance = AsyncMock(return_value=np.full(16000, 5000, dtype=np.int16))
                session._stream_hermes_speak = AsyncMock(return_value=("", False, False))
                with patch.object(dispatch, "open_stream", return_value=(None, False)), \
                     patch.object(dispatch, "transcribe_turn", return_value="Tell me the weather tomorrow") as stt, \
                     patch.object(sessions, "save_turn_audio", return_value=Path("fake.wav")):
                    await session._run_turn(b"", False, manual=True)
                stt.assert_awaited_once_with(
                    session._record_utterance.return_value, None, 1, False,
                    start_failed=False, settings=session._settings, manual=True,
                )
                frames = [call.args[0] for call in session._send_json.await_args_list]
                self.assertIn(protocol.transcript(text="Tell me the weather tomorrow", turn=1, command=True), frames)
                for state in ("waking", "recording"):
                    self.assertIn(protocol.state(state=state, turn=1, followup=False, manual=True), frames)
                session._record_utterance.assert_awaited_once_with(b"", realtime=None)
                session._stream_hermes_speak.assert_awaited_once_with("Tell me the weather tomorrow", 1)

    async def test_release_closes_recorder_before_silence_timeout(self):
        session = self.make_session()
        session._manual_active = True
        speech = np.full(8000, 5000, dtype=np.int16).tobytes()
        session.recv_queue.put_nowait(speech)
        recording = asyncio.create_task(session._record_utterance())
        deadline = time.monotonic() + 1
        while not session.recv_queue.empty() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        self.assertTrue(session.recv_queue.empty())
        started = time.monotonic()
        self.assertTrue(session.manual_turn_end())
        pcm = await asyncio.wait_for(recording, 0.6)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.6)
        self.assertLess(elapsed, session._settings.silence_end_s)
        self.assertEqual(pcm.tobytes(), speech)
        # A release received before recording starts must leave queued silence alone.
        silence = bytes(80000)
        session.recv_queue.put_nowait(silence)
        pcm = await session._record_utterance(prelude=speech)
        self.assertEqual(pcm.tobytes(), speech)
        self.assertEqual(session.recv_queue.get_nowait(), silence)

    async def test_release_preserves_all_collected_audio(self):
        session = self.make_session()
        session._manual_active = True
        self.assertTrue(session.manual_turn_end())
        prelude = np.concatenate((np.full(8000, 5000, dtype=np.int16), np.zeros(8000, dtype=np.int16))).tobytes()
        pcm = await session._record_utterance(prelude)
        self.assertEqual(pcm.tobytes(), prelude)

    async def test_wake_gate_cannot_preempt_manual_claim(self):
        session = self.make_session()
        session._manual_claim.set()
        with patch('lari.wake.runtime.vosk_wake') as gate, \
             patch('lari.wake.runtime.confirm_candidate') as confirm:
            self.assertFalse(session._launch_local_candidate(b"\x01\x00" * 16000))
        gate.assert_not_called()
        confirm.assert_not_called()
        self.assertEqual(session._candidate_from_queue(b"prelude", True), b"prelude")
        self.assertEqual(session.state, "listening")
