"""Regression tests for wake recognition on real STT variants."""
import unittest
import tempfile
import asyncio
import os
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
import wave
from unittest.mock import AsyncMock, Mock, patch

import server
from server import WAKE_RE, wake_command, save_turn_audio, Session, resolve_command


class WakeTests(unittest.TestCase):
    def test_configured_aliases_match_only_explicit_prefixes(self):
        env = os.environ.copy()
        env['LARI_WAKE_PHRASE'] = 'Hey Lari'
        env['LARI_WAKE_ALIASES'] = 'Ehi Lari,Ehi Lika,Hey Nic,Ehi Nick,Hey Nico'
        result = subprocess.run(
            [sys.executable, '-c', "import server; print([server.wake_command(p) for p in ('Hey Lari, buongiorno', 'Ehi Lika, buongiorno', 'Hey Nic, buongiorno', 'Ehi Nick, buongiorno', 'Hey, Nico. Buon appetito.', 'Hey Other, buongiorno')])"],
            capture_output=True, text=True, check=True, env=env,
        )
        self.assertEqual(result.stdout.strip(), "['buongiorno', 'buongiorno', 'buongiorno', 'buongiorno', 'Buon appetito.', None]")

    def test_configured_aliases_are_included_in_vosk_grammar(self):
        with patch.object(server, 'WAKE_PHRASE', 'Hey Lari'), \
             patch.object(server, 'WAKE_ALIASES', ('Ehi Lari', 'Ehi Lika', 'Hey Nic')), \
             patch.object(server, '_wake_phrases', ('Hey Lari', 'Ehi Lari', 'Ehi Lika', 'Hey Nic')), \
             patch.object(server, 'get_vosk', return_value=object()), \
             patch('vosk.KaldiRecognizer') as recognizer:
            recognizer.return_value.FinalResult.return_value = '{"text":"hey nic"}'
            with patch.object(server, 'WAKE_RE', re.compile(r'^(?:hey lari|ehi lari|ehi lika|hey nic)\b', re.I)):
                self.assertTrue(server.vosk_wake(np.zeros(16000, dtype=np.int16)))
            grammar = recognizer.call_args.args[2]
            self.assertIn('hey nic', grammar)
            self.assertIn('ehi lika', grammar)

    def test_default_italian_wake_phrase(self):
        self.assertEqual(wake_command("Ehi Lari, che tempo fa?"), "che tempo fa?")
        self.assertIsNone(wake_command("Hey Other, che tempo fa?"))
        self.assertIsNone(wake_command("Lari, che tempo fa?"))

    def test_configured_english_default_phrase(self):
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["LARI_LANGUAGE"] = "en"
        env.pop("LARI_WAKE_PHRASE", None)
        result = subprocess.run(
            [sys.executable, "-c", "import server; print(server.WAKE_PHRASE, server.wake_command('Hey Lari, hello'))"],
            check=True, capture_output=True, text=True, env=env,
        )
        self.assertIn("Hey Lari hello", result.stdout)

    def test_configured_custom_wake_phrase(self):
        import os
        import subprocess
        import sys
        env = os.environ.copy()
        env["LARI_LANGUAGE"] = "en"
        env["LARI_WAKE_PHRASE"] = "Computer Lare"
        result = subprocess.run(
            [sys.executable, "-c", "import server; print(server.wake_command('Computer Lare, hello'))"],
            check=True, capture_output=True, text=True, env=env,
        )
        self.assertIn("hello", result.stdout)


    def test_does_not_trigger_on_a_name_in_background(self):
        self.assertIsNone(wake_command("I spoke with Lari about work."))

    def test_followup_without_wake_only_during_conversation_window(self):
        phrase = 'E tu cosa mi consigli?'
        self.assertIsNone(resolve_command(phrase, conversation_until=0, now=100))
        self.assertEqual(resolve_command(phrase, conversation_until=130, now=100), phrase)
        self.assertIsNone(resolve_command(phrase, conversation_until=99, now=100))
        self.assertEqual(resolve_command('Ehi Lari, che tempo fa?', conversation_until=0, now=100),
                         'che tempo fa?')

    def test_playback_ack_opens_followup_window(self):
        async def sender(_):
            pass
        session = Session(None, sender)
        session.awaiting_playback = True
        session.mark_playback_done(now=100.0)
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 100.0 + 30.0)
        session.mark_playback_done(now=200.0)  # A duplicate playback acknowledgement must not extend the session
        self.assertEqual(session.conversation_until, 130.0)

    def test_hermes_session_id_belongs_to_the_websocket_session(self):
        async def sender(_):
            pass
        first = Session(None, sender)
        second = Session(None, sender)
        first.hermes_session_id = 'first-transcript'
        second.hermes_session_id = 'second-transcript'
        self.assertNotEqual(first.hermes_session_id, second.hermes_session_id)

    def test_preserves_quiet_wake_word_in_preroll(self):
        async def sender(_):
            pass
        session = Session(None, sender)
        # The wake phrase in pre-roll may be quieter than the command. Trimming must
        # preserve it even when it is below the VAD energy threshold.
        wake = np.full(16000, 250, dtype=np.int16)
        loud = np.full(16000, 3000, dtype=np.int16)
        for offset in range(0, len(loud), 1600):
            session.recv_queue.put_nowait(loud[offset:offset+1600].tobytes())
        session.recv_queue.put_nowait(None)
        audio = asyncio.run(session._record_utterance(wake.tobytes()))
        self.assertIsNotNone(audio)
        np.testing.assert_array_equal(audio[:len(wake)], wake)

    def test_long_request_survives_a_natural_pause(self):
        async def sender(_):
            pass
        session = Session(None, sender)
        voice = np.full(16000, 3000, dtype=np.int16)
        silence = np.zeros(16000, dtype=np.int16)
        # Keep 15 seconds of speech and a 2-second pause without truncating the
        # utterance at a fixed short limit. A long final pause should close it.
        for chunk in [voice] * 15 + [silence] * 2 + [voice] * 2 + [silence] * 3:
            session.recv_queue.put_nowait(chunk.tobytes())
        audio = asyncio.run(session._record_utterance())
        self.assertIsNotNone(audio)
        self.assertGreaterEqual(len(audio), 19 * 16000)
        np.testing.assert_array_equal(audio[17 * 16000:18 * 16000], voice)

    def test_turn_audio_saved_privately_and_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            pcm = np.arange(1600, dtype=np.int16)
            for i in range(7):
                save_turn_audio(pcm, directory=directory, name=f'turn_{i}.wav', keep=3)
            files = sorted(directory.glob('turn_*.wav'))
            self.assertEqual([p.name for p in files],
                             ['turn_4.wav', 'turn_5.wav', 'turn_6.wav'])
            with wave.open(str(files[-1]), 'rb') as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.readframes(1600), pcm.tobytes())
            self.assertTrue(all(p.stat().st_mode & 0o077 == 0 for p in files))

    def test_partial_transcript_is_cleared_when_a_turn_is_discarded(self):
        sent = []

        async def send(message):
            sent.append(message)

        session = Session(None, send)
        session._partial_turn = 4
        asyncio.run(session._clear_partial(5))
        asyncio.run(session._clear_partial(5))

        self.assertEqual(sent, [{
            'type': 'partial_transcript', 'text': '', 'turn': 4,
        }])

    def test_recording_sends_quiet_chunks_to_realtime_contiguously(self):
        async def send(_):
            pass

        class Recorder:
            def __init__(self):
                self.chunks = []

            async def send_audio(self, chunk):
                self.chunks.append(chunk)
                return True

        session = Session(None, send)
        quiet = np.zeros(1600, dtype=np.int16).tobytes()
        voice = np.full(1600, 3000, dtype=np.int16).tobytes()
        chunks = [quiet, voice, quiet]
        for chunk in chunks:
            session.recv_queue.put_nowait(chunk)
        session.recv_queue.put_nowait(None)
        realtime = Recorder()

        audio = asyncio.run(session._record_utterance(realtime=realtime))

        self.assertIsNotNone(audio)
        self.assertEqual(realtime.chunks, chunks)


class LocalWakeGateTests(unittest.TestCase):
    def test_ambient_candidate_does_not_schedule_realtime_turn(self):
        async def send(_):
            pass

        session = Session(None, send)
        pcm = np.full(1600, 3000, dtype=np.int16).tobytes()
        with patch.object(server, "vosk_wake", return_value=False) as gate, \
             patch.object(server.asyncio, "run_coroutine_threadsafe") as schedule, \
             patch.object(server.stt_backends.RealtimeScribe, "connect", new_callable=AsyncMock) as connect:
            self.assertFalse(session._launch_local_candidate(pcm))

        gate.assert_called_once()
        schedule.assert_not_called()
        connect.assert_not_awaited()

    def test_confirmed_candidate_schedules_one_turn_with_audio_seed(self):
        async def send(_):
            pass

        session = Session(None, send)
        pcm = np.full(1600, 3000, dtype=np.int16).tobytes()
        scheduled = []

        def schedule(coro, _loop):
            scheduled.append(coro)
            coro.close()
            return Mock()

        with (
            patch.object(server, "vosk_wake", return_value=True) as gate,
            patch.object(server, "transcribe", return_value="Ehi Lari, ciao") as local_asr,
            patch.object(server.asyncio, "run_coroutine_threadsafe", side_effect=schedule),
        ):
            self.assertTrue(session._launch_local_candidate(pcm))

        gate.assert_called_once()
        local_asr.assert_called_once()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(session.state, "waking")

    def test_vosk_grammar_false_positive_is_rejected_by_local_asr(self):
        async def send(_):
            pass

        session = Session(None, send)
        pcm = np.full(1600, 3000, dtype=np.int16).tobytes()
        with (
            patch.object(server, "vosk_wake", return_value=True),
            patch.object(server, "transcribe", return_value="Che tempo fa domani a Roma?"),
            patch.object(server.asyncio, "run_coroutine_threadsafe") as schedule,
            patch.object(server.stt_backends.RealtimeScribe, "connect", new_callable=AsyncMock) as connect,
        ):
            self.assertFalse(session._launch_local_candidate(pcm))

        schedule.assert_not_called()
        connect.assert_not_awaited()


class RealtimeAfterLocalWakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_connects_once_only_after_local_confirmation(self):
        sent = []

        async def send(message):
            sent.append(message)

        session = Session(None, send)
        seed = np.full(1600, 3000, dtype=np.int16).tobytes()
        session._record_utterance = AsyncMock(return_value=np.ones(16000, dtype=np.int16))
        session._ask_hermes = AsyncMock(return_value="ok")
        session._speak = AsyncMock()
        realtime = AsyncMock()
        with (
            patch.object(server, "STT_BACKEND", server.stt_backends.REALTIME_BACKEND),
            patch.object(server, "WAKE_PROVIDER", "whisper"),
            patch.object(server, "AGENT_BACKEND", "deepseek"),
            patch.object(server.stt_backends.RealtimeScribe, "connect", new_callable=AsyncMock,
                         return_value=realtime) as connect,
            patch.object(server, "_transcribe_realtime_or_batch", new_callable=AsyncMock,
                         return_value=("Ehi Lari, aggiunta", False)),
            patch.object(server, "save_turn_audio", return_value=Path("fake.wav")),
        ):
            await session._on_wake(seed, local_wake_confirmed=True)

        connect.assert_awaited_once()
        session._record_utterance.assert_awaited_once()
        self.assertEqual(session._record_utterance.call_args.args[0], seed)
        self.assertTrue(any(m.get("type") == "transcript" for m in sent))


if __name__ == '__main__':
    unittest.main()
