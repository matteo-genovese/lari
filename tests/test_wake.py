"""Regression tests for wake recognition on real STT variants."""
from lari.config import get_settings
import unittest
import tempfile
import asyncio
from pathlib import Path
import numpy as np
import wave

from functools import partial

from lari.wake.detector import wake_command as _wake_command, resolve_command as _resolve_command, resolve_vosk_command
from lari.audio import save_turn_audio
from lari.session import Session
from lari.wake.config import build_wake_config

# Parity suite for the tuned live phrase: every observed variant must survive
# the derived wake config unchanged.
LARI = build_wake_config("hey lari")
wake_command = partial(_wake_command, cfg=LARI)
resolve_command = partial(_resolve_command, cfg=LARI)


class WakeTests(unittest.TestCase):
    def test_real_phone_transcript_is_not_discarded(self):
        text = ('E Lari, mi dica che tempo fara domani in centro, '
                'che devo uscire senza ombrellone.')
        self.assertIsNotNone(LARI.command_re.search(text))
        self.assertEqual(wake_command(text, cfg=get_settings().wake_config), text.split(',', 1)[1].strip())

    def test_wake_variants_from_asr_are_accepted(self):
        # ASR renders the wake's final vowel across its confusion set
        # (lari -> lare/lary) and a consonant core with up to two spurious
        # letters (ric -> rica/rici/rick), never long lookalikes.
        self.assertEqual(wake_command('Ehi Lare, che tempo fa?', cfg=get_settings().wake_config), 'che tempo fa?')
        self.assertEqual(wake_command('Ehi Lary, come the weather?', cfg=get_settings().wake_config), 'come the weather?')
        self.assertIsNone(wake_command('Hey Larice, che tempo fa?', cfg=get_settings().wake_config))
        self.assertIsNone(wake_command("Non c'è Larice stasera.", cfg=get_settings().wake_config))
        ric = build_wake_config('hey ric')
        self.assertEqual(ric.command('Ehi Rica, che tempo fa?'), 'che tempo fa?')
        self.assertEqual(ric.command('Ehi Rici, dimmi.'), 'dimmi.')
        self.assertEqual(ric.command('Hey Rick, dimmi.'), 'dimmi.')
        self.assertIsNone(ric.command('Hey Riccardo, che tempo fa?'))

    def test_does_not_trigger_on_nickname_in_background(self):
        self.assertIsNone(wake_command('Ho parlato con Lari di lavoro.', cfg=get_settings().wake_config))

    def test_standard_wake(self):
        self.assertEqual(wake_command('Hey Lari, che tempo fa domani a Roma?', cfg=get_settings().wake_config),
                         'che tempo fa domani a Roma?')

    def test_followup_without_wake_only_during_conversation_window(self):
        phrase = 'E tu cosa mi consigli?'
        self.assertIsNone(resolve_command(phrase, conversation_until=0, now=100, cfg=get_settings().wake_config))
        self.assertEqual(resolve_command(phrase, conversation_until=130, now=100, cfg=get_settings().wake_config), phrase)
        self.assertIsNone(resolve_command(phrase, conversation_until=99, now=100, cfg=get_settings().wake_config))
        self.assertEqual(resolve_command('Hey Lari, che tempo fa?', conversation_until=0, now=100, cfg=get_settings().wake_config),
                         'che tempo fa?')

    def test_monologue_before_the_wake_is_never_the_command(self):
        cfg = build_wake_config('hey lari')
        self.assertEqual(
            resolve_vosk_command('Bla bla bla bla bla. Ehi Lari, che ore sono?',
                                 wake=True, followup=False, cfg=cfg),
            'che ore sono?')
        self.assertEqual(
            resolve_vosk_command('Bla bla bla bla bla. Ehi Lari.',
                                 wake=True, followup=False, cfg=cfg),
            '')
        # A wake dropped by free-form ASR still leaves the request intact.
        self.assertEqual(
            resolve_vosk_command('che ore sono?', wake=True, followup=False, cfg=cfg),
            'che ore sono?')

    def test_playback_ack_opens_followup_window(self):
        async def sender(_):
            pass
        session = Session(None, sender, settings=get_settings())
        session.awaiting_playback = True
        session.mark_playback_done(now=100.0)
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 100.0 + 30.0)
        session.mark_playback_done(now=200.0)  # a duplicate ack does not extend the session
        self.assertEqual(session.conversation_until, 130.0)

    def test_hermes_session_id_belongs_to_the_websocket_session(self):
        async def sender(_):
            pass
        first = Session(None, sender, settings=get_settings())
        second = Session(None, sender, settings=get_settings())
        first.hermes_session_id = 'first-transcript'
        second.hermes_session_id = 'second-transcript'
        self.assertNotEqual(first.hermes_session_id, second.hermes_session_id)

    def test_preserves_quiet_wake_word_in_preroll(self):
        async def sender(_):
            pass
        session = Session(None, sender, settings=get_settings())
        # The wake in the pre-roll is quieter than the command; trimming must
        # never remove it even when it stays below the VAD energy threshold.
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
        session = Session(None, sender, settings=get_settings())
        voice = np.full(16000, 3000, dtype=np.int16)
        silence = np.zeros(16000, dtype=np.int16)
        # 15s spoken + 2s pause + second part: never end at 12s nor inside
        # the pause. A long enough trailing pause closes the utterance.
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

        session = Session(None, send, settings=get_settings())
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

        session = Session(None, send, settings=get_settings())
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


if __name__ == '__main__':
    unittest.main()
