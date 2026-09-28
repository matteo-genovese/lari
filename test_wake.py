"""Regression tests for wake recognition on real STT variants."""
import unittest
import tempfile
import asyncio
from pathlib import Path
import numpy as np
import wave

from server import WAKE_RE, wake_command, save_turn_audio, Session, resolve_command


class WakeTests(unittest.TestCase):
    def test_real_phone_transcript_is_not_discarded(self):
        text = ('E Nic, che ti ando farò mani a Roma, voi andare a first show, '
                'quindi il centro con me ci ha le augure.')
        self.assertIsNotNone(WAKE_RE.search(text))
        self.assertEqual(wake_command(text), text.split(',', 1)[1].strip())

    def test_wake_variants_from_asr_are_accepted(self):
        # I modelli ASR rendono «Ehi Nic» come "Nica"/"Nici": il gate deve
        # accettarle (fino a 2 lettere spurie) ma non nomi lunghi come Nicola/Nicole.
        self.assertEqual(wake_command('Ehi Nica, che tempo fa?'), 'che tempo fa?')
        self.assertEqual(wake_command('Ehi Nici, come the weather?'), 'come the weather?')
        self.assertEqual(wake_command('Hey Nick, dimmi.'), 'dimmi.')
        self.assertIsNone(wake_command('Hey Nicola, che tempo fa?'))
        self.assertIsNone(wake_command("Non c'è Nicola stasera."))

    def test_does_not_trigger_on_nickname_in_background(self):
        self.assertIsNone(wake_command('Ho parlato con Nic di lavoro.'))

    def test_standard_hey_nic(self):
        self.assertEqual(wake_command('Hey Nic, che tempo fa domani a Roma?'),
                         'che tempo fa domani a Roma?')

    def test_followup_without_wake_only_during_conversation_window(self):
        phrase = 'E tu cosa mi consigli?'
        self.assertIsNone(resolve_command(phrase, conversation_until=0, now=100))
        self.assertEqual(resolve_command(phrase, conversation_until=130, now=100), phrase)
        self.assertIsNone(resolve_command(phrase, conversation_until=99, now=100))
        self.assertEqual(resolve_command('Hey Nic, che tempo fa?', conversation_until=0, now=100),
                         'che tempo fa?')

    def test_playback_ack_opens_followup_window(self):
        async def sender(_):
            pass
        session = Session(None, sender)
        session.awaiting_playback = True
        session.mark_playback_done(now=100.0)
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 100.0 + 30.0)
        session.mark_playback_done(now=200.0)  # ack duplicato non estende la sessione
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
        # Il wake nel pre-roll è più piano del comando; il trim non deve
        # eliminarlo, anche se non supera la soglia energetica del VAD.
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
        # 15 s parlati + 2 s di pausa + seconda parte: non finire a 12 s
        # né durante la pausa. Una pausa finale sufficientemente lunga chiude.
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


if __name__ == '__main__':
    unittest.main()
