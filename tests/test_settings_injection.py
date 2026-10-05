"""Injected configuration affects behavior; only immutable models/config are shared."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari.config import get_settings, load_settings
from lari.session import Session
from lari.stt import dispatch, local, providers, realtime
from lari.wake import detector, runtime


class SettingsInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_followup_and_echo_are_observable(self):
        process = get_settings()
        settings = replace(process, followup_s=process.followup_s + 13,
                           echo_mute_s=process.echo_mute_s + 4)
        send = AsyncMock()
        session = Session(None, send, settings=settings)
        session.turn = 1
        session._begin_playback(1)
        with patch('lari.session.time.monotonic', return_value=100), \
             patch('lari.session.time.time', return_value=200):
            self.assertTrue(await session.playback_completed(1))
        self.assertEqual(send.call_args.args[0],
                         {'type': 'followup', 'seconds': settings.followup_s})
        self.assertEqual(session.conversation_until, 100 + settings.followup_s)
        self.assertAlmostEqual(session.last_tts, 200 - settings.echo_mute_s + .7)
        await session.disconnect()

    async def test_custom_minimum_speech_rejects_short_turn_before_stt(self):
        process = get_settings()
        settings = replace(process, min_speech_s=process.min_speech_s + 2)
        session = Session(None, AsyncMock(), settings=settings)
        session._record_utterance = AsyncMock(return_value=np.ones(
            int((process.min_speech_s + 1) * 16000), dtype=np.int16))
        with patch.object(dispatch, 'open_stream', return_value=(None, False)) as stream, \
             patch.object(dispatch, 'transcribe_turn') as transcribe:
            await session._on_wake(local_wake_confirmed=True)
        transcribe.assert_not_called()
        self.assertEqual(session._send_json.call_args.args[0]['note'], 'niente da trascrivere')
        self.assertIs(stream.call_args.kwargs['settings'], settings)
        await session.disconnect()

    async def test_custom_audio_threshold_and_idle_abort(self):
        process = get_settings()
        # Below the custom threshold but above the process threshold.
        threshold = max(process.vad_min_rms, 500 * process.vad_noise_mult)
        settings = replace(process, vad_min_rms=threshold + 2000, idle_abort_s=.05)
        session = Session(None, AsyncMock(), settings=settings)
        self.assertEqual(session.update_noise_floor(threshold + 100), settings.vad_min_rms)
        session.recv_queue.put(np.full(1600, threshold + 100, dtype=np.int16).tobytes())
        session.recv_queue.put(None)
        self.assertIsNone(await session._record_utterance())
        await session.disconnect()

    async def test_worker_uses_injected_echo_mute(self):
        settings = replace(get_settings(), echo_mute_s=10)
        session = Session(None, AsyncMock(), settings=settings)
        session.last_tts = 95
        session.conversation_until = float('inf')
        with patch.object(runtime.time, 'time', return_value=100), \
             patch.object(runtime.asyncio, 'run_coroutine_threadsafe') as schedule:
            self.assertFalse(session._launch_local_candidate(b'pcm'))
        schedule.assert_not_called()
        await session.disconnect()

    async def test_shared_settings_and_private_connection_state(self):
        settings = load_settings({})
        first = Session(None, AsyncMock(), settings=settings)
        second = Session(None, AsyncMock(), settings=settings)
        self.assertIs(first._settings, second._settings)
        with self.assertRaises(FrozenInstanceError):
            settings.followup_s = 99
        with self.assertRaises(FrozenInstanceError):
            settings.wake_config.phrase = 'changed'
        first.turn = 4
        first.state = 'speaking'
        first.conversation_until = 123
        first._begin_playback(4)
        first._realtime = Mock(close=AsyncMock())
        first._reading = first._new_reading(4)
        second._reading = second._new_reading(0)
        self.assertIsNot(first._reading.queue, second._reading.queue)
        self.assertIsNot(first._reading.buffer, second._reading.buffer)
        self.assertIsNone(second._realtime)
        self.assertEqual((second.turn, second.state, second.conversation_until), (0, 'listening', 0))
        self.assertFalse(second.awaiting_playback)
        await first.disconnect()
        self.assertFalse(second.stop.is_set())
        await second.disconnect()

    async def test_tts_receives_custom_voice_and_limits(self):
        settings = replace(get_settings(), tts_voice='custom-voice', stream_tts_queue_max=2,
                           stream_sentence_max_chars=7, stream_text_max_chars=11)
        session = Session(None, AsyncMock(), settings=settings)
        reading = session._new_reading(1)
        self.assertEqual(reading.queue.maxsize, 2)
        self.assertEqual(reading.buffer.feed('uno due tre'), ['uno due'])
        communicate = Mock(return_value=Mock())
        async def chunks():
            yield {'type': 'audio', 'data': b'mp3'}
        communicate.return_value.stream = chunks
        with patch.dict(sys.modules, {'edge_tts': Mock(Communicate=communicate)}):
            self.assertEqual(await reading._synthesize('ciao'), b'mp3')
        communicate.assert_called_once_with('ciao', 'custom-voice')
        with self.assertRaises(RuntimeError):
            reading.buffer.feed('!')
        await session.disconnect()

    async def test_dispatch_and_realtime_bias_use_custom_settings(self):
        settings = load_settings({'LARI_STT_BACKEND': 'groq', 'LARI_WAKE_PHRASE': 'hey luna',
                                  'GROQ_API_KEY': 'test'})
        with patch.object(providers, '_post_multipart', return_value={'text': 'hey luna, ciao'}) as post:
            self.assertEqual(dispatch.stt_transcribe(np.zeros(100, dtype=np.int16), settings), 'hey luna, ciao')
        self.assertIn(('prompt', settings.wake_config.style_prompt), post.call_args.kwargs['data_tuples'])
        from urllib.parse import parse_qs, urlsplit
        query = parse_qs(urlsplit(realtime.realtime_url(settings.wake_config)).query)
        self.assertEqual(query['keyterms'], list(settings.wake_config.realtime_keyterms))
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(settings, usage_ledger=Path(directory)/'turns.json',
                               realtime_usage_file=Path(directory)/'audio.json', realtime_daily_seconds=3)
            session = Session(None, AsyncMock(), settings=settings)
            self.assertEqual(session.usage_ledger.path, settings.usage_ledger)
            budgets = [realtime.realtime_daily_budget(settings) for _ in range(2)]
            self.assertTrue(budgets[0].reserve(2))
            self.assertEqual(budgets[1].remaining(), 1)
            self.assertFalse(budgets[1].reserve(2))
            await session.disconnect()


class SharedModelTests(unittest.TestCase):
    def test_replaced_model_path_reaches_public_wake_config(self):
        settings = replace(load_settings({}), vosk_model_dir=Path('/models/custom'))
        self.assertEqual(settings.wake_config.model_dir, '/models/custom')


    def test_heavy_models_are_shared_by_model_and_path(self):
        with patch.dict(local._stt, {}, clear=True), \
             patch.dict(detector._vosk_models, {}, clear=True), \
             patch.dict(sys.modules, {'faster_whisper': Mock(WhisperModel=Mock(side_effect=lambda *a, **k: object())),
                                     'vosk': Mock(Model=Mock(side_effect=lambda *a: object()))}):
            self.assertIs(local.get_stt('base'), local.get_stt('base'))
            self.assertIsNot(local.get_stt('base'), local.get_stt('small'))
            self.assertIs(detector.get_vosk('/models/a'), detector.get_vosk('/models/a'))
            self.assertIsNot(detector.get_vosk('/models/a'), detector.get_vosk('/models/b'))
