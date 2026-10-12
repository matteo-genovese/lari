"""Offline regressions for TTS retries and consecutive segment failures."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from lari.config import get_settings
from lari import tts as voice


class TTSRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_then_audio_retries_once(self):
        settings = get_settings()
        with patch.object(voice, "TTS_RETRY_BACKOFF_S", (0.0, 0.0, 0.0)), \
             patch.object(voice, "synthesize_once", new_callable=AsyncMock,
                          side_effect=[RuntimeError("provider detail"), b"audio"]) as synth:
            self.assertEqual(await voice.tts("Ciao.", settings), b"audio")
        self.assertEqual(synth.await_count, 2)
        synth.assert_awaited_with("Ciao.", settings.tts_voice)

    async def test_all_attempts_fail_with_safe_public_error(self):
        with patch.object(voice, "TTS_RETRY_BACKOFF_S", (0.0, 0.0, 0.0)), \
             patch.object(voice, "synthesize_once", new_callable=AsyncMock,
                          side_effect=RuntimeError("provider detail")) as synth:
            with self.assertRaisesRegex(voice.TTSUnavailable, "^tts non disponibile$") as error:
                await voice.tts("Ciao.", get_settings())
        self.assertEqual(synth.await_count, voice.TTS_ATTEMPTS)
        self.assertIsNone(error.exception.__cause__)

    async def test_empty_audio_retries(self):
        with patch.object(voice, "TTS_RETRY_BACKOFF_S", (0.0, 0.0, 0.0)), \
             patch.object(voice, "synthesize_once", new_callable=AsyncMock,
                          side_effect=[b"", b"audio"]) as synth:
            self.assertEqual(await voice.tts("Ciao.", get_settings()), b"audio")
        self.assertEqual(synth.await_count, 2)

    async def test_cancellation_during_backoff_propagates(self):
        sleeping = asyncio.Event()

        async def backoff(delay):
            sleeping.set()
            await asyncio.Event().wait()

        with patch.object(voice, "synthesize_once", new_callable=AsyncMock,
                          side_effect=RuntimeError("provider detail")) as synth, \
             patch.object(voice.asyncio, "sleep", side_effect=backoff) as sleep:
            task = asyncio.create_task(voice.tts("Ciao.", get_settings()))
            try:
                await asyncio.wait_for(sleeping.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(synth.await_count, 1)
        sleep.assert_awaited_once_with(voice.TTS_RETRY_BACKOFF_S[0])


class ReadingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def run_reading(self, outcomes, text):
        messages = []
        audio = []

        async def send_json(data):
            messages.append(data)

        async def send_audio(chunk):
            audio.append(chunk)

        reading = voice.Reading(1, send_json, send_audio, AsyncMock(),
                                Mock(), lambda: True, settings=get_settings())
        with patch.object(voice, "tts", new_callable=AsyncMock,
                          side_effect=outcomes) as synth:
            reading.start()
            try:
                await reading.feed(text)
                await reading.finish()
            finally:
                await reading.cancel()
        self.assertTrue(reading.worker.done())
        self.assertTrue(reading.queue.empty())
        return reading, messages, audio, synth

    async def test_failed_segment_does_not_skip_next_audio(self):
        reading, messages, audio, synth = await self.run_reading(
            [voice.TTSUnavailable("tts non disponibile"), b"second"],
            "Prima. Seconda.")
        self.assertEqual(audio, [b"second"])
        self.assertIsNotNone(reading.error)
        self.assertEqual(synth.await_count, 2)
        self.assertEqual(messages, [
            {"type": "audio_start", "turn": 1},
            {"type": "audio_chunk", "turn": 1, "seq": 0},
            {"type": "audio_end", "turn": 1},
        ])

    async def test_two_consecutive_failures_set_error_and_drain(self):
        reading, messages, audio, synth = await self.run_reading(
            [voice.TTSUnavailable("tts non disponibile") for _ in range(2)],
            "Prima. Seconda. Terza.")
        self.assertIsInstance(reading.error, voice.TTSUnavailable)
        self.assertEqual(str(reading.error), "tts non disponibile")
        self.assertEqual(synth.await_count, 2)
        self.assertEqual(audio, [])
        self.assertEqual(messages, [])

    async def test_failure_between_successes_keeps_delivering(self):
        reading, _, audio, synth = await self.run_reading(
            [voice.TTSUnavailable("tts non disponibile"), b"second",
             voice.TTSUnavailable("tts non disponibile"), b"fourth"],
            "Prima. Seconda. Terza. Quarta.")
        self.assertIsNotNone(reading.error)
        self.assertEqual(audio, [b"second", b"fourth"])
        self.assertEqual(synth.await_count, 4)
