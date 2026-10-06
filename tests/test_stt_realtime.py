"""Offline tests for the bounded ElevenLabs Scribe v2 Realtime path."""
from lari.config import get_settings
import asyncio
import json
import logging
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari.stt import dispatch
from lari.stt import providers
from lari.stt import realtime as stt_backends
from lari.config import load_settings


def _reserve_budget_in_process(path, barrier, results):
    budget = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0, settings=get_settings())
    barrier.wait()
    results.put(budget.reserve(0.75))


class FakeWebSocket:
    def __init__(self, events=()):
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(json.dumps(event))
        self.sent = []
        self.closed = False

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("commit"):
            await self.events.put({
                "message_type": "committed_transcript",
                "text": "Ehi Lari, che tempo fa a Roma?",
            })

    async def recv(self):
        event = await self.events.get()
        if isinstance(event, str):
            return event
        return json.dumps(event)

    async def close(self):
        self.closed = True


class RealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_audio_reserves_budget_outside_event_loop(self):
        calls = []
        loop_thread = threading.current_thread()
        loop = asyncio.get_running_loop()

        class RecordingBudget:
            def reserve(self, seconds):
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                calls.append((threading.current_thread(), running_loop, seconds))
                return True

        fake = FakeWebSocket()
        session = stt_backends.RealtimeScribe(fake, None, RecordingBudget())
        chunks = [np.full(160, value, dtype=np.int16).tobytes() for value in range(3)]
        for chunk in chunks:
            self.assertTrue(await session.send_audio(chunk))

        self.assertIs(asyncio.get_running_loop(), loop)
        self.assertEqual(len(calls), len(chunks))
        for thread, running_loop, seconds in calls:
            self.assertIsNot(thread, loop_thread)
            self.assertIsNone(running_loop)
            self.assertAlmostEqual(seconds, 0.01)
        self.assertEqual(
            [message["audio_base_64"] for message in fake.sent],
            [stt_backends.base64.b64encode(chunk).decode("ascii") for chunk in chunks],
        )

    async def test_event_loop_ticks_while_sending_audio_with_fsync(self):
        tick_seen = threading.Event()
        ticks = 0
        real_fsync = stt_backends.os.fsync

        def fsync_with_tick(fd):
            real_fsync(fd)
            # Hold the first real write until the loop ticks. The timeout only
            # bounds a regression; no assertion depends on disk latency.
            tick_seen.wait(timeout=1.0)

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1
                tick_seen.set()

        with tempfile.TemporaryDirectory() as root:
            budget = stt_backends.DailyAudioBudget(
                path=Path(root) / "usage.json", daily_seconds=10,
                settings=get_settings(),
            )
            fake = FakeWebSocket()
            session = stt_backends.RealtimeScribe(fake, None, budget)
            chunk = np.zeros(160, dtype=np.int16).tobytes()
            chunk_count = 8
            ticker_task = asyncio.create_task(ticker())
            try:
                with patch.object(stt_backends.os, "fsync", side_effect=fsync_with_tick) as fsync:
                    for _ in range(chunk_count):
                        self.assertTrue(await session.send_audio(chunk))
                    self.assertGreaterEqual(ticks, 1)
                    self.assertEqual(fsync.call_count, chunk_count)
            finally:
                ticker_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await ticker_task

            self.assertEqual(len(fake.sent), chunk_count)
            self.assertAlmostEqual(budget.remaining(), 9.92, places=6)

    async def test_success_sends_pcm_partial_and_one_commit(self):
        fake = FakeWebSocket([
            {"message_type": "session_started"},
            {"message_type": "partial_transcript", "text": "Ehi Lari"},
        ])
        partials = []
        budget = stt_backends.DailyAudioBudget(
            path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
        , settings=get_settings())

        async def on_partial(text):
            partials.append(text)

        async def connect(_url, **_kwargs):
            return fake

        settings = load_settings({"ELEVENLABS_API_KEY": "test-key"})
        with patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(
                on_partial=on_partial, budget=budget,
                settings=settings,
            )
            self.assertTrue(await session.send_audio(np.arange(1600, dtype=np.int16).tobytes()))
            # Let the reader deliver the already queued partial event.
            await asyncio.sleep(0)
            text = await session.finish()

        self.assertEqual(partials, ["Ehi Lari"])
        self.assertEqual(text, "Ehi Lari, che tempo fa a Roma?")
        self.assertEqual(sum(bool(m.get("commit")) for m in fake.sent), 1)
        self.assertTrue(all(
            m.get("commit") is False
            for m in fake.sent[:-1]
            if m["message_type"] == "input_audio_chunk"
        ))
        self.assertEqual(fake.sent[0]["message_type"], "input_audio_chunk")
        self.assertEqual(fake.sent[0]["sample_rate"], 16000)
        self.assertEqual(fake.sent[-1]["sample_rate"], 16000)
        self.assertNotIn("test-key", json.dumps(fake.sent))
        self.assertAlmostEqual(budget.remaining(), 9.9, places=3)
        self.assertTrue(fake.closed)

    async def test_finish_accumulates_automatic_and_final_committed_segments(self):
        class SegmentedFakeWebSocket(FakeWebSocket):
            def __init__(self):
                super().__init__([{"message_type": "session_started"}])
                self.automatic_sent = False

            async def send(self, raw):
                message = json.loads(raw)
                self.sent.append(message)
                if (message["message_type"] == "input_audio_chunk"
                        and not message.get("commit", False)
                        and not self.automatic_sent):
                    self.automatic_sent = True
                    await self.events.put({
                        "message_type": "committed_transcript",
                        "text": "Ehi Lari",
                    })
                elif message.get("commit") is True:
                    await self.events.put({
                        "message_type": "committed_transcript",
                        "text": "che tempo fa a Roma?",
                    })

        fake = SegmentedFakeWebSocket()
        budget = stt_backends.DailyAudioBudget(
            path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
        , settings=get_settings())

        async def connect(_url, **_kwargs):
            return fake

        settings = load_settings({"ELEVENLABS_API_KEY": "test-key"})
        with patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(budget=budget, settings=settings)
            await session.send_audio(np.arange(1600, dtype=np.int16).tobytes())
            text = await session.finish()

        self.assertEqual(text, "Ehi Lari che tempo fa a Roma?")
        self.assertEqual(
            [m.get("commit") for m in fake.sent],
            [False, True],
        )

    async def test_provider_error_is_reported_for_batch_fallback(self):
        fake = FakeWebSocket([
            {"message_type": "session_started"},
            {"message_type": "error", "error": "provider unavailable"},
        ])
        budget = stt_backends.DailyAudioBudget(
            path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
        , settings=get_settings())

        async def connect(_url, **_kwargs):
            return fake

        settings = load_settings({"ELEVENLABS_API_KEY": "test-key"})
        with patch.object(stt_backends.websockets, "connect", connect):
            with self.assertRaises(stt_backends.RealtimeUnavailable):
                await stt_backends.RealtimeScribe.connect(budget=budget, settings=settings)

        # A failed realtime session is not an assistant turn; the caller alone
        # decides whether to invoke the existing batch backend.
        self.assertFalse(any(m.get("commit") for m in fake.sent))

    async def test_realtime_url_has_repeated_encoded_keyterms_and_no_key(self):
        fake = FakeWebSocket([{"message_type": "session_started"}])
        captured = {}

        async def connect(url, **kwargs):
            captured.update(url=url, kwargs=kwargs)
            return fake

        settings = load_settings({"ELEVENLABS_API_KEY": "secret-key"})
        with patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(
                budget=stt_backends.DailyAudioBudget(
                    path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                , settings=get_settings()),
                settings=settings,
            )
            await session.close()

        query = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(query["keyterms"], list(settings.wake_config.realtime_keyterms))
        self.assertGreater(len(query["keyterms"]), 1)
        self.assertTrue(all(len(term) <= 20 for term in query["keyterms"]))
        self.assertNotIn("secret-key", captured["url"])

    async def test_connection_does_not_log_secret_or_url_with_key(self):
        fake = FakeWebSocket([{"message_type": "session_started"}])
        secret = "secret-key"

        async def connect(_url, **_kwargs):
            return fake

        handler = logging.StreamHandler()
        import io
        output = io.StringIO()
        handler.setStream(output)
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            settings = load_settings({"ELEVENLABS_API_KEY": secret})
            with patch.object(stt_backends.websockets, "connect", connect):
                session = await stt_backends.RealtimeScribe.connect(
                    budget=stt_backends.DailyAudioBudget(
                        path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                    , settings=get_settings()),
                    settings=settings,
                )
                await session.close()
        finally:
            logger.removeHandler(handler)
            handler.close()

        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn("xi-api-key", output.getvalue())

    async def test_connect_times_out_without_session_started_and_sends_no_audio(self):
        fake = FakeWebSocket()

        async def connect(_url, **_kwargs):
            return fake

        settings = load_settings({"ELEVENLABS_API_KEY": "test-key"})
        with patch.object(stt_backends.websockets, "connect", connect), \
             patch.object(stt_backends, "REALTIME_SESSION_TIMEOUT_S", 0.01):
            with self.assertRaises(stt_backends.RealtimeUnavailable):
                await stt_backends.RealtimeScribe.connect(
                    budget=stt_backends.DailyAudioBudget(
                        path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                    , settings=get_settings()),
                    settings=settings,
                )

        self.assertEqual(fake.sent, [])
        self.assertTrue(fake.closed)

    async def test_daily_budget_persists_and_blocks_after_cap(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "usage.json"
            first = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0, settings=get_settings())
            self.assertTrue(first.reserve(0.75))
            self.assertFalse(first.reserve(0.3))
            second = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0, settings=get_settings())
            self.assertAlmostEqual(second.remaining(), 0.25, places=3)
            self.assertEqual(path.stat().st_mode & 0o077, 0)

    async def test_daily_budget_reservation_is_atomic_across_processes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "usage.json"
            ctx = multiprocessing.get_context("fork")
            barrier = ctx.Barrier(2)
            results = ctx.Queue()
            processes = [
                ctx.Process(target=_reserve_budget_in_process,
                            args=(path, barrier, results))
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(5)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(sorted(results.get() for _ in processes), [False, True])
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(data["seconds"], 0.75, places=6)

    async def test_corrupt_existing_budget_fails_closed_without_resetting_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "usage.json"
            corrupt = '{"date": "not-a-date", "seconds": "unknown"}'
            path.write_text(corrupt, encoding="utf-8")
            budget = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0, settings=get_settings())

            self.assertEqual(budget.remaining(), 0.0)
            self.assertFalse(budget.reserve(0.1))
            self.assertEqual(path.read_text(encoding="utf-8"), corrupt)

    async def test_budget_exhaustion_sends_no_audio(self):
        with tempfile.TemporaryDirectory() as root:
            budget = stt_backends.DailyAudioBudget(
                path=Path(root) / "usage.json", daily_seconds=0.1
            , settings=get_settings())
            self.assertTrue(budget.reserve(0.1))
            fake = FakeWebSocket()

            async def connect(_url, **_kwargs):
                return fake

            settings = load_settings({"ELEVENLABS_API_KEY": "test-key"})
            with patch.object(stt_backends.websockets, "connect", connect):
                with self.assertRaises(stt_backends.RealtimeUnavailable):
                    await stt_backends.RealtimeScribe.connect(budget=budget, settings=settings)
            self.assertEqual(fake.sent, [])

    async def test_realtime_failure_falls_back_local_without_batch(self):

        pcm = np.zeros(1600, dtype=np.int16)
        local = Mock(return_value="Ehi Lari, local")
        successful = AsyncMock(return_value="Ehi Lari, realtime")
        failed = AsyncMock(side_effect=stt_backends.RealtimeUnavailable())
        paid_batch = Mock(side_effect=AssertionError("paid batch fallback called"))
        with patch.object(dispatch, "_transcribe_local_fallback", local), \
             patch.object(providers, "transcribe", paid_batch):
            self.assertEqual(
                await dispatch.transcribe_realtime_or_batch(pcm, type("R", (), {
                    "finish": successful,
                })(), 1, settings=get_settings()),
                "Ehi Lari, realtime",
            )
            self.assertEqual(
                await dispatch.transcribe_realtime_or_batch(pcm, type("R", (), {
                    "finish": failed,
                })(), 2, settings=get_settings()),
                "Ehi Lari, local",
            )
        local.assert_called_once_with(pcm, get_settings())
        paid_batch.assert_not_called()

    async def test_realtime_provider_failure_never_uses_paid_batch(self):

        pcm = np.zeros(1600, dtype=np.int16)
        local = Mock(return_value="testo locale")
        paid_batch = Mock(side_effect=AssertionError("paid batch fallback called"))
        with patch.object(dispatch, "_transcribe_local_fallback", local), \
             patch.object(providers, "transcribe", paid_batch):
            text, used_batch = await dispatch._transcribe_realtime_or_batch(
                pcm, None, 9, start_failed=True
            , settings=get_settings())

        self.assertEqual(text, "testo locale")
        self.assertFalse(used_batch)
        local.assert_called_once_with(pcm, get_settings())
        paid_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
