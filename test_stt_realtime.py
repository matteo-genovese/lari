"""Offline tests for the bounded ElevenLabs Scribe v2 Realtime path."""
import asyncio
import json
import logging
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import stt_backends


def _reserve_budget_in_process(path, barrier, results):
    budget = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0)
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
    async def test_success_sends_pcm_partial_and_one_commit(self):
        fake = FakeWebSocket([
            {"message_type": "session_started"},
            {"message_type": "partial_transcript", "text": "Ehi Lari"},
        ])
        partials = []
        budget = stt_backends.DailyAudioBudget(
            path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
        )

        async def on_partial(text):
            partials.append(text)

        async def connect(_url, **_kwargs):
            return fake

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
             patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(
                on_partial=on_partial, budget=budget
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
        )

        async def connect(_url, **_kwargs):
            return fake

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
             patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(budget=budget)
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
        )

        async def connect(_url, **_kwargs):
            return fake

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
             patch.object(stt_backends.websockets, "connect", connect):
            with self.assertRaises(stt_backends.RealtimeUnavailable):
                await stt_backends.RealtimeScribe.connect(budget=budget)

        # A failed realtime session is not an assistant turn; the caller alone
        # decides whether to invoke the existing batch backend.
        self.assertFalse(any(m.get("commit") for m in fake.sent))

    async def test_realtime_url_has_repeated_encoded_keyterms_and_no_key(self):
        fake = FakeWebSocket([{"message_type": "session_started"}])
        captured = {}

        async def connect(url, **kwargs):
            captured.update(url=url, kwargs=kwargs)
            return fake

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "secret-key"}), \
             patch.object(stt_backends.websockets, "connect", connect):
            session = await stt_backends.RealtimeScribe.connect(
                budget=stt_backends.DailyAudioBudget(
                    path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                )
            )
            await session.close()

        query = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(
            query["keyterms"],
            ["Ehi Lari"],
        )
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
            with patch.dict("os.environ", {"ELEVENLABS_API_KEY": secret}), \
                 patch.object(stt_backends.websockets, "connect", connect):
                session = await stt_backends.RealtimeScribe.connect(
                    budget=stt_backends.DailyAudioBudget(
                        path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                    )
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

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
             patch.object(stt_backends.websockets, "connect", connect), \
             patch.object(stt_backends, "REALTIME_SESSION_TIMEOUT_S", 0.01):
            with self.assertRaises(stt_backends.RealtimeUnavailable):
                await stt_backends.RealtimeScribe.connect(
                    budget=stt_backends.DailyAudioBudget(
                        path=Path(tempfile.mkdtemp()) / "usage.json", daily_seconds=10
                    )
                )

        self.assertEqual(fake.sent, [])
        self.assertTrue(fake.closed)

    async def test_daily_budget_persists_and_blocks_after_cap(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "usage.json"
            first = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0)
            self.assertTrue(first.reserve(0.75))
            self.assertFalse(first.reserve(0.3))
            second = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0)
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
            budget = stt_backends.DailyAudioBudget(path=path, daily_seconds=1.0)

            self.assertEqual(budget.remaining(), 0.0)
            self.assertFalse(budget.reserve(0.1))
            self.assertEqual(path.read_text(encoding="utf-8"), corrupt)

    async def test_budget_exhaustion_sends_no_audio(self):
        with tempfile.TemporaryDirectory() as root:
            budget = stt_backends.DailyAudioBudget(
                path=Path(root) / "usage.json", daily_seconds=0.1
            )
            self.assertTrue(budget.reserve(0.1))
            fake = FakeWebSocket()

            async def connect(_url, **_kwargs):
                return fake

            with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
                 patch.object(stt_backends.websockets, "connect", connect):
                with self.assertRaises(stt_backends.RealtimeUnavailable):
                    await stt_backends.RealtimeScribe.connect(budget=budget)
            self.assertEqual(fake.sent, [])

    async def test_realtime_failure_falls_back_local_without_batch(self):
        import server

        pcm = np.zeros(1600, dtype=np.int16)
        local = Mock(return_value="Ehi Lari, local")
        successful = AsyncMock(return_value="Ehi Lari, realtime")
        failed = AsyncMock(side_effect=stt_backends.RealtimeUnavailable())
        paid_batch = Mock(side_effect=AssertionError("paid batch fallback called"))
        with patch.object(server, "_transcribe_local_fallback", local), \
             patch.object(server.stt_backends, "transcribe", paid_batch):
            self.assertEqual(
                await server.transcribe_realtime_or_batch(pcm, type("R", (), {
                    "finish": successful,
                })(), 1),
                "Ehi Lari, realtime",
            )
            self.assertEqual(
                await server.transcribe_realtime_or_batch(pcm, type("R", (), {
                    "finish": failed,
                })(), 2),
                "Ehi Lari, local",
            )
        local.assert_called_once_with(pcm)
        paid_batch.assert_not_called()

    async def test_realtime_provider_failure_never_uses_paid_batch(self):
        import server

        pcm = np.zeros(1600, dtype=np.int16)
        local = Mock(return_value="testo locale")
        paid_batch = Mock(side_effect=AssertionError("paid batch fallback called"))
        with patch.object(server, "_transcribe_local_fallback", local), \
             patch.object(server.stt_backends, "transcribe", paid_batch):
            text, used_batch = await server._transcribe_realtime_or_batch(
                pcm, None, 9, start_failed=True
            )

        self.assertEqual(text, "testo locale")
        self.assertFalse(used_batch)
        local.assert_called_once_with(pcm)
        paid_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
