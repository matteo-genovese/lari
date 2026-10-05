"""Telemetry reports provider audio, not inferred mascot states."""
from dataclasses import replace
from lari.config import get_settings
from lari import session as session_module
from lari.stt import dispatch
from lari.stt import realtime
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np



async def send_audio(chunk):
    pass


class UsageVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_provider_audio_is_reported_once_per_turn(self):
        messages = []

        async def send(message):
            messages.append(message)

        class Recorder:
            def __init__(self):
                self.chunks = []

            async def send_audio(self, chunk):
                self.chunks.append(chunk)
                return True

        session = session_module.Session(send, send_audio, settings=get_settings())
        session.turn = 7
        prelude = np.full(1600, 3000, dtype=np.int16).tobytes()
        quiet = np.zeros(1600, dtype=np.int16).tobytes()
        session.recv_queue.put_nowait(quiet)
        session.recv_queue.put_nowait(None)
        recorder = Recorder()
        await session._record_utterance(prelude, realtime=recorder)
        self.assertEqual(recorder.chunks, [prelude, quiet])
        self.assertEqual([m for m in messages if m.get("type") == "stt_status"], [
            {"type": "stt_status", "mode": "realtime", "turn": 7},
        ])

    async def test_provider_refusal_reports_local_fallback_not_paid_audio(self):
        messages = []

        async def send(message):
            messages.append(message)

        class RefusingRecorder:
            async def send_audio(self, _chunk):
                return False

        session = session_module.Session(send, send_audio, settings=get_settings())
        session.turn = 8
        prelude = np.full(1600, 3000, dtype=np.int16).tobytes()
        session.recv_queue.put_nowait(None)
        await session._record_utterance(prelude, realtime=RefusingRecorder())
        self.assertEqual([m for m in messages if m.get("type") == "stt_status"], [
            {"type": "stt_status", "mode": "local", "turn": 8},
        ])

    async def test_connection_failure_reports_local_fallback(self):
        messages = []

        async def send(message):
            messages.append(message)

        session = session_module.Session(send, send_audio, settings=get_settings())
        session._record_utterance = AsyncMock(return_value=None)
        with patch.object(session, "_settings", replace(session._settings, stt_backend=realtime.REALTIME_BACKEND)), \
             patch.object(realtime.RealtimeScribe, "connect", new_callable=AsyncMock,
                          side_effect=realtime.RealtimeUnavailable("unavailable")):
            await session._on_wake(local_wake_confirmed=True)
        self.assertEqual([m for m in messages if m.get("type") == "stt_status"], [
            {"type": "stt_status", "mode": "local", "turn": 1},
        ])

    async def test_followup_turn_is_marked_without_claiming_provider_usage(self):
        messages = []

        async def send(message):
            messages.append(message)

        session = session_module.Session(send, send_audio, settings=get_settings())
        session.conversation_until = time.monotonic() + 30
        session._record_utterance = AsyncMock(return_value=None)
        with patch.object(session, "_settings", replace(session._settings, stt_backend=realtime.REALTIME_BACKEND)), \
             patch.object(realtime.RealtimeScribe, "connect", new_callable=AsyncMock,
                          side_effect=realtime.RealtimeUnavailable("unavailable")):
            await session._on_wake(local_wake_confirmed=False)
        waking = next(m for m in messages if m.get("type") == "state" and m.get("state") == "waking")
        self.assertIs(waking.get("followup"), True)
        self.assertFalse(any(m.get("mode") == "realtime" for m in messages))


if __name__ == "__main__":
    unittest.main()
