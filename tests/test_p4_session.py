"""Session ownership, playback ACK and cancellable voice regressions."""
from contextlib import nullcontext
from dataclasses import replace
from lari.config import get_settings
import asyncio
import sys
import time
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import AsyncMock, Mock, patch

from lari.session import Session
from lari import session as sessions, tts as voice
from lari.hermes import ApprovalNotAvailable, HermesReply
from lari.stt import realtime as live
from lari.stt.realtime import RealtimeScribe


class SessionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_binary_audio_callable_preserves_bytes_and_protocol_order(self):
        events = []
        chunks = [b"ID3\x00\xff\x01", b"\xff\xfb\x00\x02"]

        async def send_json(data):
            events.append(("json", data))

        async def send_audio(chunk):
            events.append(("audio", chunk))

        session = Session(send_json, send_audio, settings=get_settings())
        session.turn = session.active_turn = 1

        async def stream(text, on_delta, **kwargs):
            await on_delta("Prima frase. Seconda frase.")
            return HermesReply("Prima frase. Seconda frase.", "done")

        with patch.object(sessions, "stream_hermes", side_effect=stream), \
             patch.object(voice, "tts", new_callable=AsyncMock, side_effect=chunks):
            await session._stream_hermes_speak("comando", 1)
        self.assertEqual(events, [
            ("json", {"type": "state", "state": "speaking", "turn": 1}),
            ("json", {"type": "audio_start", "turn": 1}),
            ("json", {"type": "audio_chunk", "turn": 1, "seq": 0}),
            ("audio", chunks[0]),
            ("json", {"type": "audio_chunk", "turn": 1, "seq": 1}),
            ("audio", chunks[1]),
            ("json", {"type": "audio_end", "turn": 1}),
        ])

        events.clear()
        with patch.object(voice, "tts", new_callable=AsyncMock, return_value=chunks[0]):
            await session._speak("Risposta singola.", 1)
        self.assertEqual(events, [
            ("json", {"type": "state", "state": "speaking", "turn": 1}),
            ("json", {"type": "audio", "fmt": "mp3", "bytes": len(chunks[0]), "turn": 1}),
            ("audio", chunks[0]),
        ])
        await session.disconnect()

    async def test_stale_ack_leaves_current_turn_and_waiter_untouched(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        session.turn = session.active_turn = 2
        session.state = "speaking"
        session._begin_playback(2)
        waiter = session._playback_waiter
        snapshot = (session.state, session.active_turn, session.last_tts,
                    session.conversation_until, session.playback_status)
        self.assertFalse(await session.playback_completed(1))
        self.assertEqual(snapshot, (session.state, session.active_turn, session.last_tts,
                                   session.conversation_until, session.playback_status))
        self.assertIs(session._playback_waiter, waiter)
        self.assertFalse(waiter.done())
        self.assertTrue(session.awaiting_playback)
        await session.disconnect()

    async def test_interrupt_during_synthesis_cancels_reading_without_resubmission(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        session.turn = session.active_turn = 3
        synthesizing = asyncio.Event()
        finalized = asyncio.Event()

        async def synth(text, settings):
            synthesizing.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalized.set()

        async def stream(text, **callbacks):
            await callbacks["on_delta"]("Prima frase. Seconda frase.")
            return HermesReply("Prima frase. Seconda frase.", "id")

        with patch.object(voice, "tts", side_effect=synth), \
             patch.object(sessions, "stream_hermes", side_effect=stream) as provider:
            task = asyncio.create_task(session._stream_hermes_speak("comando", 3))
            session._turn_task = task
            await asyncio.wait_for(synthesizing.wait(), 1)
            reading = session._reading
            self.assertTrue(await session.interrupt())
            self.assertTrue(finalized.is_set())
            self.assertTrue(task.done())
            self.assertTrue(reading.worker.done())
            self.assertTrue(reading.queue.empty())
            session._send_audio.assert_not_awaited()
            self.assertFalse(await session.playback_completed(3))
            provider.assert_awaited_once()
            followup = session._interrupted_followup_task
            await session.disconnect()
            self.assertTrue(followup.done())

    async def test_disconnect_awaits_turn_tts_and_realtime_reader(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        started = asyncio.Event()
        recording = asyncio.Event()
        socket = Mock(close=AsyncMock())
        realtime = RealtimeScribe(socket, None, Mock())
        realtime._reader_task = asyncio.create_task(asyncio.Event().wait())
        reader = realtime._reader_task

        async def record(*args, **kwargs):
            recording.set()
            await asyncio.Event().wait()

        session._record_utterance = record
        with patch.object(sessions._dispatch, "open_stream", return_value=(realtime, False)):
            task = asyncio.create_task(session._on_wake(b"pcm", True))
            await asyncio.wait_for(recording.wait(), 1)
            await session.disconnect()
        self.assertTrue(task.done())
        self.assertTrue(reader.done())
        socket.close.assert_awaited_once()
        self.assertIsNone(session._realtime)
        self.assertIsNone(session.active_turn)

        # A different disconnect point: voice synthesis and SSE are both active.
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        session.turn = session.active_turn = 4

        async def synth(text, settings):
            started.set()
            await asyncio.Event().wait()

        async def stream(text, **callbacks):
            await callbacks["on_delta"]("Frase.")
            await asyncio.Event().wait()

        with patch.object(voice, "tts", side_effect=synth), \
             patch.object(sessions, "stream_hermes", side_effect=stream):
            task = asyncio.create_task(session._stream_hermes_speak("comando", 4))
            await asyncio.wait_for(started.wait(), 1)
            reading = session._reading
            await session.disconnect()
        self.assertTrue(task.done())
        self.assertTrue(reading.worker.done())
        self.assertTrue(reading.queue.empty())
        self.assertFalse(session._jobs)
        self.assertFalse(session.awaiting_playback)

    async def test_two_sessions_keep_audio_turns_ids_and_followups_independent(self):
        messages = [[], []]
        audio = [[], []]
        async def send_audio(index, chunk):
            audio[index].append(chunk)
        async def send(index, message):
            messages[index].append(message)
        first = Session(lambda message: send(0, message), lambda chunk: send_audio(0, chunk), settings=get_settings())
        second = Session(lambda message: send(1, message), lambda chunk: send_audio(1, chunk), settings=get_settings())
        first.turn = first.active_turn = 7
        second.turn = second.active_turn = 9
        first.hermes_session_id, second.hermes_session_id = "first", "second"
        both = asyncio.Event()
        count = 0

        async def stream(text, session_id, on_delta, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                both.set()
            await both.wait()
            await on_delta(text + ". Uno. Due.")
            return HermesReply(text, session_id + "-done")

        async def synth(text, settings):
            await asyncio.sleep(0)
            return text.encode()

        await first.on_audio(b"first-pcm")
        await second.on_audio(b"second-pcm")
        self.assertEqual(first.recv_queue.get_nowait(), b"first-pcm")
        self.assertEqual(second.recv_queue.get_nowait(), b"second-pcm")
        with patch.object(sessions, "stream_hermes", side_effect=stream), \
             patch.object(voice, "tts", side_effect=synth):
            await asyncio.gather(first._stream_hermes_speak("Prima", 7),
                                 second._stream_hermes_speak("Seconda", 9))
        self.assertEqual(audio[0], [b"Prima.", b"Uno.", b"Due."])
        self.assertEqual(audio[1], [b"Seconda.", b"Uno.", b"Due."])
        self.assertEqual((first.hermes_session_id, second.hermes_session_id),
                         ("first-done", "second-done"))
        self.assertEqual((first.conversation_until, second.conversation_until), (0, 0))
        self.assertTrue(await first.playback_completed(7))
        self.assertGreater(first.conversation_until, time.monotonic())
        self.assertTrue(second.awaiting_playback)
        self.assertEqual(second.conversation_until, 0)
        self.assertFalse(await second.playback_completed(7))
        await first.disconnect()
        self.assertTrue(second.awaiting_playback)
        self.assertTrue(await second.playback_completed(9))
        self.assertGreater(second.conversation_until, time.monotonic())
        self.assertTrue(all(m.get("turn", 7) == 7 for m in messages[0]))
        self.assertTrue(all(m.get("turn", 9) == 9 for m in messages[1]))
        await second.disconnect()

    async def test_two_complete_turns_wait_for_their_own_playback_ack(self):
        ended = [asyncio.Event(), asyncio.Event()]
        messages = [[], []]
        async def send(index, message):
            messages[index].append(message)
            if message["type"] == "audio_end":
                ended[index].set()
        first = Session(lambda message: send(0, message), AsyncMock(), settings=get_settings())
        second = Session(lambda message: send(1, message), AsyncMock(), settings=get_settings())
        first.turn, second.turn = 3, 8
        first.hermes_session_id, second.hermes_session_id = "one", "two"
        first._record_utterance = AsyncMock(return_value=np.ones(16000, dtype=np.int16))
        second._record_utterance = AsyncMock(return_value=np.full(16000, 2, dtype=np.int16))
        async def transcribe(pcm, *args, **kwargs):
            return "Prima" if pcm[0] == 1 else "Seconda"
        async def stream(text, session_id, on_delta, **kwargs):
            await on_delta(text + ".")
            return HermesReply(text, session_id + "-done")
        with patch.object(sessions._dispatch, "open_stream", return_value=(None, False)), \
             patch.object(sessions._dispatch, "transcribe_turn", side_effect=transcribe), \
             patch.object(sessions._dispatch, "command_for_turn", side_effect=lambda text, *args, **kwargs: text), \
             patch.object(sessions, "save_turn_audio", return_value=Path("fake.wav")), \
             patch.object(sessions, "stream_hermes", side_effect=stream), \
             patch.object(voice, "tts", side_effect=lambda text, settings: text.encode()):
            tasks = [asyncio.create_task(first._on_wake(local_wake_confirmed=True)),
                     asyncio.create_task(second._on_wake(local_wake_confirmed=True))]
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ended)), 1)
            self.assertFalse(any(task.done() for task in tasks))
            self.assertEqual((first.active_turn, second.active_turn), (4, 9))
            self.assertTrue(await first.playback_completed(4))
            await asyncio.wait_for(tasks[0], 1)
            self.assertFalse(tasks[1].done())
            self.assertEqual(second.active_turn, 9)
            self.assertEqual(second.conversation_until, 0)
            await first.disconnect()
            self.assertTrue(await second.playback_completed(9))
            await asyncio.wait_for(tasks[1], 1)
        first._send_audio.assert_awaited_once_with(b"Prima.")
        second._send_audio.assert_awaited_once_with(b"Seconda.")
        self.assertEqual((first.hermes_session_id, second.hermes_session_id),
                         ("one-done", "two-done"))
        self.assertTrue(all(task.done() for task in tasks))
        self.assertFalse(first._jobs or second._jobs)
        await second.disconnect()

    async def test_old_turn_cannot_publish_text_audio_or_session_id(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        session.turn = session.active_turn = 12
        session.hermes_session_id = "current"
        async def stream(text, on_delta, on_approval, **kwargs):
            await on_delta("Obsoleto.")
            await on_approval({"secret": "old"})
            return HermesReply("Obsoleto.", "old-id")
        with patch.object(sessions, "stream_hermes", side_effect=stream), \
             patch.object(voice, "tts", new_callable=AsyncMock) as synth:
            with self.assertRaises(ApprovalNotAvailable):
                await session._stream_hermes_speak("vecchio", 11)
            await session._send_partial("vecchio", 11)
            await session.set_state("thinking", turn=11)
        self.assertEqual(session.hermes_session_id, "current")
        session._send_json.assert_not_awaited()
        synth.assert_not_awaited()
        session._send_audio.assert_not_awaited()
        await session.disconnect()

    async def test_disconnect_awaits_wake_work_scheduled_before_it_starts(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        session.conversation_until = time.monotonic() + 30
        session._record_utterance = AsyncMock(side_effect=AssertionError("disconnected"))
        loop = asyncio.get_running_loop()
        create_task = loop.create_task
        created = []
        def tracked(coro, **kwargs):
            task = create_task(coro, **kwargs)
            created.append(task)
            return task
        with patch.object(loop, "create_task", side_effect=tracked):
            self.assertTrue(session._launch_local_candidate(b"pcm"))
            await session.disconnect()
        self.assertTrue(created)
        self.assertTrue(all(task.done() for task in created))
        session._record_utterance.assert_not_awaited()
        self.assertFalse(session._jobs)

    async def test_single_response_provider_error_is_safe(self):
        session = Session(AsyncMock(), AsyncMock(), settings=get_settings())
        with patch.object(voice, "tts", side_effect=RuntimeError("secret/path")):
            await session._speak("test", 1)
        error = session._send_json.call_args.args[0]["error"]
        self.assertEqual(error, "tts non disponibile")
        self.assertFalse(session.awaiting_playback)
        await session.disconnect()


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_handshake_cancellation_awaits_provider_reader(self):
        entered = asyncio.Event()
        async def recv():
            entered.set()
            await asyncio.Event().wait()
        socket = Mock(recv=recv, close=AsyncMock())
        settings = replace(get_settings(), elevenlabs_api_key="key")
        budget = Mock(remaining=Mock(return_value=10))
        created = []
        create_task = asyncio.create_task
        def tracked(coro, **kwargs):
            task = create_task(coro, **kwargs)
            created.append(task)
            return task
        with patch.object(live.websockets, "connect", new_callable=AsyncMock, return_value=socket), \
             patch.object(live.asyncio, "create_task", side_effect=tracked):
            connect = create_task(RealtimeScribe.connect(settings=settings, budget=budget))
            await asyncio.wait_for(entered.wait(), 1)
            connect.cancel()
            await asyncio.gather(connect, return_exceptions=True)
        self.assertTrue(connect.done())
        self.assertTrue(created)
        self.assertTrue(all(task.done() for task in created))
        socket.close.assert_awaited_once()

    async def test_real_provider_stream_cancellation_closes_generator(self):
        entered = asyncio.Event()
        closed = asyncio.Event()
        async def chunks():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield {"type": "audio", "data": b"unused"}
            finally:
                closed.set()
        provider = Mock(Communicate=Mock(return_value=Mock(stream=chunks)))
        with patch.dict(sys.modules, {"edge_tts": provider}):
            task = asyncio.create_task(voice.tts("ciao", settings=get_settings()))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.done())
        self.assertTrue(closed.is_set())

    async def test_bounded_queue_applies_backpressure_and_preserves_order(self):
        started = asyncio.Event()
        release = asyncio.Event()
        audio = []
        async def synth(text, settings):
            started.set()
            await release.wait()
            return text.encode()
        async def send(data):
            audio.append(data)
        with nullcontext(replace(get_settings(), stream_tts_queue_max=1)) as settings, \
             patch.object(voice, "tts", side_effect=synth):
            reading = voice.Reading(1, AsyncMock(), send, AsyncMock(), lambda: None, lambda: True, settings=settings)
            reading.start()
            await reading.feed("Prima.")
            await started.wait()
            await reading.feed("Seconda.")
            blocked = asyncio.create_task(reading.feed("Terza."))
            await asyncio.sleep(0)
            self.assertFalse(blocked.done())
            self.assertEqual(reading.queue.qsize(), 1)
            release.set()
            await blocked
            await reading.finish()
        self.assertTrue(reading.worker.done())
        self.assertEqual(audio, [b"Prima.", b"Seconda.", b"Terza."])

    async def test_limits_and_typed_error(self):
        buffer = voice.SpeakableSentenceBuffer(max_chars=5, total_limit=10)
        self.assertEqual(buffer.feed("uno due"), ["uno"])
        self.assertEqual(buffer.flush(), ["due"])
        with self.assertRaises(RuntimeError):
            buffer.feed("troppo lungo")
        with patch.dict(sys.modules, {"edge_tts": Mock(Communicate=Mock(side_effect=RuntimeError("secret")))}):
            with self.assertRaisesRegex(voice.TTSUnavailable, "^tts non disponibile$"):
                await voice.tts("ciao", settings=get_settings())
