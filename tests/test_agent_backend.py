"""Hermes requests, session continuity, streaming and approvals."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from lari import server
from lari.server import Session


class AgentDispatchTests(unittest.TestCase):
    def test_hermes_request_pins_provider_and_model(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'choices': [{'message': {'content': ' Ciao! '}}]})
        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient
        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)
        with patch('httpx.AsyncClient', side_effect=client), \
             patch.object(server, 'HERMES_PROVIDER', 'deepseek'), \
             patch.object(server, 'HERMES_MODEL', 'deepseek-flash'):
            out = asyncio.run(server.ask_hermes('ciao'))
        self.assertEqual(out, 'Ciao!')
        self.assertEqual(len(requests), 1)
        body = __import__('json').loads(requests[0].content)
        self.assertEqual((body['provider'], body['model']), ('deepseek', 'deepseek-flash'))
        self.assertEqual(body['model_options']['reasoning'], {'enabled': False})
        self.assertEqual(requests[0].headers['X-Hermes-Session-Key'], server.SESSION_KEY)

    def test_session_continuity_uses_response_id_on_next_turn(self):
        requests = []

        def handler(request):
            requests.append(request)
            content = 'prima' if len(requests) == 1 else 'seconda'
            return httpx.Response(
                200, json={'choices': [{'message': {'content': content}}]},
                headers={'X-Hermes-Session-Id': 'transcript-1'},
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def turns():
            session = Session(None, AsyncMock())
            first = await session._ask_hermes('ciao')
            second = await session._ask_hermes('come stai?')
            return first, second, session.hermes_session_id

        with patch.object(httpx, 'AsyncClient', side_effect=client), \
             patch.object(server, 'HERMES_KEY', 'api-key'):
            first, second, session_id = asyncio.run(turns())

        self.assertEqual((first, second, session_id), ('prima', 'seconda', 'transcript-1'))
        self.assertNotIn('X-Hermes-Session-Id', requests[0].headers)
        self.assertEqual(requests[1].headers['X-Hermes-Session-Id'], 'transcript-1')
        self.assertEqual(requests[0].headers['X-Hermes-Session-Key'], server.SESSION_KEY)
        self.assertEqual(requests[1].headers['X-Hermes-Session-Key'], server.SESSION_KEY)

    def test_concurrent_sessions_do_not_share_transcript_ids(self):
        requests = []

        def handler(request):
            requests.append(request)
            text = __import__('json').loads(request.content)['messages'][-1]['content']
            ids = {'a-first': 'transcript-a', 'b-first': 'transcript-b'}
            response_id = request.headers.get('X-Hermes-Session-Id') or ids[text]
            return httpx.Response(
                200, json={'choices': [{'message': {'content': text}}]},
                headers={'X-Hermes-Session-Id': response_id},
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def turns():
            sessions = [Session(None, AsyncMock()), Session(None, AsyncMock())]
            await asyncio.gather(
                sessions[0]._ask_hermes('a-first'), sessions[1]._ask_hermes('b-first'))
            await asyncio.gather(
                sessions[0]._ask_hermes('a-second'), sessions[1]._ask_hermes('b-second'))
            return sessions

        with patch.object(httpx, 'AsyncClient', side_effect=client), \
             patch.object(server, 'HERMES_KEY', 'api-key'):
            sessions = asyncio.run(turns())

        self.assertEqual(sessions[0].hermes_session_id, 'transcript-a')
        self.assertEqual(sessions[1].hermes_session_id, 'transcript-b')
        by_text = {
            __import__('json').loads(request.content)['messages'][-1]['content']: request
            for request in requests
        }
        self.assertNotIn('X-Hermes-Session-Id', by_text['a-first'].headers)
        self.assertNotIn('X-Hermes-Session-Id', by_text['b-first'].headers)
        self.assertEqual(by_text['a-second'].headers['X-Hermes-Session-Id'], 'transcript-a')
        self.assertEqual(by_text['b-second'].headers['X-Hermes-Session-Id'], 'transcript-b')

    def test_continued_turn_error_does_not_use_cli_or_change_id(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    200, json={'choices': [{'message': {'content': 'ok'}}]},
                    headers={'X-Hermes-Session-Id': 'stable-id'},
                )
            return httpx.Response(503, json={'error': {'message': 'unavailable'}})

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def turns():
            session = Session(None, AsyncMock())
            await session._ask_hermes('prima')
            with self.assertRaises(server.HermesContinuationError) as raised:
                await session._ask_hermes('seguito')
            return session, str(raised.exception)

        with patch.object(httpx, 'AsyncClient', side_effect=client), \
             patch.object(server, 'HERMES_KEY', 'api-key'), \
             patch.object(server, '_ask_cli', return_value='cli transcript') as cli:
            session, error = asyncio.run(turns())

        self.assertEqual(error, 'continuazione Hermes non disponibile')
        self.assertEqual(session.hermes_session_id, 'stable-id')
        cli.assert_not_called()
        self.assertEqual(calls[1].headers['X-Hermes-Session-Id'], 'stable-id')


class _SplitStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


class HermesStreamingTests(unittest.TestCase):
    @staticmethod
    def split_frame(frame, sizes=(1, 2, 5, 3)):
        raw = frame.encode()
        chunks = []
        offset = 0
        for size in sizes:
            if offset >= len(raw):
                break
            chunks.append(raw[offset:offset + size])
            offset += size
        if offset < len(raw):
            chunks.append(raw[offset:])
        return chunks

    def test_streams_speech_ignoring_reasoning_tools_and_forwards_approval(self):
        frames = [
            ": keepalive\n\n",
            "event: hermes.tool.progress\n"
            'data: {"run_id":"run-42","message":"tool output"}\n\n',
            "event: approval.request\n"
            'data: {"run_id":"run-42","action":"send"}\n\n',
            'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"Ciao"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" mondo"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        requests = []

        def handler(request):
            requests.append(request)
            chunks = []
            for frame in frames:
                chunks.extend(self.split_frame(frame))
            return httpx.Response(
                200,
                headers={"X-Hermes-Session-Id": "stream-1"},
                stream=_SplitStream(chunks),
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def run():
            deltas = []
            approvals = []

            async def on_delta(value):
                deltas.append(value)

            async def on_approval(value):
                approvals.append(value)

            reply = await server.stream_hermes(
                "ciao", on_delta=on_delta, on_approval=on_approval)
            return reply, deltas, approvals

        with patch.object(httpx, 'AsyncClient', side_effect=client), \
             patch.object(server, 'HERMES_KEY', 'api-key'):
            reply, deltas, approvals = asyncio.run(run())

        self.assertEqual(reply, 'Ciao mondo')
        self.assertEqual(deltas, ['Ciao', ' mondo'])
        self.assertEqual(approvals, [{'run_id': 'run-42', 'action': 'send'}])
        self.assertEqual(reply.session_id, 'stream-1')
        self.assertEqual(reply.run_id, 'run-42')
        self.assertEqual(len(requests), 1)
        body = __import__('json').loads(requests[0].content)
        self.assertEqual(body['stream'], True)
        self.assertEqual((body['provider'], body['model']), ('deepseek', 'deepseek-flash'))
        self.assertEqual(body['model_options']['reasoning'], {'enabled': False})
        self.assertEqual(requests[0].headers['X-Hermes-Session-Key'], server.SESSION_KEY)

    def test_stream_continuity_uses_session_id_only_after_done(self):
        requests = []

        def handler(request):
            requests.append(request)
            text = 'prima' if len(requests) == 1 else 'seconda'
            frames = [
                'data: ' + json.dumps({
                    'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}],
                }) + '\n\n',
                'data: [DONE]\n\n',
            ]
            return httpx.Response(
                200,
                headers={'X-Hermes-Session-Id': 'stream-transcript'},
                stream=_SplitStream([part.encode() for frame in frames for part in [frame]]),
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def turns():
            first = await server.stream_hermes('uno', on_delta=AsyncMock())
            second = await server.stream_hermes(
                'due', session_id=first.session_id, on_delta=AsyncMock())
            return first, second

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            first, second = asyncio.run(turns())

        self.assertEqual((str(first), str(second)), ('prima', 'seconda'))
        self.assertNotIn('X-Hermes-Session-Id', requests[0].headers)
        self.assertEqual(requests[1].headers['X-Hermes-Session-Id'], 'stream-transcript')

    def test_error_finish_does_not_retry_or_emit_error_delta(self):
        calls = []

        def handler(request):
            calls.append(request)
            stream = _SplitStream([
                b'data: {"choices":[{"delta":{"content":"partial"},'
                b'"finish_reason":"error"}]}\n\n',
                b'data: [DONE]\n\n',
            ])
            return httpx.Response(200, stream=stream)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def run():
            deltas = []

            async def on_delta(value):
                deltas.append(value)

            with self.assertRaisesRegex(RuntimeError, 'terminato con errore'):
                await server.stream_hermes(
                    'ciao', session_id='prior-id', on_delta=on_delta)
            return deltas

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            deltas = asyncio.run(run())

        self.assertEqual(deltas, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers['X-Hermes-Session-Id'], 'prior-id')

    def test_length_finish_does_not_retry_or_emit_truncated_delta(self):
        calls = []

        def handler(request):
            calls.append(request)
            stream = _SplitStream([
                b'data: {"choices":[{"delta":{"content":"partial"},'
                b'"finish_reason":"length"}]}' + b'\n\n',
                b'data: [DONE]\n\n',
            ])
            return httpx.Response(200, stream=stream)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def run():
            deltas = []

            async def on_delta(value):
                deltas.append(value)

            with self.assertRaisesRegex(RuntimeError, r"finish_reason='length'"):
                await server.stream_hermes(
                    'ciao', session_id='prior-id', on_delta=on_delta)
            return deltas

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            deltas = asyncio.run(run())

        self.assertEqual(deltas, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers['X-Hermes-Session-Id'], 'prior-id')

    def test_incomplete_stream_fails_without_replacing_prior_session(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                headers={'X-Hermes-Session-Id': 'new-id'},
                stream=_SplitStream([
                    b': keepalive\n\n',
                    b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
                ]),
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def run():
            prior_id = 'prior-id'
            with self.assertRaisesRegex(RuntimeError, r'manca \[DONE\]'):
                await server.stream_hermes(
                    'ciao', session_id=prior_id, on_delta=AsyncMock())
            return prior_id

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            prior_id = asyncio.run(run())

        self.assertEqual(prior_id, 'prior-id')
        self.assertEqual(len(calls), 1)

    def test_done_without_stop_finish_fails_without_publishing_session(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                headers={'X-Hermes-Session-Id': 'new-id'},
                stream=_SplitStream([
                    b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
                    b'data: [DONE]\n\n',
                ]),
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        async def run():
            with self.assertRaisesRegex(RuntimeError, r"manca finish_reason='stop'"):
                await server.stream_hermes('ciao', session_id='prior-id')

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            asyncio.run(run())

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers['X-Hermes-Session-Id'], 'prior-id')


class _AudioWebSocket:
    def __init__(self):
        self.binary = []
        self.binary_sent = asyncio.Event()

    async def send_bytes(self, data):
        self.binary.append(data)
        self.binary_sent.set()


class SegmentedPlaybackTests(unittest.TestCase):
    def test_sentence_buffer_waits_for_words_and_splits_at_punctuation_or_limit(self):
        buffer = server.SpeakableSentenceBuffer(max_chars=14, total_limit=100)
        self.assertEqual(buffer.feed('Ciao mondo'), [])
        self.assertEqual(buffer.feed('. Seconda '), ['Ciao mondo.'])
        self.assertEqual(buffer.feed('frase lunga qui'), ['Seconda frase'])
        self.assertEqual(buffer.flush(), ['lunga qui'])

    def test_segmented_audio_order_and_session_id_after_done(self):
        ws = _AudioWebSocket()
        sent = []

        async def send_json(value):
            sent.append(value)

        async def fake_stream(text, session_id=None, on_delta=None, on_approval=None):
            self.assertIsNone(session_id)
            await on_delta('Prima frase. ')
            await on_delta('Seconda frase.')
            return server.HermesReply('Prima frase. Seconda frase.', 'done-id')

        async def run():
            session = Session(ws, send_json)
            with patch.object(server, 'stream_hermes', side_effect=fake_stream), \
                 patch.object(server, 'tts', side_effect=[b'one', b'two']):
                result = await session._stream_hermes_speak('ciao', 4)
            return session, result

        session, result = asyncio.run(run())
        self.assertEqual(result, ('Prima frase. Seconda frase.', True, False))
        self.assertEqual(session.hermes_session_id, 'done-id')
        self.assertEqual(
            [(item['type'], item.get('seq')) for item in sent
             if item['type'] in {'audio_start', 'audio_chunk', 'audio_end'}],
            [('audio_start', None), ('audio_chunk', 0),
             ('audio_chunk', 1), ('audio_end', None)],
        )
        self.assertEqual(ws.binary, [b'one', b'two'])

    def test_tts_failure_does_not_retry_hermes_or_replace_completed_session(self):
        ws = _AudioWebSocket()
        sent = []

        async def fake_stream(text, session_id=None, on_delta=None, on_approval=None):
            await on_delta('Una risposta.')
            return server.HermesReply('Una risposta.', 'completed-id')

        async def run():
            async def send_json(value):
                sent.append(value)
            session = Session(ws, send_json)
            with patch.object(server, 'stream_hermes', side_effect=fake_stream) as stream, \
                 patch.object(server, 'tts', side_effect=RuntimeError('provider secret/path')):
                result = await session._stream_hermes_speak('ciao', 5)
            return session, stream, result

        session, stream, result = asyncio.run(run())
        self.assertEqual(result, ('Una risposta.', False, True))
        self.assertEqual(session.hermes_session_id, 'completed-id')
        stream.assert_awaited_once()
        self.assertFalse(any(item['type'].startswith('audio_') for item in sent))

    def test_stream_error_ends_existing_audio_without_reissuing_turn(self):
        ws = _AudioWebSocket()
        sent = []
        calls = []

        async def fake_stream(text, session_id=None, on_delta=None, on_approval=None):
            calls.append(text)
            await on_delta('Parziale.')
            raise RuntimeError('sensitive transport detail')

        async def run():
            async def send_json(value):
                sent.append(value)
            session = Session(ws, send_json)
            with patch.object(server, 'stream_hermes', side_effect=fake_stream), \
                 patch.object(server, 'tts', return_value=b'audio'):
                with self.assertRaises(server.HermesStreamTurnError) as raised:
                    await session._stream_hermes_speak('ciao', 6)
            return raised.exception

        error = asyncio.run(run())
        self.assertTrue(error.had_audio)
        self.assertEqual(calls, ['ciao'])
        self.assertEqual([item['type'] for item in sent],
                         ['state', 'audio_start', 'audio_chunk', 'audio_end'])
        self.assertNotIn('sensitive', str(error))

    def test_playback_ack_requires_current_turn_and_failed_does_not_open_followup(self):
        async def send_json(_):
            pass

        session = Session(None, send_json)
        async def run():
            session._begin_playback(9)
            self.assertFalse(session.mark_playback_done(turn=8, status='completed'))
            self.assertTrue(session.mark_playback_done(turn=9, status='failed'))
            self.assertFalse(session.mark_playback_done(turn=9, status='completed'))

        asyncio.run(run())
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 0.0)

    def test_midstream_interrupt_cancels_sse_and_tts_without_retry_or_audio_end(self):
        ws = _AudioWebSocket()
        sent = []
        stream_cancelled = asyncio.Event()
        stream_calls = []

        async def send_json(value):
            sent.append(value)

        async def fake_stream(text, session_id=None, on_delta=None, on_approval=None):
            stream_calls.append(text)
            await on_delta('Prima frase.')
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stream_cancelled.set()
                raise

        async def run():
            session = Session(ws, send_json)
            task = asyncio.create_task(session._stream_hermes_speak('ciao', 7))
            session.active_turn = 7
            session._turn_task = task
            await asyncio.wait_for(ws.binary_sent.wait(), 1)
            self.assertTrue(await session.interrupt_current_turn(7))
            with self.assertRaises(asyncio.CancelledError):
                await task
            return session

        with patch.object(server, 'stream_hermes', side_effect=fake_stream), \
             patch.object(server, 'tts', return_value=b'audio'):
            session = asyncio.run(run())

        self.assertEqual(stream_calls, ['ciao'])
        self.assertTrue(stream_cancelled.is_set())
        self.assertTrue(session._turn_task is not None and session._turn_task.cancelled())
        self.assertEqual(session.playback_status, 'interrupted')
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 0.0)
        self.assertEqual(ws.binary, [b'audio'])
        self.assertEqual(
            [item['type'] for item in sent],
            ['state', 'audio_start', 'audio_chunk', 'interrupt_ack', 'state'],
        )
        self.assertNotIn('audio_end', [item['type'] for item in sent])

    def test_stale_or_wrong_turn_interrupt_is_rejected_without_cancelling_current_turn(self):
        async def run():
            session = Session(None, AsyncMock())
            task = asyncio.create_task(asyncio.Event().wait())
            session.active_turn = 12
            session._turn_task = task
            stale = await session.interrupt_current_turn(11)
            wrong = await session.interrupt_current_turn(13)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return session, stale, wrong

        session, stale, wrong = asyncio.run(run())
        self.assertFalse(stale)
        self.assertFalse(wrong)
        self.assertEqual(session.active_turn, 12)
        self.assertIsNone(session.playback_status)

    def test_interrupted_playback_ack_never_opens_followup(self):
        async def send_json(_):
            pass

        session = Session(None, send_json)
        async def run():
            session._begin_playback(14)
            session._mark_playback_interrupted(14)
            self.assertFalse(session.mark_playback_done(turn=14, status='completed'))

        asyncio.run(run())
        self.assertEqual(session.playback_status, 'interrupted')
        self.assertFalse(session.awaiting_playback)
        self.assertEqual(session.conversation_until, 0.0)


if __name__ == '__main__':
    unittest.main()
