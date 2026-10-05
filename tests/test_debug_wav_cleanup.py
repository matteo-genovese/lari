"""Temporary debug microphone audio survives sending and is always removed."""
import asyncio
from contextlib import ExitStack
import io
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

from lari import server


async def inline_file_io(function, *args, **kwargs):
    """Exercise real file operations without an external worker-thread wakeup."""
    return function(*args)


class DebugWavCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch('anyio.to_thread.run_sync', inline_file_io))
        self.directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(server.tempfile, 'tempdir', str(self.directory)))
        stack.enter_context(patch.object(server, 'TOKEN', 'test-token'))
        self.pcm = b'\x01\x00' * 16000
        session = Mock(recent_lock=threading.Lock(), recent=self.pcm)
        stack.enter_context(patch.object(server, '_sessions', {session}))
        self.scope = {'type': 'http', 'method': 'GET', 'headers': []}

    def assert_empty(self):
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_success_preserves_audio_headers_and_private_file_until_final_send(self):
        for _ in range(3):
            response = await server.debug_last('test-token')
            path = Path(response.path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            expected = path.read_bytes()
            messages = []
            async def send(message):
                self.assertTrue(path.is_file())
                messages.append(message)
            await response(self.scope, AsyncMock(), send)
            self.assert_empty()
            headers = dict(messages[0]['headers'])
            self.assertEqual(headers[b'content-type'], b'audio/wav')
            self.assertEqual(headers[b'content-disposition'], b'attachment; filename="last.wav"')
            body = b''.join(msg['body'] for msg in messages if msg['type'] == 'http.response.body')
            self.assertEqual(body, expected)
            with wave.open(io.BytesIO(body)) as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, server.SAMPLE_RATE))
                self.assertEqual(wav.readframes(wav.getnframes()), self.pcm)

    async def test_send_failure_or_cancellation_mid_body_cleans_up(self):
        for error in (OSError('connection lost'), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                response = await server.debug_last('test-token')
                response.chunk_size = 1024
                path = Path(response.path)
                body_count = 0
                async def send(message):
                    nonlocal body_count
                    self.assertTrue(path.exists())
                    if message['type'] == 'http.response.body':
                        body_count += 1
                        if body_count == 2:
                            raise error
                with self.assertRaises(type(error)):
                    await response(self.scope, AsyncMock(), send)
                self.assertEqual(body_count, 2)
                self.assert_empty()

    async def test_writing_or_response_construction_failure_cleans_up(self):
        for target in ('wave.Wave_write.writeframes', 'lari.server._TemporaryWavResponse'):
            with self.subTest(target=target), patch(target, side_effect=OSError('failure')):
                with self.assertRaises(OSError):
                    await server.debug_last('test-token')
                self.assert_empty()

    async def test_invalid_range_cleans_up(self):
        response = await server.debug_last('test-token')
        scope = dict(self.scope, headers=[(b'range', b'bytes=9999999-')])
        send = AsyncMock()
        await response(scope, AsyncMock(), send)
        self.assertEqual(send.call_args_list[0].args[0]['status'], 416)
        self.assert_empty()

    async def test_rejected_token_and_empty_audio_create_no_file(self):
        self.assertEqual((await server.debug_last('invalid')).status_code, 403)
        with patch.object(server, '_sessions', set()):
            self.assertEqual((await server.debug_last('test-token')).status_code, 404)
        self.assert_empty()
