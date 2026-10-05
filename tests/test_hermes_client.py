"""Stream fallback boundaries using HTTP mocks and no real credentials."""
import asyncio
from contextlib import contextmanager
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from lari.config import load_settings
from lari.hermes.client import stream_hermes
from lari.hermes.events import HermesReply


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), error=None):
        self.chunks = chunks
        self.error = error

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


@contextmanager
def _mock_http(handler):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client(*args, **kwargs):
        return real_client(*args, transport=transport, **kwargs)

    with patch.object(httpx, 'AsyncClient', side_effect=client):
        yield


class HermesStreamFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = load_settings({})

    async def test_initial_connect_failure_calls_cli_once_and_emits_final_delta(self):
        for error_type in (httpx.ConnectError, httpx.ConnectTimeout):
            for emit_delta in (False, True):
                with self.subTest(error_type=error_type, emit_delta=emit_delta):
                    error = error_type('private URL https://private/?token=secret')
                    handler = Mock(side_effect=error)
                    delta = AsyncMock() if emit_delta else None
                    loop = asyncio.get_running_loop()

                    def run_cli_inline(executor, call, *args):
                        # Keep the mocked CLI deterministic without starting a
                        # thread pool in the test's isolated event loop.
                        result = loop.create_future()
                        result.set_result(call(*args))
                        return result

                    with _mock_http(handler), \
                            patch('lari.hermes.client._ask_cli', return_value='risposta CLI') as cli, \
                            patch.object(loop, 'run_in_executor', side_effect=run_cli_inline) as executor, \
                            self.assertLogs('lari', level='WARNING') as logs:
                        reply = await stream_hermes('ciao', self.settings, on_delta=delta)
                    cli.assert_called_once_with('ciao', settings=self.settings)
                    executor.assert_called_once()
                    handler.assert_called_once()
                    self.assertNotIn('X-Hermes-Session-Id', handler.call_args.args[0].headers)
                    self.assertIsInstance(reply, HermesReply)
                    self.assertEqual(reply, 'risposta CLI')
                    self.assertIsNone(reply.session_id)
                    self.assertIsNone(reply.run_id)
                    if delta is not None:
                        delta.assert_awaited_once_with('risposta CLI')
                    self.assertIn('fallback hermes chat -q', logs.output[0])
                    self.assertIn(error_type.__name__, logs.output[0])
                    self.assertNotIn('private', logs.output[0])
                    self.assertNotIn('secret', logs.output[0])

    async def test_open_stream_error_never_calls_cli_and_preserves_partial_delta(self):
        for error_type in (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError):
            for chunks in ((), (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',)):
                with self.subTest(error_type=error_type, has_partial=bool(chunks)):
                    error = error_type('stream failed')
                    handler = Mock(return_value=httpx.Response(200, stream=_Stream(chunks, error)))
                    delta = AsyncMock()
                    with _mock_http(handler), patch('lari.hermes.client._ask_cli') as cli:
                        with self.assertRaises(error_type) as raised:
                            await stream_hermes('ciao', self.settings, on_delta=delta)
                    self.assertIs(raised.exception, error)
                    cli.assert_not_called()
                    handler.assert_called_once()
                    if chunks:
                        delta.assert_awaited_once_with('partial')
                    else:
                        delta.assert_not_awaited()

    async def test_session_connect_failure_never_calls_cli(self):
        for error_type in (httpx.ConnectError, httpx.ConnectTimeout):
            with self.subTest(error_type=error_type):
                error = error_type('connect failed')
                handler = Mock(side_effect=error)
                delta = AsyncMock()
                with _mock_http(handler), patch('lari.hermes.client._ask_cli') as cli:
                    with self.assertRaises(error_type) as raised:
                        await stream_hermes('ciao', self.settings, session_id='prior-id', on_delta=delta)
                self.assertIs(raised.exception, error)
                cli.assert_not_called()
                delta.assert_not_awaited()
                handler.assert_called_once()
                self.assertEqual(handler.call_args.args[0].headers['X-Hermes-Session-Id'], 'prior-id')

    async def test_http_error_never_calls_cli(self):
        handler = Mock(return_value=httpx.Response(503))
        delta = AsyncMock()
        with _mock_http(handler), patch('lari.hermes.client._ask_cli') as cli:
            with self.assertRaises(httpx.HTTPStatusError):
                await stream_hermes('ciao', self.settings, on_delta=delta)
        cli.assert_not_called()
        delta.assert_not_awaited()
        handler.assert_called_once()

    async def test_non_connect_transport_failure_before_stream_never_calls_cli(self):
        # A request may already have been sent before headers arrive. These
        # failures do not prove that submitting the same turn again is safe.
        for error_type in (httpx.ReadError, httpx.ReadTimeout, httpx.WriteError,
                           httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type):
                error = error_type('transport failed')
                handler = Mock(side_effect=error)
                with _mock_http(handler), patch('lari.hermes.client._ask_cli') as cli:
                    with self.assertRaises(error_type) as raised:
                        await stream_hermes('ciao', self.settings)
                self.assertIs(raised.exception, error)
                cli.assert_not_called()
                handler.assert_called_once()

    async def test_invalid_or_incomplete_stream_never_calls_cli_and_keeps_prior_delta(self):
        for tail in (b'', b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
                     b'data: {broken}\n\n'):
            with self.subTest(tail=tail):
                chunks = (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', tail)
                handler = Mock(return_value=httpx.Response(200, stream=_Stream(chunks)))
                delta = AsyncMock()
                with _mock_http(handler), patch('lari.hermes.client._ask_cli') as cli:
                    with self.assertRaises(RuntimeError):
                        await stream_hermes('ciao', self.settings, on_delta=delta)
                cli.assert_not_called()
                handler.assert_called_once()
                delta.assert_awaited_once_with('partial')


if __name__ == '__main__':
    unittest.main()
