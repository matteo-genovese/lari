"""The HTTP adapter forwards satellite input without owning voice work."""
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from lari import server


class WebSocketAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_controls_and_pcm_are_forwarded_and_cleanup_is_awaited(self):
        ws = Mock(accept=AsyncMock(), close=AsyncMock(),
                  send_json=AsyncMock(), send_bytes=AsyncMock())
        ws.receive = AsyncMock(side_effect=[
            {"bytes": b"\x01\x00"},
            {"text": "{"},
            {"text": "[]"},
            {"text": "null"},
            {"text": '{"type": "unknown"}'},
            {"text": json.dumps({"type": "ping"})},
            {"text": json.dumps({"type": "playback_done", "turn": 4, "status": "failed"})},
            {"text": json.dumps({"type": "interrupt", "turn": 3})},
            {"type": "websocket.disconnect"},
        ])
        session = Mock(
            state="listening", start=AsyncMock(), disconnect=AsyncMock(),
            on_audio=AsyncMock(), send_json=AsyncMock(),
            playback_completed=AsyncMock(), interrupt=AsyncMock(return_value=False),
        )
        active = set()
        with patch.object(server, "TOKEN", "test-token"), \
             patch.object(server, "Session", return_value=session) as factory, \
             patch.object(server, "_sessions", active):
            await server.ws_endpoint("test-token", ws)
        ws.accept.assert_awaited_once()
        factory.assert_called_once()
        self.assertIs(factory.call_args.kwargs["settings"], server._SETTINGS)
        send_json, send_audio = factory.call_args.args
        await send_json({"type": "pong", "state": "listening"})
        await send_audio(b"ID3\x00\xff")
        ws.send_json.assert_awaited_once_with({"type": "pong", "state": "listening"})
        ws.send_bytes.assert_awaited_once_with(b"ID3\x00\xff")
        session.start.assert_awaited_once()
        session.on_audio.assert_awaited_once_with(b"\x01\x00")
        session.playback_completed.assert_awaited_once_with(4, status="failed")
        session.interrupt.assert_awaited_once_with(3)
        self.assertEqual([call.args[0] for call in session.send_json.await_args_list], [
            {"type": "pong", "state": "listening"},
            {"type": "interrupt_rejected", "turn": 3},
        ])
        session.disconnect.assert_awaited_once()
        self.assertFalse(active)

    async def test_invalid_token_does_not_start_a_session(self):
        ws = Mock(accept=AsyncMock(), close=AsyncMock())
        with patch.object(server, "TOKEN", "test-token"), \
             patch.object(server, "Session") as factory:
            await server.ws_endpoint("wrong-token", ws)
        ws.close.assert_awaited_once_with(code=4403)
        ws.accept.assert_not_awaited()
        factory.assert_not_called()
