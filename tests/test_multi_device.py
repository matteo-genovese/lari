"""Multi-satellite identity, Hermes routing and public status contract."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode

import httpx

from lari import server, tts
from lari.config import load_settings
from lari.device import validate_device
from lari.session import Session


@contextmanager
def mock_http(handler):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    with patch.object(httpx, "AsyncClient", side_effect=lambda **kw: client(transport=transport, **kw)):
        yield


class MultiDeviceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = load_settings({
            "LARI_TOKEN": "test-token", "LARI_SESSION_KEY": "test-session",
            "LARI_HERMES_KEY": "test-api-key",
        })

    def make_session(self, device="default"):
        return Session(AsyncMock(), AsyncMock(), settings=self.settings, device=device)

    async def test_hermes_headers_for_parallel_sessions_and_default(self):
        received = []

        def handler(request):
            received.append((request.headers["X-Hermes-Session-Key"],
                             request.headers.get("X-Hermes-Session-Id")))
            key = request.headers["X-Hermes-Session-Key"]
            if json.loads(request.content)["stream"]:
                return httpx.Response(200, headers={"X-Hermes-Session-Id": key + "-id"},
                    content=b'data: {"choices":[{"delta":{"content":"Ciao."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            return httpx.Response(200, headers={"X-Hermes-Session-Id": key + "-id"},
                                  json={"choices": [{"message": {"content": "Ciao."}}]})

        satellites = [self.make_session(name) for name in ("phone", "desktop", "default")]
        with mock_http(handler), patch.object(tts, "tts", AsyncMock(return_value=b"mp3")):
            await asyncio.gather(*(s._ask_hermes("ciao") for s in satellites))
            await asyncio.gather(*(s._stream_hermes_speak("ciao", 0) for s in satellites))
        self.assertCountEqual(received, [
            ("test-session-phone", None), ("test-session-desktop", None), ("test-session", None),
            ("test-session-phone", "test-session-phone-id"),
            ("test-session-desktop", "test-session-desktop-id"),
            ("test-session", "test-session-id"),
        ])
        self.assertEqual(self.settings.session_key, "test-session")
        self.assertIs(satellites[-1]._hermes_settings, self.settings)
        for satellite in satellites:
            self.assertIs(satellite._settings, self.settings)
            await satellite.disconnect()

    async def websocket(self, query="", token="test-token"):
        """Exercise actual ASGI routing/query decoding without starting wake."""
        messages = []
        receive = AsyncMock(side_effect=[{"type": "websocket.connect"},
                                        {"type": "websocket.disconnect", "code": 1000}])
        async def send(message):
            messages.append(message)
        scope = {"type": "websocket", "asgi": {"version": "3.0"},
                 "scheme": "ws", "path": f"/{token}/ws", "query_string": query.encode(),
                 "headers": [], "client": ("127.0.0.1", 1234), "server": ("test", 80),
                 "root_path": "", "subprotocols": []}
        satellite = Mock(state="listening", start=AsyncMock(), disconnect=AsyncMock())
        active = set()
        with patch.object(server, "TOKEN", "test-token"), \
             patch.object(server, "Session", return_value=satellite) as factory, \
             patch.object(server, "_sessions", active):
            await server.app(scope, receive, send)
        self.assertFalse(active)
        return messages, factory, satellite

    async def test_invalid_device_closes_like_invalid_token(self):
        invalid = ["Phone", "PHONE", "phone_desktop", "phone desktop", "a" * 33,
                   "", "../", "téléphone", "手机", "-phone", "phone\n"]
        token_messages, _, _ = await self.websocket(token="wrong-token")
        for name in invalid:
            with self.subTest(device=name):
                with self.assertRaises(ValueError):
                    validate_device(name)
                with self.assertRaises(ValueError):
                    self.make_session(name)
                messages, factory, satellite = await self.websocket(urlencode({"device": name}))
                self.assertEqual(messages, token_messages)
                self.assertEqual(messages, [{"type": "websocket.close", "code": 4403, "reason": ""}])
                factory.assert_not_called()
                satellite.start.assert_not_awaited()

    async def test_valid_names_and_legacy_path(self):
        for name in ("default", "phone", "desktop", "0", "phone-2", "a" * 32):
            with self.subTest(device=name):
                messages, factory, satellite = await self.websocket(urlencode({"device": name}))
                self.assertEqual(messages[0]["type"], "websocket.accept")
                self.assertEqual(factory.call_args.kwargs["device"], name)
                satellite.disconnect.assert_awaited_once()
        messages, factory, satellite = await self.websocket()
        self.assertEqual(messages[0]["type"], "websocket.accept")
        self.assertEqual(factory.call_args.kwargs["device"], "default")

    async def get_status(self, path, active=()):
        with patch.object(server, "TOKEN", "test-token"), \
             patch.object(server, "_SETTINGS", self.settings), \
             patch.object(server, "_sessions", set(active)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                         base_url="http://test") as client:
                return await client.get(path)

    async def test_status_shape_sorting_no_secrets_and_one_row_per_connection(self):
        phone = self.make_session("phone")
        desktop = self.make_session("desktop")
        other_phone = self.make_session("phone")
        phone.state, phone.turn = "thinking", 2
        desktop.state, desktop.turn = "recording", 3
        other_phone.state, other_phone.turn = "speaking", 5
        expected = {"version": 1, "phrase": self.settings.wake_config.display,
                    "stt_backend": self.settings.stt_backend, "devices": [
                        {"device": "desktop", "connected": True, "state": "recording", "turn": 3},
                        {"device": "phone", "connected": True, "state": "speaking", "turn": 5},
                        {"device": "phone", "connected": True, "state": "thinking", "turn": 2},
                    ]}
        response = await self.get_status("/test-token/status", [phone, desktop, other_phone])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.json(), expected)
        for secret in ("test-token", "test-session", "test-api-key"):
            self.assertNotIn(secret, response.text)
        empty = await self.get_status("/test-token/status")
        self.assertEqual(empty.json()["devices"], [])
        for satellite in (phone, desktop, other_phone):
            await satellite.disconnect()

    async def test_status_has_only_token_route_and_rejects_wrong_or_unconfigured_token(self):
        self.assertNotIn("/status", [route.path for route in server.app.routes])
        # The bare path now uses FastAPI's generic /{token}/ page redirect.
        missing = await self.get_status("/status")
        self.assertEqual(missing.status_code, 307)
        self.assertEqual(missing.headers["location"], "http://test/status/")
        self.assertEqual((await self.get_status("/wrong-token/status")).status_code, 403)
        with patch.object(server, "TOKEN", ""):
            self.assertEqual((await server.satellite_status("test-token")).status_code, 403)

    async def test_status_tracks_live_state_and_disconnect(self):
        active = set()
        satellite = self.make_session("phone")
        checked = []
        async def receive():
            await satellite.set_state("thinking")
            satellite.turn = 8
            checked.append((await self.get_status("/test-token/status", active)).json()["devices"])
            return {"type": "websocket.disconnect"}
        ws = Mock(accept=AsyncMock(), close=AsyncMock(), receive=receive)
        with patch.object(server, "TOKEN", "test-token"), \
             patch.object(server, "Session", return_value=satellite), \
             patch.object(server, "_sessions", active), \
             patch.object(satellite, "start", AsyncMock()):
            await server.ws_endpoint("test-token", ws, device="phone")
        self.assertEqual(checked, [[{"device": "phone", "connected": True,
                                    "state": "thinking", "turn": 8}]])
        self.assertFalse(active)

    async def test_backend_and_session_logs_keep_parallel_device_context(self):
        phone, desktop = self.make_session("phone"), self.make_session("desktop")
        def handler(request):
            raise httpx.ConnectError("offline")
        with mock_http(handler), patch("lari.hermes.client._ask_cli", return_value="Ciao."), \
             self.assertLogs("lari", level="INFO") as logs:
            await asyncio.gather(phone._ask_hermes("ciao"), desktop._ask_hermes("ciao"))
            await phone.disconnect()
            await desktop.disconnect()
        self.assertEqual(len(logs.output), 2)
        self.assertTrue(any("[device=phone]" in line for line in logs.output))
        self.assertTrue(any("[device=desktop]" in line for line in logs.output))


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser URL checks")
class DevicePageTests(unittest.TestCase):
    def test_real_page_websocket_url_propagates_device_without_changing_legacy_url(self):
        html = (Path(server.BASE_DIR) / "static/index.html").read_text()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        # Run the real page, exposing URL and microphone setup for assertions.
        script = script.rsplit("})();", 1)[0] + (
            "globalThis.url = WS_URL; globalThis.startMic = startMic;})();"
        )
        harness = r'''
const assert = require("assert/strict"), vm = require("vm");
const note = "Invalid or empty device name; using default.";
(async () => {
for (const [search, suffix, warning] of [["", "", false], ["?raw=1", "", false],
  ["?device=phone", "?device=phone", false], ["?device=desktop", "?device=desktop", false],
  ["?device=Phone", "?device=phone", false], ["?device=%20Phone%20", "?device=phone", false],
  ["?device=../etc", "", true], ["?device=", "", true], ["?device=%20%20", "", true],
  ["?device=phone%20desktop", "", true], ["?device=-phone", "", true],
  ["?device=" + "a".repeat(32), "?device=" + "a".repeat(32), false],
  ["?device=" + "a".repeat(33), "", true], ["?device=phone-2", "?device=phone-2", false],
  ["?raw=1&device=phone", "?device=phone", false]]) {
  for (const protocol of ["https:", "http:"]) {
    const widgets = new Map(), lines = [], captures = [];
    function widget(id){
      if (!widgets.has(id)) widgets.set(id, {
        textContent:"", style:{}, classList:{add(){}, remove(){}, toggle(){}},
        addEventListener(){}, appendChild(node){lines.push(node.lastChild.textContent)},
      });
      return widgets.get(id);
    }
    class AudioContext {
      constructor(){this.state = "running"; this.sampleRate = 48000;}
      createMediaStreamSource(){return {connect(){}};}
      createScriptProcessor(){return {connect(){}};}
    }
    const context = {
      location:{search, protocol, host:"example.invalid", pathname:"/test-token/"}, URLSearchParams,
      document:{getElementById:widget, addEventListener(){},
        createElement(){return {lastChild:{textContent:""}}}},
      window:{AudioContext}, navigator:{mediaDevices:{async getUserMedia(options){
        captures.push(options); return {};
      }}},
      setInterval(){}, clearInterval(){}, setTimeout(){}, clearTimeout(){},
    };
    vm.createContext(context);
    vm.runInContext(SOURCE, context);
    assert.equal(context.url, (protocol === "https:" ? "wss" : "ws") + "://example.invalid/test-token/ws" + suffix);
    assert.deepEqual(lines, warning ? [note] : [], search);
    await context.startMic();
    const raw = new URLSearchParams(search).get("raw") === "1";
    assert.equal(captures[0].audio.echoCancellation, !raw, search);
    assert.equal(captures[0].audio.noiseSuppression, !raw, search);
    if (raw) assert.equal(widget("hint").textContent, "Raw mode active (no echo cancellation).");
  }
}
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
        result = subprocess.run([shutil.which("node"), "-e", harness.replace("SOURCE", json.dumps(script))],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
