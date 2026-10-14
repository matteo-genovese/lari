"""Durable device transcripts with real files and mocked Hermes transport."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from lari import tts
from lari.config import load_settings
from lari.session import Session
from lari.session_state import SessionState


class SessionStateTests(unittest.TestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(root) / "state/sessions.json"
        self.settings = load_settings({"LARI_SESSION_STATE": str(self.path),
                                       "LARI_SESSION_KEY": "private-channel-key",
                                       "LARI_TOKEN": "private-url-token",
                                       "LARI_HERMES_KEY": "private-api-key"})
        self.store = SessionState(self.settings)

    def test_config_paths_and_switch(self):
        self.assertTrue(load_settings({}).session_resume)
        self.assertEqual(load_settings({}).session_state,
                         Path.home() / ".local/state/lari/sessions.json")
        self.assertEqual(load_settings({"XDG_STATE_HOME": "/tmp/state"}).session_state,
                         Path("/tmp/state/lari/sessions.json"))
        self.assertEqual(load_settings({"XDG_STATE_HOME": "", "LARI_SESSION_STATE": "~/custom"}).session_state,
                         Path.home() / "custom")

    def test_round_trip_permissions_and_no_secrets(self):
        previous = os.umask(0)
        try:
            self.store.save("transcript-123")
        finally:
            os.umask(previous)
        self.assertEqual(SessionState(self.settings).get(), "transcript-123")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        data = self.path.read_text()
        self.assertEqual(list(json.loads(data).values()), ["transcript-123"])
        for secret in (self.settings.session_key, self.settings.token, self.settings.hermes_key):
            self.assertNotIn(secret, data)

    def test_absent_empty_corrupt_unreadable_and_invalid_ids(self):
        self.assertIsNone(self.store.get())
        self.path.parent.mkdir()
        for content in ("", "{broken", "[]", "null", '{"x": "id"}', "{}"):
            self.path.write_text(content)
            self.assertIsNone(self.store.get())
        for value in (None, 12, [], {}, "", "a\r\nb", "a b", "x" * 257, "é"):
            self.path.write_text(json.dumps({self.store.key: value}))
            self.assertIsNone(self.store.get())
        self.path.write_bytes(b"\xff")
        self.assertIsNone(self.store.get())
        with patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertIsNone(self.store.get())

    def test_atomic_replace_unchanged_value_and_failure_cleanup(self):
        self.store.save("old")
        replace_file = os.replace

        def observe(source, target):
            self.assertEqual(self.store.get(), "old")
            self.assertEqual(Path(source).parent, self.path.parent)
            self.assertEqual(stat.S_IMODE(Path(source).stat().st_mode), 0o600)
            self.assertEqual(json.loads(Path(source).read_text())[self.store.key], "new")
            replace_file(source, target)

        with patch("lari.session_state.os.replace", side_effect=observe) as rename:
            self.store.save("new")
        rename.assert_called_once()
        with patch("lari.session_state.os.replace") as rename:
            self.store.save("new")
        rename.assert_not_called()
        with patch("lari.session_state.os.replace", side_effect=OSError), self.assertLogs("lari"):
            self.store.save("failed")
        self.assertEqual(self.store.get(), "new")
        self.assertFalse(list(self.path.parent.glob("*.tmp")))

    def test_parallel_updates_preserve_other_devices_and_conditional_forget(self):
        stores = [SessionState(replace(self.settings, session_key=f"device-{i}")) for i in range(12)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda pair: pair[1].save(f"id-{pair[0]}"), enumerate(stores)))
        self.assertEqual([store.get() for store in stores], [f"id-{i}" for i in range(12)])
        self.store.save("new")
        self.store.forget("old")
        self.assertEqual(self.store.get(), "new")
        self.store.forget("new")
        self.assertIsNone(self.store.get())
        self.assertEqual(stores[0].get(), "id-0")


class SessionResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(root) / "sessions.json"
        self.settings = load_settings({"LARI_SESSION_STATE": str(self.path)})
        self.received = []
        self.ids = {}
        self.statuses = []
        self.stream_body = b'data: {"choices":[{"delta":{"content":"Ciao."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        real_client = self.client_class = httpx.AsyncClient

        def handler(request):
            key = request.headers["X-Hermes-Session-Key"]
            self.received.append((key, request.headers.get("X-Hermes-Session-Id")))
            status = self.statuses.pop(0) if self.statuses else 200
            headers = {"X-Hermes-Session-Id": self.ids.get(key, "id-" + key)}
            if json.loads(request.content)["stream"]:
                return httpx.Response(status, headers=headers, content=self.stream_body)
            return httpx.Response(status, headers=headers,
                                  json={"choices": [{"message": {"content": "Ciao."}}]})

        self.enterContext(patch.object(httpx, "AsyncClient", side_effect=lambda **kw:
                                      real_client(transport=httpx.MockTransport(handler), **kw)))
        self.enterContext(patch.object(tts, "tts", AsyncMock(return_value=b"mp3")))
        self.cli = self.enterContext(patch("lari.hermes.client._ask_cli"))

    def session(self, device="default", settings=None):
        return Session(AsyncMock(), AsyncMock(), settings=settings or self.settings, device=device)

    async def call(self, session, streaming=False):
        if streaming:
            return await session._stream_hermes_speak("ciao", 0)
        return await session._ask_hermes("ciao")

    async def test_device_isolation_restart_rotation_and_default(self):
        for device in ("phone", "desktop", "default"):
            original = self.session(device)
            await self.call(original)
            key = original._hermes_settings.session_key
            self.assertEqual(self.received[-1], (key, None))
            restarted = self.session(device)
            self.ids[key] = "rotated-" + device
            await self.call(restarted, True)
            self.assertEqual(self.received[-1], (key, "id-" + key))
            await self.call(original)
            self.assertEqual(self.received[-1], (key, "rotated-" + device))
        self.assertEqual(len(json.loads(self.path.read_text())), 3)

    async def test_missing_corrupt_or_invalid_file_sends_no_id(self):
        for content in (None, "", "{broken", json.dumps({SessionState(self.settings).key: "bad\nid"})):
            if self.path.exists():
                self.path.unlink()
            if content is not None:
                self.path.write_text(content)
            await self.call(self.session(), True)
            self.assertIsNone(self.received[-1][1])

    async def test_disabled_never_sends_or_saves_id_even_on_same_connection(self):
        SessionState(self.settings).save("saved")
        before = self.path.read_bytes()
        settings = replace(self.settings, session_resume=False)
        session = self.session(settings=settings)
        session.hermes_session_id = "memory-id"
        for streaming in (False, True, False, True):
            await self.call(session, streaming)
        self.assertTrue(all(value is None for _, value in self.received))
        self.assertEqual(self.path.read_bytes(), before)
        self.path.unlink()
        await self.call(self.session(settings=settings))
        self.assertFalse(self.path.exists())

    async def test_4xx_forgets_and_retries_once_without_id_for_both_transports(self):
        for streaming in (False, True):
            for status in (400, 404, 422):
                SessionState(self.settings).save("rejected")
                self.statuses = [status, 200]
                await self.call(self.session(), streaming)
                self.assertEqual(self.received[-2:], [("lari", "rejected"), ("lari", None)])
                self.assertEqual(SessionState(self.settings).get(), "id-lari")
        self.cli.assert_not_called()

    async def test_second_stream_4xx_is_not_retried_and_entry_stays_forgotten(self):
        SessionState(self.settings).save("rejected")
        self.statuses = [404, 404]
        with self.assertRaises(RuntimeError):
            await self.call(self.session(), True)
        self.assertEqual(len(self.received), 2)
        self.assertIsNone(SessionState(self.settings).get())
        self.cli.assert_not_called()

    async def test_header_saved_even_if_stream_is_incomplete(self):
        self.stream_body = b'data: {"choices":[{"delta":{"content":"Parziale."}}]}\n\n'
        with self.assertRaises(RuntimeError):
            await self.call(self.session(), True)
        self.assertEqual(SessionState(self.settings).get(), "id-lari")

    async def test_read_and_write_failures_do_not_fail_a_turn(self):
        for streaming in (False, True):
            with patch.object(Path, "read_text", side_effect=PermissionError), \
                    patch("lari.session_state.tempfile.mkstemp", side_effect=OSError), \
                    self.assertLogs("lari", level="WARNING"):
                await self.call(self.session(), streaming)
            self.assertIsNone(self.received[-1][1])
        self.cli.assert_not_called()

    async def test_same_device_overlapping_requests_last_header_wins(self):
        SessionState(self.settings).save("shared")
        entered = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def handler(request):
            nonlocal count
            self.assertEqual(request.headers["X-Hermes-Session-Id"], "shared")
            count += 1
            current = count
            if current == 1:
                entered.set()
                await release.wait()
            else:
                release.set()
            return httpx.Response(200, headers={"X-Hermes-Session-Id": f"response-{current}"},
                                  json={"choices": [{"message": {"content": "Ciao."}}]})

        # Replace the setup's transport for just this overlapping-request check.
        with patch.object(httpx, "AsyncClient", side_effect=lambda **kw:
                          self.client_class(transport=httpx.MockTransport(handler), **kw)):
            first = asyncio.create_task(self.call(self.session()))
            await entered.wait()
            await asyncio.gather(self.call(self.session()), first)
        self.assertEqual(SessionState(self.settings).get(), "response-1")
