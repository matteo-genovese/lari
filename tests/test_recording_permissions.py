import os
import stat
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from lari import server


class RecordingPermissionsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        previous_umask = os.umask(0o022)
        self.addCleanup(os.umask, previous_umask)
        self.data = b"\x01\x00" * (2 * server.SAMPLE_RATE)

    def assert_recording(self, path):
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with wave.open(str(path), "rb") as wav:
            self.assertEqual(wav.readframes(wav.getnframes()), self.data)

    async def test_session_close_calibration_recording_is_owner_only(self):
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.receive = AsyncMock(return_value={"type": "websocket.disconnect"})
        session = server.Session(websocket, AsyncMock())
        session.recent.extend(self.data)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(server, "BASE_DIR", Path(root)), \
             patch.object(server, "TOKEN", "unit-test-token"), \
             patch.object(server, "Session", return_value=session), \
             patch.object(server.threading, "Thread"), \
             patch.object(server, "_sessions", set()):
            await server.ws_endpoint("unit-test-token", websocket)
            files = list((Path(root) / "calibration").glob("mic_*.wav"))
            self.assertEqual(len(files), 1)
            self.assert_recording(files[0])

    async def test_debug_recording_is_owner_only(self):
        session = server.Session(None, AsyncMock())
        session.recent.extend(self.data)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(server.tempfile, "tempdir", root), \
             patch.object(server, "TOKEN", "unit-test-token"), \
             patch.object(server, "_sessions", {session}):
            response = await server.debug_last("unit-test-token")
            self.assert_recording(Path(response.path))


if __name__ == "__main__":
    unittest.main()
