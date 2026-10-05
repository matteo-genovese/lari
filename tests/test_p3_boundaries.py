"""Subsystem boundaries and calibration naming regressions."""
import ast
import hashlib
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class WakeBoundaryTests(unittest.TestCase):
    def test_wake_imports_and_runs_without_server(self):
        source = '''
import sys
from unittest.mock import Mock, patch
import numpy as np
from lari.wake.config import WakeConfig, build_wake_config
from lari.wake import detector, confirm
cfg = build_wake_config("ehi lari", aliases=("ehi lar",))
assert isinstance(cfg, WakeConfig)
assert detector.wake_command("Ehi Lare, dimmi", cfg) == "dimmi"
recognizer = Mock()
recognizer.FinalResult.return_value = '{"text": "ehi lar"}'
with patch.dict(sys.modules, {"vosk": Mock(KaldiRecognizer=Mock(return_value=recognizer))}), patch.object(detector, "get_vosk"):
    assert detector.vosk_wake(np.zeros(48000, dtype=np.int16), cfg)
with patch.object(confirm, "transcribe_vosk", return_value=""), patch.object(confirm, "transcribe", side_effect=AssertionError("slow model on doubt")):
    assert confirm.confirm_candidate(np.zeros(48000, dtype=np.int16), cfg)
with patch.object(confirm, "transcribe_vosk", return_value="che tempo fa"), patch.object(confirm, "transcribe", return_value="vorrei andare a fare shopping"):
    assert not confirm.confirm_candidate(np.zeros(48000, dtype=np.int16), cfg)
assert "lari.server" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wake_functions_keep_reference_ast(self):
        reference = subprocess.check_output(
            ["git", "show", "bb8f23e:lari/server.py"], cwd=ROOT, text=True)
        original = {node.name: node for node in ast.parse(reference).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        moved = {
            "lari/wake/detector.py": ("_wake_engine_builder", "wake_command", "resolve_command",
                                      "get_vosk", "vosk_wake", "resolve_vosk_command", "make_engine"),
            "lari/wake/confirm.py": ("confirm_candidate",),
            "lari/stt/vosk.py": ("transcribe_vosk",),
        }
        for file, names in moved.items():
            current = {node.name: node for node in ast.parse((ROOT / file).read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for name in names:
                with self.subTest(function=name):
                    self.assertEqual(ast.dump(original[name]), ast.dump(current[name]))
        # Exact P2 bytes, independent of subsequent commits deleting the old path.
        self.assertEqual(hashlib.sha256((ROOT / "lari/wake/config.py").read_bytes()).hexdigest(),
                         "b3393437889f1f2118a1234d124d17ca01e10bb3f53ac44cd420f4f55e96bb07")


class STTBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_mocked_provider_dispatch_needs_no_credentials(self):
        from lari.stt import dispatch, providers
        pcm = np.zeros(1600, dtype=np.int16)
        with patch.object(dispatch, "STT_BACKEND", "groq"), \
             patch.object(providers, "transcribe", return_value="ehi lari, dimmi") as provider:
            self.assertEqual(dispatch.stt_transcribe(pcm), "ehi lari, dimmi")
            provider.assert_called_once_with(pcm, "groq")

    async def test_realtime_failure_only_uses_local_fallback(self):
        from lari.stt import dispatch, providers, realtime
        pcm = np.zeros(1600, dtype=np.int16)
        stream = Mock(finish=AsyncMock(side_effect=realtime.RealtimeUnavailable("failure")))
        with patch.object(dispatch, "transcribe_vosk", return_value="comando locale") as local, \
             patch.object(providers, "transcribe", side_effect=AssertionError("paid fallback")):
            self.assertEqual(await dispatch.transcribe_realtime_or_batch(pcm, stream, 1),
                             "comando locale")
            self.assertEqual(await dispatch.transcribe_realtime_or_batch(pcm, None, 2, True),
                             "comando locale")
            self.assertEqual(local.call_count, 2)

    async def test_unavailable_vosk_falls_back_to_local_whisper(self):
        from lari.stt import dispatch, providers
        with patch.object(dispatch, "transcribe_vosk", side_effect=RuntimeError), \
             patch.object(dispatch, "transcribe", return_value="locale"), \
             patch.object(providers, "transcribe", side_effect=AssertionError("paid fallback")):
            self.assertEqual(dispatch._transcribe_local_fallback(np.zeros(1600, dtype=np.int16)),
                             "locale")


class CalibrationBoundaryTests(unittest.TestCase):
    def test_same_second_sessions_have_unique_private_wavs_and_rotate_eight(self):
        from lari import audio
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(audio.time, "strftime", return_value="20261005_120000"):
            paths = []
            for index in range(10):
                session = audio.SessionAudio()
                session._init_audio()
                session.recent.extend(np.full(32000, index, dtype=np.int16).tobytes())
                path = session.save_calibration(Path(directory))
                paths.append(path)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(len(set(paths)), 10)
            self.assertEqual(len(list(Path(directory).glob("mic_*.wav"))), 8)
