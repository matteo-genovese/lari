"""The default wake path is independent of the optional Hermes checkout."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari.config import load_settings
from lari.session import Session
from lari.wake import detector

ROOT = Path(__file__).resolve().parent.parent


class OptionalWakeImportTests(unittest.TestCase):
    def test_fresh_import_preserves_sys_path(self):
        result = subprocess.run([sys.executable, '-c', '''
import sys
before = sys.path[:]
import lari.wake.detector
assert sys.path == before, (before, sys.path)
assert 'tools.wake_word' not in sys.modules
'''], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_normal_energy_and_vosk_path_without_hermes(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(load_settings({}), hermes_root=Path(directory) / 'absent')
            session = Session(AsyncMock(), AsyncMock(), settings=settings)
            session.recv_queue.put(np.zeros(1280, dtype=np.int16).tobytes())
            session.recv_queue.put(None)
            before = sys.path[:]
            with patch('lari.wake.runtime.make_engine', side_effect=AssertionError('optional engine')):
                session.wake_worker()
            self.assertIsNone(session.engine)
            recognizer = Mock()
            recognizer.FinalResult.return_value = '{"text": "ehi lari"}'
            vosk = Mock(KaldiRecognizer=Mock(return_value=recognizer))
            with patch.dict(sys.modules, {'vosk': vosk}), patch.object(detector, 'get_vosk'):
                self.assertTrue(detector.vosk_wake(np.zeros(16000, dtype=np.int16), settings.wake_config))
            self.assertEqual(sys.path, before)
            optional = replace(settings, wake_provider='openwakeword')
            with self.assertRaises(RuntimeError) as error:
                detector.make_engine(optional)
            message = str(error.exception)
            self.assertIn(str(optional.hermes_root / 'tools' / 'wake_word.py'), message)
            self.assertIn('missing Hermes module', message)
            self.assertIn('LARI_HERMES_ROOT', message)
            self.assertEqual(sys.path, before)

    def test_configured_module_and_lazy_sibling_import_preserve_path(self):
        before = sys.path[:]
        # A distinctive sibling name avoids relying on installed Hermes packages.
        self.addCleanup(sys.modules.pop, '_lari_test_wake_sibling', None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            (root / '_lari_test_wake_sibling.py').write_text('MARKER = "configured engine"\n')
            (root / 'tools' / 'wake_word.py').write_text('''
from dataclasses import dataclass
@dataclass
class Engine:
    marker: str
    config: dict

def _build_engine(cfg):
    from _lari_test_wake_sibling import MARKER
    return Engine(MARKER, cfg)
''')
            settings = replace(load_settings({}), hermes_root=root, wake_provider='openwakeword',
                               wake_sensitivity=.7, confirm_frames=5)
            engine = detector.make_engine(settings)
            self.assertEqual(engine.marker, 'configured engine')
            self.assertEqual(engine.config['provider'], 'openwakeword')
            self.assertEqual(engine.config['phrase'], settings.wake_config.phrase)
            self.assertEqual(engine.config['sensitivity'], .7)
            self.assertEqual(engine.config['confirmation_frames'], 5)
            self.assertEqual(sys.path, before)
            # Do not reuse a globally cached tools.wake_word from a previous root.
            with tempfile.TemporaryDirectory() as other:
                (Path(other) / 'tools').mkdir()
                (Path(other) / 'tools' / 'wake_word.py').write_text(
                    'def _build_engine(cfg): return "other checkout"\n')
                self.assertEqual(detector.make_engine(replace(settings, hermes_root=Path(other))),
                                 'other checkout')
        self.assertEqual(sys.path, before)

    def test_optional_failures_restore_path_and_explain_missing_dependency(self):
        before = sys.path[:]
        modules = set(sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            path = root / 'tools' / 'wake_word.py'
            settings = replace(load_settings({}), hermes_root=root, wake_provider='openwakeword')
            for source in ('import _lari_missing_wake_dependency\n',
                           'def _build_engine(cfg):\n    import _lari_missing_wake_dependency\n',
                           'def _build_engine(cfg):\n    raise ValueError("bad engine")\n',
                           'not_a_builder = True\n'):
                with self.subTest(source=source):
                    path.write_text(source)
                    with self.assertRaises((RuntimeError, ValueError)) as error:
                        detector.make_engine(settings)
                    if '_lari_missing_wake_dependency' in source:
                        self.assertIn('_lari_missing_wake_dependency', str(error.exception))
                        self.assertIn(str(path), str(error.exception))
                        self.assertIn('install', str(error.exception))
                    elif 'not_a_builder' in source:
                        self.assertIn('_build_engine', str(error.exception))
                        self.assertIn(str(path), str(error.exception))
                    self.assertEqual(sys.path, before)
                    self.assertFalse(any(name.startswith('_lari_hermes_wake_')
                                         for name in set(sys.modules) - modules))
