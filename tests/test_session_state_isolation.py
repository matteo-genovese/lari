"""The suite must never persist test conversations in the user's default state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


class SessionStateIsolationTests(unittest.TestCase):
    def test_suite_does_not_create_default_session_state(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, HOME=home)
            env.pop("XDG_STATE_HOME", None)
            env.pop("LARI_SESSION_STATE", None)
            # Run the actual test fixtures and save paths in a fresh process.
            # Exclude this module to avoid recursively starting the guard.
            result = subprocess.run([sys.executable, "-c", '''
import unittest

def cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from cases(test)
        elif not test.id().startswith("test_session_state_isolation."):
            yield test

suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner().run(unittest.TestSuite(cases(suite)))
raise SystemExit(not result.wasSuccessful())
'''], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            default = Path(home) / ".local/state/lari/sessions.json"
            self.assertFalse(default.exists(),
                             "The suite saved a test conversation under HOME")
            self.assertFalse(default.with_suffix(".json.lock").exists(),
                             "The suite attempted to save session state under HOME")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
