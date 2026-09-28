import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class LocaleDefaultsTests(unittest.TestCase):
    def test_english_locale_defaults_survive_blank_optional_environment_values(self):
        with tempfile.TemporaryDirectory() as home:
            env = os.environ.copy()
            for key in list(env):
                if key.startswith(("LARI_", "BUDDY_")):
                    env.pop(key)
            env.update({
                "HOME": home,
                "LARI_LANGUAGE": "en",
                "LARI_WAKE_PHRASE": "",
                "LARI_TTS_VOICE": "",
                "LARI_STT_LANG": "",
                "LARI_VOSK_MODEL_DIR": "",
                "LARI_HERMES_ROOT": "",
            })
            script = (
                "import json, pathlib, server; "
                "print(json.dumps({"
                "'language': server.LANGUAGE, "
                "'wake': server.WAKE_PHRASE, "
                "'voice': server.TTS_VOICE, "
                "'stt_language': server.STT_LANG, "
                "'vosk_model': server.VOSK_MODEL_DIR.name, "
                "'hermes_root_is_default': server.HERMES_ROOT == pathlib.Path.home() / '.hermes' / 'hermes-agent'"
                "}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            config = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(config["language"], "en")
        self.assertEqual(config["wake"], "Hey Lari")
        self.assertEqual(config["voice"], "en-US-JennyNeural")
        self.assertEqual(config["stt_language"], "en")
        self.assertEqual(config["vosk_model"], "vosk-model-small-en")
        self.assertTrue(config["hermes_root_is_default"])


if __name__ == "__main__":
    unittest.main()
