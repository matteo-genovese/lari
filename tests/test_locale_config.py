"""The UI brand must not silently change the known-working voice configuration."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def inspect_config(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LARI_", "BRAND_"))}
    env.update(extra)
    code = (
        "import json, pathlib; from lari import server; "
        "print(json.dumps({'wake': server.WAKE_PHRASE, "
        "'backend': server.STT_BACKEND, "
        "'vosk': server.VOSK_MODEL_DIR.name, "
        "'aliases': list(server.WAKE_CONFIG.aliases), "
        "'root': str(server.HERMES_ROOT)}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


class WorkingVoiceConfigTests(unittest.TestCase):
    def test_default_model_directory_and_hermes_root_are_portable(self):
        config = inspect_config({})
        self.assertEqual(config["vosk"], "vosk-model-small-it-0.22")
        self.assertEqual(config["root"], str(Path.home() / ".hermes" / "hermes-agent"))

    def test_hermes_root_can_be_selected_without_a_personal_home_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as custom_root:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("LARI_", "BRAND_"))}
            env["LARI_HERMES_ROOT"] = custom_root
            code = (
                "import sys,types; sys.modules['tools']=types.ModuleType('tools'); "
                "m=types.ModuleType('tools.wake_word'); m._build_engine=None; "
                "sys.modules['tools.wake_word']=m; from lari import server; print(server.HERMES_ROOT)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=env,
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), custom_root)

    def test_default_wake_phrase_follows_the_language(self):
        self.assertEqual(inspect_config({"LARI_STT_LANG": "it"})["wake"], "ehi lari")
        self.assertEqual(inspect_config({"LARI_STT_LANG": "en"})["wake"], "hey lari")

    def test_wake_aliases_reach_the_runtime_config(self):
        config = inspect_config({
            "LARI_WAKE_PHRASE": "ehi lari",
            "LARI_WAKE_ALIASES": "ehi lare, hey lar",
        })
        self.assertEqual(config["aliases"], ["ehi lare", "hey lar"])

    def test_legacy_voice_settings_are_not_replaced_by_brand_settings(self):
        config = inspect_config({
            "LARI_WAKE_PHRASE": "ciao luna",
            "LARI_STT_BACKEND": "elevenlabs_realtime",
            "BRAND_WAKE_PHRASE": "not-a-runtime-setting",
            "BRAND_STT_BACKEND": "whisper",
        })
        self.assertEqual(config["wake"], "ciao luna")
        self.assertEqual(config["backend"], "elevenlabs_realtime")


if __name__ == "__main__":
    unittest.main()
