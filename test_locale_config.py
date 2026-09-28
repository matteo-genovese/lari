"""The UI brand must not silently change the known-working voice configuration."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inspect_config(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BUDDY_", "LARI_"))}
    env.update(extra)
    code = (
        "import json, pathlib, server; "
        "print(json.dumps({'wake': server.WAKE_PHRASE, "
        "'backend': server.STT_BACKEND, "
        "'vosk': server.VOSK_MODEL_DIR.name, "
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
            env = {k: v for k, v in os.environ.items() if not k.startswith(("BUDDY_", "LARI_"))}
            env["BUDDY_HERMES_ROOT"] = custom_root
            code = (
                "import sys,types; sys.modules['tools']=types.ModuleType('tools'); "
                "m=types.ModuleType('tools.wake_word'); m._build_engine=None; "
                "sys.modules['tools.wake_word']=m; import server; print(server.HERMES_ROOT)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=env,
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), custom_root)

    def test_legacy_voice_settings_are_not_replaced_by_brand_settings(self):
        config = inspect_config({
            "BUDDY_WAKE_PHRASE": "hey nic",
            "BUDDY_STT_BACKEND": "elevenlabs_realtime",
            "LARI_WAKE_PHRASE": "not-a-runtime-setting",
            "LARI_STT_BACKEND": "whisper",
        })
        self.assertEqual(config["wake"], "hey nic")
        self.assertEqual(config["backend"], "elevenlabs_realtime")


if __name__ == "__main__":
    unittest.main()
