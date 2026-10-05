"""The UI brand must not silently change the known-working voice configuration."""

import unittest
from pathlib import Path
from lari.config import load_settings

ROOT = Path(__file__).resolve().parent.parent


def inspect_config(extra):
    settings = load_settings(extra)
    return {"wake": settings.wake_phrase, "backend": settings.stt_backend,
            "vosk": settings.vosk_model_dir.name, "aliases": list(settings.wake_config.aliases),
            "root": str(settings.hermes_root)}


class WorkingVoiceConfigTests(unittest.TestCase):
    def test_default_model_directory_and_hermes_root_are_portable(self):
        config = inspect_config({})
        self.assertEqual(config["vosk"], "vosk-model-small-it-0.22")
        self.assertEqual(config["root"], str(Path.home() / ".hermes" / "hermes-agent"))

    def test_hermes_root_can_be_selected_without_a_personal_home_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as custom_root:
            settings = load_settings({"LARI_HERMES_ROOT": custom_root})
            self.assertEqual(str(settings.hermes_root), custom_root)

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
