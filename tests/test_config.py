"""Configuration snapshots are pure, validated and safe to inspect."""
import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from lari import config, usage
from lari.stt import providers as stt_backends, realtime


class SettingsTests(unittest.TestCase):
    def test_load_is_pure_and_uses_only_the_supplied_mapping(self):
        env = {"LARI_TOKEN": " installation-token ", "LARI_PORT": "9000"}
        with patch.dict(os.environ, {"LARI_TOKEN": "other-token"}):
            before = dict(os.environ)
            settings = config.load_settings(env)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(env, {"LARI_TOKEN": " installation-token ", "LARI_PORT": "9000"})
        self.assertEqual(settings.token, "installation-token")
        self.assertEqual(settings.port, 9000)

    def test_defaults_keep_runtime_paths_and_model_choices(self):
        settings = config.load_settings({})
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(settings.port, 8643)
        self.assertEqual(settings.hermes_root, Path.home() / ".hermes" / "hermes-agent")
        self.assertEqual(settings.vosk_model_dir, root / "models/vosk-model-small-it-0.22")
        self.assertEqual(settings.usage_ledger, root / "usage.json")
        self.assertEqual(settings.realtime_usage_file, root / ".realtime_stt_usage.json")
        self.assertEqual(settings.realtime_daily_seconds, 600.0)
        self.assertEqual(settings.stt_backend, "whisper")
        self.assertEqual(settings.stt_model, "base")
        self.assertEqual(settings.tts_voice, "it-IT-ElsaNeural")
        self.assertIsNone(settings.usage_eur_per_min)

    def test_settings_are_immutable(self):
        settings = config.load_settings({})
        with self.assertRaises(FrozenInstanceError):
            settings.port = 9000

    def test_repr_never_contains_credentials(self):
        names = ("LARI_TOKEN", "LARI_HERMES_KEY", "ELEVENLABS_API_KEY",
                 "GROQ_API_KEY", "OPENAI_API_KEY")
        secrets = {name: f"secret-{name}" for name in names}
        rendered = repr(config.load_settings(secrets))
        for secret in secrets.values():
            self.assertNotIn(secret, rendered)

    def test_invalid_numbers_and_boolean_fail_early_without_echoing_values(self):
        cases = {
            "LARI_PORT": ("bad-secret", "0", "65536"),
            "LARI_CONFIRM_FRAMES": ("1.5", "0"),
            "LARI_STREAM_TTS_QUEUE_MAX": ("0",),
            "LARI_STREAM_TEXT_MAX_CHARS": ("-1",),
            "LARI_STREAM_SENTENCE_MAX_CHARS": ("0",),
            "LARI_SENSITIVITY": ("-0.1", "1.1"),
            "LARI_AGENT_TIMEOUT": ("0", "nan", "inf"),
            "LARI_CLI_TIMEOUT": ("-1",),
            "LARI_VAD_MIN_RMS": ("-1",),
            "LARI_REALTIME_DAILY_SECONDS": ("nan", "-1"),
            "LARI_USAGE_EUR_PER_MIN": ("bad-secret", "-1"),
            "LARI_WAKE_CONFIRM": ("bad-secret", "2"),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name) as error:
                        config.load_settings({name: value})
                    self.assertNotIn("bad-secret", str(error.exception))

    def test_zero_optional_limits_and_boolean_flags_keep_their_meaning(self):
        settings = config.load_settings({
            "LARI_REALTIME_DAILY_SECONDS": "0", "LARI_FOLLOWUP_S": "0",
            "LARI_AMBIENT_PAUSE": "0", "LARI_ECHO_MUTE": "0",
            "LARI_USAGE_EUR_PER_MIN": "0", "LARI_WAKE_CONFIRM": " 0 ",
        })
        self.assertEqual(settings.realtime_daily_seconds, 0)
        self.assertEqual(settings.followup_s, 0)
        self.assertEqual(settings.usage_eur_per_min, 0)
        self.assertFalse(settings.wake_confirm)
        self.assertTrue(config.load_settings({"LARI_WAKE_CONFIRM": "1"}).wake_confirm)

    def test_wake_settings_preserve_language_aliases_vocabulary_and_override(self):
        env = {
            "LARI_STT_LANG": "it", "LARI_WAKE_PHRASE": "ciao luna",
            "LARI_WAKE_ALIASES": "ehi luna", "LARI_STT_KEYTERMS": "Aurora",
            "LARI_WAKE_RE": r"^ciao luna",
        }
        settings = config.load_settings(env)
        self.assertEqual(settings.wake_phrase, "ciao luna")
        self.assertEqual(settings.wake_config.aliases, ("ehi luna",))
        self.assertIn("Aurora", settings.wake_config.batch_keyterms)
        self.assertEqual(settings.wake_config.command("ciao luna, dimmi"), "dimmi")
        self.assertEqual(config.load_settings({}).wake_phrase, "hey lari")
        self.assertEqual(config.load_settings({"LARI_STT_LANG": "it"}).wake_phrase, "ehi lari")

    def test_process_cache_keeps_one_snapshot_and_preserves_offline_override(self):
        config.get_settings.cache_clear()
        self.addCleanup(config.get_settings.cache_clear)
        with patch.dict(os.environ, {"LARI_PORT": "9100", "HF_HUB_OFFLINE": "0"}):
            first = config.get_settings()
            with patch.dict(os.environ, {"LARI_PORT": "9200"}):
                self.assertIs(config.get_settings(), first)
                self.assertEqual(config.get_settings().port, 9100)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
        config.get_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            config.get_settings()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_ledgers_and_stt_use_injected_settings(self):
        settings = config.load_settings({
            "LARI_USAGE_LEDGER": "/tmp/config-usage.json",
            "LARI_REALTIME_USAGE_FILE": "/tmp/config-realtime.json",
            "LARI_REALTIME_DAILY_SECONDS": "120",
            "GROQ_API_KEY": "injected-key", "LARI_GROQ_MODEL": "custom-model",
        })
        self.assertEqual(usage.UsageLedger(settings=settings).path, Path("/tmp/config-usage.json"))
        budget = realtime.DailyAudioBudget(settings=settings)
        self.assertEqual(budget.path, Path("/tmp/config-realtime.json"))
        self.assertEqual(budget.daily_seconds, 120)
        with patch.object(stt_backends, "_post_multipart", return_value={"text": "ciao"}) as post:
            self.assertEqual(stt_backends.transcribe([0, 1], "groq", settings=settings), "ciao")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer injected-key")
        self.assertEqual(dict(post.call_args.kwargs["data_tuples"])["model"], "custom-model")
