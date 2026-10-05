"""Backend names fail fast without credentials, model loading or network calls."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lari import config
from lari.stt import dispatch, providers
from lari.stt.realtime import REALTIME_BACKEND


PCM = np.zeros(320, dtype=np.int16)


class BackendConfigTests(unittest.TestCase):
    def test_all_supported_backends_are_accepted_without_credentials(self):
        for backend in config.STT_BACKENDS:
            with self.subTest(backend=backend):
                settings = config.load_settings({"LARI_STT_BACKEND": backend})
                self.assertEqual(settings.stt_backend, backend)
                self.assertEqual(settings.elevenlabs_api_key, "")
                self.assertEqual(settings.groq_api_key, "")
                self.assertEqual(settings.openai_api_key, "")

    def test_unknown_backends_are_rejected_with_the_expected_message(self):
        for backend in ("bogus", "elevenlab_realtime", "", "Whisper"):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError) as error:
                    config.load_settings({"LARI_STT_BACKEND": backend})
                self.assertEqual(
                    str(error.exception),
                    f"LARI_STT_BACKEND: unsupported value {backend!r}; expected one of: "
                    "whisper, vosk, elevenlabs_realtime, elevenlabs, groq, openai",
                )

    def test_default_is_whisper_and_backend_whitespace_is_stripped(self):
        self.assertEqual(config.load_settings({}).stt_backend, "whisper")
        self.assertEqual(config.load_settings({"LARI_STT_BACKEND": " groq "}).stt_backend, "groq")

    def test_canonical_backends_match_the_dispatch_implementations(self):
        self.assertEqual(
            set(config.STT_BACKENDS),
            {"whisper", "vosk", REALTIME_BACKEND} | set(providers.BACKENDS),
        )
        self.assertEqual(providers.BACKENDS, tuple(providers.KEY_ENV))

    def test_wake_provider_remains_extensible(self):
        self.assertEqual(config.load_settings({"LARI_WAKE_PROVIDER": "auto"}).wake_provider, "auto")


class BackendDispatchTests(unittest.TestCase):
    def test_sync_dispatch_rejects_directly_constructed_unknown_backend(self):
        settings = replace(config.load_settings({}), stt_backend="bogus")
        calls = (
            lambda: dispatch.stt_transcribe(PCM, settings),
            lambda: dispatch.decode_utterance(PCM, False, settings=settings),
            lambda: dispatch.command_for_turn("ciao", False, False, settings),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "LARI_STT_BACKEND: unsupported value 'bogus'"):
                    call()

    def test_decode_rejects_an_explicit_unknown_backend(self):
        settings = config.load_settings({})
        for backend in ("bogus", "elevenlab_realtime", ""):
            for followup in (False, True):
                with self.subTest(backend=backend, followup=followup):
                    with self.assertRaises(ValueError) as error:
                        dispatch.decode_utterance(PCM, followup, backend=backend, settings=settings)
                    self.assertIn(f"unsupported value {backend!r}", str(error.exception))
                    self.assertIn("LARI_STT_BACKEND", str(error.exception))
                    for allowed in config.STT_BACKENDS:
                        self.assertIn(allowed, str(error.exception))

    def test_raw_dispatch_uses_the_selected_recognizer(self):
        for backend in config.STT_BACKENDS:
            with self.subTest(backend=backend), \
                 patch.object(dispatch, "transcribe", return_value="whisper") as whisper, \
                 patch.object(dispatch, "transcribe_vosk", return_value="vosk") as vosk, \
                 patch.object(dispatch, "_transcribe_local_fallback", return_value="fallback") as fallback, \
                 patch.object(providers, "transcribe", return_value="cloud") as cloud:
                settings = config.load_settings({"LARI_STT_BACKEND": backend})
                text = dispatch.stt_transcribe(PCM, settings)
                selected = {
                    "whisper": whisper, "vosk": vosk, REALTIME_BACKEND: fallback,
                }.get(backend, cloud)
                selected.assert_called_once()
                self.assertEqual(text, selected.return_value)
                for other in (whisper, vosk, fallback, cloud):
                    if other is not selected:
                        other.assert_not_called()
                if selected is cloud:
                    cloud.assert_called_once_with(PCM, backend, settings=settings)


class AsyncBackendDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_dispatch_rejects_directly_constructed_unknown_backend(self):
        settings = replace(config.load_settings({}), stt_backend="bogus")
        with self.assertRaisesRegex(ValueError, "LARI_STT_BACKEND: unsupported value 'bogus'"):
            await dispatch.open_stream(None, settings=settings)
        with self.assertRaisesRegex(ValueError, "LARI_STT_BACKEND: unsupported value 'bogus'"):
            await dispatch.transcribe_turn(PCM, None, 1, False, settings=settings)
