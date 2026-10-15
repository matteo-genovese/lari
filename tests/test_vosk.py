"""Unit tests for local Vosk wake and transcription routing."""
from dataclasses import replace
from lari.config import get_settings, load_settings
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari import audio as audio_module
from lari import session as session_module
from lari.wake import detector
from lari.stt import dispatch
from lari.wake.config import build_wake_config


async def send_audio(chunk):
    pass


class VoskRoutingTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.zeros(audio_module.SAMPLE_RATE, dtype=np.int16)

    def test_unaddressed_speech_is_rejected_before_transcription(self):
        with patch.object(dispatch, "vosk_wake", return_value=False) as wake, \
             patch.object(dispatch, "transcribe_vosk") as transcribe:
            result = dispatch.decode_utterance(self.audio, followup=False, backend="vosk", settings=get_settings())
        self.assertIsNone(result)
        wake.assert_called_once()
        transcribe.assert_not_called()

    def test_configured_wake_phrase_is_removed_from_the_command(self):
        transcript = f"{get_settings().wake_config.phrase}, please help"
        with patch.object(dispatch, "vosk_wake", return_value=True), \
             patch.object(dispatch, "transcribe_vosk", return_value=transcript):
            result = dispatch.decode_utterance(self.audio, followup=False, backend="vosk", settings=get_settings())
        self.assertEqual(result, "please help")

    def test_followup_transcription_does_not_require_another_wake(self):
        with patch.object(dispatch, "vosk_wake") as wake, \
             patch.object(dispatch, "transcribe_vosk", return_value="follow-up question"):
            result = dispatch.decode_utterance(self.audio, followup=True, backend="vosk", settings=get_settings())
        self.assertEqual(result, "follow-up question")
        wake.assert_not_called()

    def test_configured_junk_is_cleaned_once_before_dispatch(self):
        settings = load_settings({
            "LARI_STT_BACKEND": "vosk", "LARI_WAKE_PHRASE": "hey nic",
            "LARI_WAKE_OBSERVED_JUNK": "heinrich,inc",
        })
        for transcript, expected in (
            ("heinrich che tempo fa domani a roma", "che tempo fa domani a roma"),
            ("inc che tempo fa domani a roma", "che tempo fa domani a roma"),
            ("che tempo fa domani a roma", "che tempo fa domani a roma"),
            ("vorrei andare a roma", "vorrei andare a roma"),
            ("e che ne dici", "e che ne dici"),
            ("hey nic che tempo fa", "che tempo fa"),
            ("ehi heinrich che tempo fa", "che tempo fa"),
            ("ehi inc che tempo fa", "che tempo fa"),
            ("heinrichson che tempo fa", "heinrichson che tempo fa"),
            ("incredibile che tempo fa", "incredibile che tempo fa"),
            ("che tempo fa heinrich", "che tempo fa heinrich"),
            ("che tempo fa inc", "che tempo fa inc"),
            # The two-token limit applies once, even across both dispatch steps.
            ("ehi heinrich hey che tempo fa", "hey che tempo fa"),
            ("ehi inc hey che tempo fa", "hey che tempo fa"),
            ("heinrich inc heinrich che tempo fa", "heinrich che tempo fa"),
            ("inc heinrich inc che tempo fa", "inc che tempo fa"),
        ):
            with self.subTest(transcript=transcript), \
                 patch.object(dispatch, "vosk_wake", return_value=True), \
                 patch.object(dispatch, "transcribe_vosk", return_value=transcript):
                text = dispatch.decode_utterance(self.audio, False, settings=settings)
                self.assertEqual(text, expected)
                self.assertEqual(dispatch.command_for_turn(text, False, True, settings), expected)

    def test_configured_junk_is_preserved_in_manual_followup_and_when_unconfigured(self):
        for rendering in ("heinrich", "inc"):
            transcript = f"{rendering} che tempo fa domani a roma"
            for phrase, followup, manual in (("hey nic", True, False),
                                              ("hey nic", False, True),
                                              ("hey lari", False, False)):
                settings = load_settings({
                    "LARI_STT_BACKEND": "vosk", "LARI_WAKE_PHRASE": phrase,
                    "LARI_WAKE_OBSERVED_JUNK": "heinrich,inc" if followup or manual else "",
                })
                with self.subTest(phrase=phrase, followup=followup, manual=manual), \
                     patch.object(dispatch, "vosk_wake", return_value=True), \
                     patch.object(dispatch, "transcribe_vosk", return_value=transcript):
                    self.assertEqual(dispatch.decode_utterance(
                        self.audio, followup, settings=settings, manual=manual), transcript)


class VoskGrammarTests(unittest.TestCase):
    def test_configured_junk_is_cleanup_only_never_a_wake_trigger(self):
        cfg = load_settings({
            "LARI_WAKE_PHRASE": "hey nic", "LARI_WAKE_OBSERVED_JUNK": "heinrich,inc",
        }).wake_config
        uncalibrated = build_wake_config("hey nic")
        for field in ("grammar", "keyterms", "batch_keyterms", "realtime_keyterms",
                      "aliases", "prompt", "style_prompt", "doubt_starters"):
            self.assertEqual(getattr(cfg, field), getattr(uncalibrated, field))
        for field in ("loose_re", "command_re", "mention_re"):
            self.assertEqual(getattr(cfg, field).pattern, getattr(uncalibrated, field).pattern)
            self.assertEqual(getattr(cfg, field).flags, getattr(uncalibrated, field).flags)
        for rendering in ("heinrich", "inc"):
            transcript = f"{rendering} che tempo fa"
            with self.subTest(rendering=rendering):
                self.assertEqual(cfg.strip_junk(transcript), "che tempo fa")
                self.assertIsNone(cfg.command(transcript))
                self.assertEqual(detector.wake_command(transcript, cfg),
                                 detector.wake_command(transcript, uncalibrated))
                self.assertIsNone(cfg.loose_re.search(transcript))
                self.assertNotIn(rendering, cfg.grammar)
                self.assertNotIn(rendering, tuple(term.lower() for term in cfg.keyterms))
                self.assertIsNone(detector.resolve_command(transcript, 0.0, 1.0, cfg))
                self.assertIsNone(detector.resolve_vosk_command(
                    transcript, wake=False, followup=False, cfg=cfg))
                recognizer = Mock()
                recognizer.FinalResult.return_value = json.dumps({"text": transcript})
                fake_vosk = types.ModuleType("vosk")
                fake_vosk.KaldiRecognizer = Mock(return_value=recognizer)
                with patch.object(detector, "get_vosk", return_value=object()), \
                     patch.dict(sys.modules, {"vosk": fake_vosk}):
                    self.assertFalse(detector.vosk_wake(np.zeros(1600, dtype=np.int16), cfg))

    def test_vosk_gate_uses_the_configured_grammar_and_detection(self):
        cfg = build_wake_config("ehi lari")
        captured = {}
        fake_vosk = types.ModuleType("vosk")

        class FakeRecognizer:
            def __init__(self, model, rate, grammar=None):
                captured["grammar"] = grammar

            def AcceptWaveform(self, raw):
                return True

            def FinalResult(self):
                return json.dumps({"text": "ehi lari"})

            def Result(self):
                return json.dumps({"text": ""})

        fake_vosk.KaldiRecognizer = FakeRecognizer
        fake_vosk.Model = lambda *args, **kwargs: object()
        fake_vosk.SetLogLevel = lambda *args: None
        with patch.object(detector, "get_vosk", return_value=object()), \
             patch.dict(sys.modules, {"vosk": fake_vosk}):
            self.assertTrue(detector.vosk_wake(np.zeros(1600, dtype=np.int16), cfg))
        self.assertEqual(captured["grammar"], json.dumps(list(cfg.grammar)))


class VoskSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_routes_vosk_command_to_agent(self):
        sent = []

        async def send(message):
            sent.append(message)

        root = self.enterContext(tempfile.TemporaryDirectory())
        settings = replace(get_settings(), session_state=Path(root) / "sessions.json")
        session = session_module.Session(send, send_audio, settings=settings)
        session._record_utterance = AsyncMock(return_value=np.ones(audio_module.SAMPLE_RATE, dtype=np.int16))
        session._speak = AsyncMock()
        session._record_usage = Mock()
        with patch.object(session, "_settings", replace(session._settings, stt_backend="vosk")), \
             patch.object(session_module, "save_turn_audio", return_value=Path("fake.wav")), \
             patch.object(dispatch, "decode_utterance", return_value="please help") as decode, \
             patch.object(session_module, "stream_hermes", new_callable=AsyncMock, return_value="How can I help?") as agent:
            await session._on_wake()
        decode.assert_called_once()
        self.assertEqual(decode.call_args.args[1], False)
        agent.assert_awaited_once()
        self.assertEqual(agent.call_args.args, ("please help",))
        self.assertRegex(agent.call_args.kwargs["session_id"], r"^lari-default-[0-9a-f]{8}$")
        self.assertTrue(
            any(msg.get("type") == "transcript" and msg.get("text") == "please help" for msg in sent)
        )


if __name__ == "__main__":
    unittest.main()
