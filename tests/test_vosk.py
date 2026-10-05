"""Unit tests for local Vosk wake and transcription routing."""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from lari import server
from lari.wake_config import build_wake_config


class VoskRoutingTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.zeros(server.SAMPLE_RATE, dtype=np.int16)

    def test_unaddressed_speech_is_rejected_before_transcription(self):
        with patch.object(server, "vosk_wake", return_value=False) as wake, \
             patch.object(server, "transcribe_vosk") as transcribe:
            result = server.decode_utterance(self.audio, followup=False, backend="vosk")
        self.assertIsNone(result)
        wake.assert_called_once()
        transcribe.assert_not_called()

    def test_configured_wake_phrase_is_removed_from_the_command(self):
        transcript = f"{server.WAKE_PHRASE}, please help"
        with patch.object(server, "vosk_wake", return_value=True), \
             patch.object(server, "transcribe_vosk", return_value=transcript):
            result = server.decode_utterance(self.audio, followup=False, backend="vosk")
        self.assertEqual(result, "please help")

    def test_followup_transcription_does_not_require_another_wake(self):
        with patch.object(server, "vosk_wake") as wake, \
             patch.object(server, "transcribe_vosk", return_value="follow-up question"):
            result = server.decode_utterance(self.audio, followup=True, backend="vosk")
        self.assertEqual(result, "follow-up question")
        wake.assert_not_called()


class VoskGrammarTests(unittest.TestCase):
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
        with patch.object(server, "get_vosk", return_value=object()), \
             patch.dict(sys.modules, {"vosk": fake_vosk}):
            self.assertTrue(server.vosk_wake(np.zeros(1600, dtype=np.int16), cfg))
        self.assertEqual(captured["grammar"], json.dumps(list(cfg.grammar)))


class VoskSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_routes_vosk_command_to_agent(self):
        sent = []

        async def send(message):
            sent.append(message)

        session = server.Session(None, send)
        session._record_utterance = AsyncMock(return_value=np.ones(server.SAMPLE_RATE, dtype=np.int16))
        session._speak = AsyncMock()
        session._record_usage = Mock()
        with patch.object(server, "STT_BACKEND", "vosk"), \
             patch.object(server, "save_turn_audio", return_value=Path("fake.wav")), \
             patch.object(server, "decode_utterance", return_value="please help") as decode, \
             patch.object(server, "stream_hermes", new_callable=AsyncMock, return_value="How can I help?") as agent:
            await session._on_wake()
        decode.assert_called_once()
        self.assertEqual(decode.call_args.args[1], False)
        agent.assert_awaited_once()
        self.assertEqual(agent.call_args.args, ("please help",))
        self.assertIsNone(agent.call_args.kwargs["session_id"])
        self.assertTrue(
            any(msg.get("type") == "transcript" and msg.get("text") == "please help" for msg in sent)
        )


if __name__ == "__main__":
    unittest.main()
