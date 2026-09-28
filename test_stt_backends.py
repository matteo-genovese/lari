"""Tests for the A/B/C cloud STT backends (ElevenLabs Scribe, Groq Whisper, OpenAI Whisper).

No network in tests: the multipart transport is mocked and every request
construction and response parse is asserted.
"""
import os
import unittest
from unittest.mock import patch

import numpy as np

import stt_backends

PCM = np.arange(3200, dtype=np.int16)  # 0.2 s of 16 kHz mono


def fake_post(payload):
    def _post(**kwargs):
        fake_post.captured = kwargs
        return payload
    return _post


class ElevenLabsTests(unittest.TestCase):
    def test_posts_scribe_v2_with_raw_pcm_keyterms_and_parses_text(self):
        fake_post.captured = None
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'k-el'}), \
             patch.object(stt_backends, 'KEYTERMS', ['Sentinel Term']), \
             patch.object(stt_backends, '_post_multipart', fake_post({'text': 'Ehi Nic, che tempo fa a Roma?'})):
            text = stt_backends.transcribe(PCM, 'elevenlabs')
        self.assertEqual(text, 'Ehi Nic, che tempo fa a Roma?')
        call = fake_post.captured
        self.assertEqual(call['url'], 'https://api.elevenlabs.io/v1/speech-to-text')
        self.assertEqual(call['headers'].get('xi-api-key'), 'k-el')
        data = dict(call['data_tuples'])
        self.assertEqual(data['model_id'], 'scribe_v2')
        self.assertEqual(data['file_format'], 'pcm_s16le_16')
        self.assertEqual(data['language_code'], 'it')
        keyterms = [v for k, v in call['data_tuples'] if k == 'keyterms']
        self.assertEqual(keyterms, ['Sentinel Term'])
        self.assertEqual(call['file_bytes'], PCM.tobytes())  # PCM nudo, non WAV


class GroqTests(unittest.TestCase):
    def test_posts_whisper_turbo_with_prompt_and_parses_text(self):
        fake_post.captured = None
        with patch.dict(os.environ, {'GROQ_API_KEY': 'k-gq'}), \
             patch.object(stt_backends, 'STYLE_PROMPT', 'sentinel style'), \
             patch.object(stt_backends, '_post_multipart', fake_post({'text': 'Che tempo fa domani a Roma?'})):
            text = stt_backends.transcribe(PCM, 'groq')
        self.assertEqual(text, 'Che tempo fa domani a Roma?')
        call = fake_post.captured
        self.assertEqual(call['url'], 'https://api.groq.com/openai/v1/audio/transcriptions')
        self.assertEqual(call['headers'].get('Authorization'), 'Bearer k-gq')
        data = dict(call['data_tuples'])
        self.assertEqual(data['model'], 'whisper-large-v3')
        self.assertEqual(data['language'], 'it')
        self.assertEqual(data['prompt'], 'sentinel style')
        self.assertEqual(call['file_bytes'][:4], b'RIFF')  # WAV per Groq


class OpenAITests(unittest.TestCase):
    def test_posts_whisper1_and_parses_text(self):
        fake_post.captured = None
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'k-oa'}), \
             patch.object(stt_backends, '_post_multipart', fake_post({'text': 'Vorrei andare a fare shopping.'})):
            text = stt_backends.transcribe(PCM, 'openai')
        self.assertEqual(text, 'Vorrei andare a fare shopping.')
        call = fake_post.captured
        self.assertEqual(call['url'], 'https://api.openai.com/v1/audio/transcriptions')
        self.assertEqual(call['headers'].get('Authorization'), 'Bearer k-oa')
        data = dict(call['data_tuples'])
        self.assertEqual(data['model'], 'whisper-1')
        self.assertEqual(data['language'], 'it')


class FailureTests(unittest.TestCase):
    def test_missing_key_fails_before_any_http(self):
        fake_post.captured = None
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(stt_backends, '_post_multipart', fake_post({'text': 'non deve arrivare qui'})):
            with self.assertRaisesRegex(RuntimeError, 'ELEVENLABS_API_KEY'):
                stt_backends.transcribe(PCM, 'elevenlabs')
        self.assertIsNone(fake_post.captured)

    def test_http_error_names_the_backend(self):
        def boom(**kwargs):
            raise ConnectionError('boom')
        with patch.dict(os.environ, {'GROQ_API_KEY': 'k'}), \
             patch.object(stt_backends, '_post_multipart', boom):
            with self.assertRaisesRegex(RuntimeError, 'groq'):
                stt_backends.transcribe(PCM, 'groq')

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            stt_backends.transcribe(PCM, 'craiyon')


class WakeTermDerivationTests(unittest.TestCase):
    def test_provider_terms_derive_from_the_configured_wake(self):
        import json
        import subprocess
        import sys
        env = {
            "PATH": os.environ.get("PATH", ""),
            "BUDDY_WAKE_PHRASE": "ehi lari",
            "BUDDY_STT_KEYTERMS": "Aura",
        }
        code = (
            "import json, stt_backends; print(json.dumps({"
            "'rt': list(stt_backends.REALTIME_KEYTERMS), "
            "'kt': list(stt_backends.KEYTERMS), "
            "'p': stt_backends.STYLE_PROMPT}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, text=True,
            capture_output=True, check=True,
        )
        data = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(data["rt"], ["Ehi Lari", "Hey Lari", "Lari", "Aura"])
        self.assertEqual(data["kt"], data["rt"])
        self.assertIn("Ehi Lari / Hey Lari", data["p"])
        self.assertIn("Aura", data["p"])


class ServerRoutingTests(unittest.TestCase):
    def test_cloud_decode_keeps_wake_gate_and_followup(self):
        import server
        transcript = f'{server.WAKE_CONFIG.display}, che tempo fa a Roma?'
        with patch.object(server.stt_backends, 'transcribe',
                          return_value=transcript) as mocked:
            cmd = server.decode_utterance(PCM, followup=False, backend='groq')
            self.assertEqual(cmd, 'che tempo fa a Roma?')
            follow = server.decode_utterance(PCM, followup=True, backend='groq')
            # Il residuo di wake viene tolto anche nel follow-up: stessa
            # semantica del percorso whisper. La differenza del follow-up si
            # vede sul testo SENZA wake (test_cloud_decode_rejects_speech_without_wake).
            self.assertEqual(follow, 'che tempo fa a Roma?')
        self.assertEqual(mocked.call_args.args[1], 'groq')

    def test_cloud_decode_rejects_speech_without_wake(self):
        import server
        with patch.object(server.stt_backends, 'transcribe',
                          return_value='Ho parlato con Nic di lavoro.'):
            self.assertIsNone(server.decode_utterance(PCM, followup=False, backend='openai'))
            self.assertEqual(server.decode_utterance(PCM, followup=True, backend='openai'),
                             'Ho parlato con Nic di lavoro.')


class RuntimeDispatchTests(unittest.TestCase):
    def test_configured_cloud_backend_is_the_one_called(self):
        """Il routing runtime deve usare il backend cloud di .env, non Whisper
        locale: il wiring in _on_wake era rotto anche con decode_utterance giusto."""
        import server
        with patch.object(server, 'STT_BACKEND', 'groq'), \
             patch.object(server.stt_backends, 'transcribe',
                          return_value='Ehi Nic, che tempo fa?') as cloud, \
             patch.object(server, 'transcribe',
                          side_effect=AssertionError('whisper locale non deve girare')) as local:
            text = server.stt_transcribe(PCM)
        self.assertEqual(text, 'Ehi Nic, che tempo fa?')
        self.assertEqual(cloud.call_args.args[1], 'groq')
        local.assert_not_called()

    def test_whisper_local_is_the_fallback(self):
        import server
        with patch.object(server, 'STT_BACKEND', 'whisper'), \
             patch.object(server, 'transcribe', return_value='Ehi Nic.') as local:
            self.assertEqual(server.stt_transcribe(PCM), 'Ehi Nic.')
        local.assert_called_once()


if __name__ == '__main__':
    unittest.main()
