"""Regression: the real httpx multipart encoding must accept our payload shape.

The mocked-transport unit tests missed a TypeError thrown by httpx's own
multipart encoder ("sequence item 1: expected a bytes-like object, tuple
found"). This exercises the actual encoder with MockTransport — no network.
"""
import os
import unittest
from unittest.mock import patch

import httpx
import numpy as np

import stt_backends

PCM = np.arange(3200, dtype=np.int16)


class MultipartEncodingTests(unittest.TestCase):
    def _run(self, backend, response_json):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured['content_type'] = request.headers.get('content-type', '')
            captured['body'] = request.read()
            return httpx.Response(200, json=response_json)

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def client_factory(*args, **kwargs):
            kwargs['transport'] = transport
            return real_client(*args, **kwargs)

        env = {
            'ELEVENLABS_API_KEY': 'k-el', 'GROQ_API_KEY': 'k-gq',
            'OPENAI_API_KEY': 'k-oa',
        }
        with patch.dict(os.environ, env), \
             patch('httpx.Client', client_factory):
            text = stt_backends.transcribe(PCM, backend)
        self.assertEqual(text, 'trascrizione')
        self.assertIn('multipart/form-data', captured['content_type'])
        if backend == 'elevenlabs':
            self.assertIn(PCM.tobytes(), captured['body'])   # Raw PCM payload
        else:
            self.assertIn(b'RIFF', captured['body'])          # WAV payload
        return captured

    def test_elevenlabs_real_multipart_encoding(self):
        self._run('elevenlabs', {'text': 'trascrizione'})

    def test_groq_real_multipart_encoding(self):
        self._run('groq', {'text': 'trascrizione'})

    def test_openai_real_multipart_encoding(self):
        self._run('openai', {'text': 'trascrizione'})


if __name__ == '__main__':
    unittest.main()
