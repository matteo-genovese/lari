"""VAD endpointing: a quiet utterance must close the recorder, never hang it.

Real-voice failure (phone, raw mic, 2m): the wake-confirmed utterance counted
little "voiced" time, so the SILENCE_END close never fired and the recorder
collected 22s while the user repeated the phrase (duplicated transcript).
"""
from lari.config import get_settings
import asyncio
import queue
import time
import unittest

import numpy as np


from lari import audio as audio_module
from lari import session as session_module
SR = audio_module.SAMPLE_RATE


def _chunk(rms: float, seconds: float = 0.1) -> bytes:
    # A constant signal has rms == amplitude.
    n = int(SR * seconds)
    return np.full(n, int(min(32767.0, rms)), dtype=np.int16).tobytes()


def _session(noise_floor: float = 363.0) -> session_module.Session:
    s = session_module.Session.__new__(session_module.Session)
    s._settings = get_settings()
    s.recv_queue = queue.Queue()
    s.noise_floor = noise_floor
    s.last_silent = time.time() - 0.5    # recent silence, as at real speech onset
    s.turn = 1
    return s


class VadEndpointingTests(unittest.IsolatedAsyncioTestCase):
    async def test_quiet_utterance_with_thin_voiced_still_closes(self):
        """Wake-confirmed speech at 2m counted little above-threshold audio;
        the recorder must still close ~SILENCE_END_S after silence instead of
        hanging until the user repeats louder (22s in the real log)."""
        s = _session()
        s.recv_queue.put(_chunk(1270.0))   # two loud frames: started, voiced 0.2s
        s.recv_queue.put(_chunk(1270.0))
        for _ in range(200):               # 20s of quiet speech below threshold
            s.recv_queue.put(_chunk(900.0))
        s.recv_queue.put(None)

        pcm = await asyncio.wait_for(s._record_utterance(), timeout=15)
        collected_s = len(pcm) / SR
        self.assertLessEqual(collected_s, 6.5,
                             f"recorder hung: collected {collected_s:.2f}s")

    async def test_noise_floor_does_not_learn_during_speech(self):
        """Sub-threshold syllable edges used to raise the noise floor mid
        utterance until the threshold exceeded the speaker's own peaks."""
        s = _session(noise_floor=363.0)
        before = s.noise_floor
        for _ in range(15):                # 3s of natural speech: loud/soft edges
            s.recv_queue.put(_chunk(1270.0))
            s.recv_queue.put(_chunk(800.0))
        s.recv_queue.put(None)

        await asyncio.wait_for(s._record_utterance(), timeout=15)
        self.assertLessEqual(s.noise_floor, before * 1.1,
                             f"floor inflated during speech: {s.noise_floor:.0f}")

    async def test_solid_speech_still_collects_the_whole_utterance(self):
        s = _session()
        s.recv_queue.put(_chunk(1270.0, 2.0))   # one continuous phrase
        for _ in range(30):                     # then silence
            s.recv_queue.put(_chunk(100.0))
        s.recv_queue.put(None)

        pcm = await asyncio.wait_for(s._record_utterance(), timeout=15)
        collected_s = len(pcm) / SR
        self.assertGreaterEqual(collected_s, 1.8)
        self.assertLessEqual(collected_s, 6.5)


if __name__ == "__main__":
    unittest.main()
