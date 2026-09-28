"""The second local gate may veto only on a confident mismatch.

A false negative costs more than a few seconds of credits: it destroys trust
in the device. When in doubt, the candidate always passes to Realtime.
"""
import unittest
from unittest.mock import patch

import numpy as np

from lari import server
from lari.wake_config import build_wake_config

LARI = build_wake_config("hey lari")


class ConfirmedAbsenceTests(unittest.TestCase):
    def test_clear_non_wake_speech_from_both_recognizers_is_vetoed(self):
        self.assertTrue(LARI.confirmed_absent(
            "che tempo fa domani a Roma", "vorrei andare a fare shopping"))

    def test_one_clean_recognizer_is_not_enough_to_veto(self):
        self.assertFalse(LARI.confirmed_absent("che tempo fa domani a Roma", "uh uh"))

    def test_empty_or_unintelligible_transcripts_never_veto(self):
        self.assertFalse(LARI.confirmed_absent("", "che tempo fa domani a Roma"))
        self.assertFalse(LARI.confirmed_absent("sì", "che tempo fa domani a Roma"))

    def test_observed_wake_renderings_never_veto(self):
        for text in ("E Lare, che tempo fa?", "Lary, che tempo fa?",
                     "Ehi Lari, che tempo fa?", "lari che tempo fa"):
            with self.subTest(text=text):
                self.assertFalse(LARI.confirmed_absent(text, "che tempo fa domani a Roma"))

    def test_bare_interjection_start_keeps_the_benefit_of_the_doubt(self):
        self.assertFalse(LARI.confirmed_absent("Ehi, che tempo fa domani a Roma",
                                              "ehi che tempo fa domani a roma"))

    def test_mid_sentence_mentions_block_the_veto(self):
        self.assertFalse(LARI.confirmed_absent("ho parlato con Lari di lavoro",
                                              "ho parlato con lari di lavoro"))

    def test_lookalike_words_are_not_mentions(self):
        self.assertTrue(LARI.confirmed_absent("un unico biglietto per milano",
                                            "unico biglietto per milano"))


class ConfigurablePhraseVetoTests(unittest.TestCase):
    def test_new_phrase_renderings_keep_the_benefit_of_the_doubt(self):
        cfg = build_wake_config("ehi lari", aliases=("ehi lar",))
        for text in ("Ehi Lari, che tempo fa?", "Ehi Lare, che tempo fa?",
                     "ehi lar che tempo fa", "ho detto lari ieri"):
            with self.subTest(text=text):
                self.assertFalse(cfg.confirmed_absent(text, "che tempo fa"))

    def test_new_phrase_still_vetoes_clear_non_wake_speech(self):
        cfg = build_wake_config("ehi lari")
        self.assertTrue(cfg.confirmed_absent("vorrei andare a fare shopping",
                                             "che tempo fa domani a Roma"))


class SecondGateWiringTests(unittest.TestCase):
    def test_confirm_candidate_reads_only_the_candidate_prefix(self):
        with patch.object(server, "transcribe_vosk", return_value="che tempo fa") as free, \
             patch.object(server, "transcribe",
                          return_value="vorrei andare a fare shopping") as heard:
            passed = server.confirm_candidate(np.full(16000 * 6, 3000, dtype=np.int16))
        self.assertFalse(passed)
        self.assertLessEqual(free.call_args.args[0].shape[0],
                             int(2.5 * server.SAMPLE_RATE))
        self.assertLessEqual(heard.call_args.args[0].shape[0],
                             int(2.5 * server.SAMPLE_RATE))

    def test_doubt_passes_without_running_the_slow_model(self):
        with patch.object(server, "transcribe_vosk",
                          return_value="Ehi Lare, che tempo fa"), \
             patch.object(server, "transcribe",
                          side_effect=AssertionError("must not run")):
            self.assertTrue(server.confirm_candidate(np.zeros(4000, dtype=np.int16)))

    def test_vetoed_candidate_never_schedules_a_turn(self):
        async def sender(_):
            pass
        session = server.Session(None, sender)
        with patch.object(server, "WAKE_CONFIRM", True), \
             patch.object(server, "vosk_wake", return_value=True), \
             patch.object(server, "confirm_candidate", return_value=False), \
             patch.object(server.asyncio, "run_coroutine_threadsafe") as schedule:
            started = session._launch_local_candidate(
                np.zeros(3200, dtype=np.int16).tobytes())
        self.assertFalse(started)
        schedule.assert_not_called()

    def test_disabled_second_gate_keeps_the_single_gate_path(self):
        async def sender(_):
            pass
        session = server.Session(None, sender)
        with patch.object(server, "WAKE_CONFIRM", False), \
             patch.object(server, "vosk_wake", return_value=True), \
             patch.object(server, "confirm_candidate",
                          side_effect=AssertionError("must not run")), \
             patch.object(server.asyncio, "run_coroutine_threadsafe") as schedule:
            started = session._launch_local_candidate(
                np.zeros(3200, dtype=np.int16).tobytes())
        self.assertTrue(started)
        schedule.assert_called_once()


if __name__ == "__main__":
    unittest.main()
