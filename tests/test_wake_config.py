"""The wake phrase is one setting: grammar, regex, cleanup and ASR terms derive from it.

Pure derivation tests: no server import, no provider, no network.
"""
import re
import unittest
from unittest.mock import patch

from lari.wake import config as wake_config

from lari.wake.config import build_wake_config, default_phrase


class ProvenWakeParityTests(unittest.TestCase):
    """The default live config must keep its battle-tested behavior."""

    def setUp(self):
        self.cfg = build_wake_config("ehi lari")

    def test_grammar_matches_the_proven_list_exactly(self):
        self.assertEqual(self.cfg.grammar,
                         ("hey lari", "ehi lari", "lari", "[unk]"))

    def test_prompt_matches_the_proven_prompt_exactly(self):
        self.assertEqual(self.cfg.prompt, "Ehi Lari. Hey Lari.")

    def test_keyterms_lead_with_the_wake_forms(self):
        self.assertEqual(self.cfg.keyterms[:3], ("Ehi Lari", "Hey Lari", "Lari"))

    def test_loose_detection_accepts_any_grammar_form(self):
        for heard in ("ehi lari", "hey lari", "lari"):
            self.assertIsNotNone(self.cfg.loose_re.search(heard), heard)
        self.assertIsNone(self.cfg.loose_re.search("ehi larice"))

    def test_cleanup_strips_short_interjections_and_alias_words(self):
        self.assertEqual(self.cfg.strip_junk("Ehi, che tempo fa?"), "che tempo fa?")
        self.assertEqual(self.cfg.strip_junk("Che tempo fa?"), "Che tempo fa?")
        aliased = build_wake_config("ehi lari", aliases=("ehi lar",))
        self.assertEqual(aliased.strip_junk("lar dimmi"), "dimmi")

    def test_observed_asr_junk_is_per_installation_calibration(self):
        with patch.dict(wake_config.OBSERVED_JUNK, {"lari": ("larix",)}):
            cfg = build_wake_config("ehi lari")
        self.assertEqual(cfg.strip_junk("larix che tempo fa?"), "che tempo fa?")
        self.assertEqual(cfg.strip_junk("Heinrich che tempo fa?"),
                         "Heinrich che tempo fa?")


class ConfigurablePhraseTests(unittest.TestCase):
    """A different phrase rebuilds every derived artifact coherently."""

    def setUp(self):
        self.cfg = build_wake_config("ehi lari")

    def test_command_is_extracted_after_the_new_wake(self):
        self.assertEqual(self.cfg.command("Ehi Lari, che tempo fa domani?"),
                         "che tempo fa domani?")

    def test_asr_final_vowel_variants_are_accepted(self):
        self.assertEqual(self.cfg.command("Hey Lare, dimmi."), "dimmi.")
        self.assertEqual(self.cfg.command("Ehi Lary, dimmi."), "dimmi.")

    def test_long_lookalikes_and_background_mentions_are_rejected(self):
        self.assertIsNone(self.cfg.command("Ehi Larice, ciao."))
        self.assertIsNone(self.cfg.command("Ho parlato con lari di lavoro."))
        self.assertIsNone(self.cfg.command("Il lario di Como."))

    def test_wake_after_a_monologue_at_a_sentence_boundary(self):
        # Talking freely and then addressing the device is normal use: only
        # the request after the wake counts, and a bare wake has no request.
        self.assertEqual(
            self.cfg.command("Bla bla bla bla bla. Ehi Lari, che ore sono?"),
            "che ore sono?")
        self.assertEqual(self.cfg.command("Bla bla bla bla bla. Ehi Lari."), "")

    def test_grammar_carries_the_new_phrase_and_its_core(self):
        self.assertIn("ehi lari", self.cfg.grammar)
        self.assertIn("hey lari", self.cfg.grammar)
        self.assertIn("lari", self.cfg.grammar)
        self.assertEqual(self.cfg.grammar[-1], "[unk]")

    def test_prompt_and_keyterms_carry_the_new_phrase(self):
        self.assertIn("Ehi Lari", self.cfg.prompt)
        self.assertLessEqual(len(self.cfg.prompt), 40)
        self.assertIn("Ehi Lari", self.cfg.keyterms)
        self.assertIn("Lari", self.cfg.keyterms)

    def test_aliases_extend_matching_grammar_and_cleanup(self):
        cfg = build_wake_config("ehi lari", aliases=("ehi lar",))
        self.assertEqual(cfg.command("Ehi lar, dimmi"), "dimmi")
        self.assertIn("ehi lar", cfg.grammar)
        self.assertEqual(cfg.strip_junk("ehi lar dimmi"), "dimmi")

    def test_fresh_phrase_has_no_built_in_junk_words(self):
        self.assertEqual(self.cfg.strip_junk("Heinrich lari"), "Heinrich lari")


class DefaultPhraseTests(unittest.TestCase):
    def test_default_follows_the_configured_language(self):
        self.assertEqual(default_phrase("it"), "ehi lari")
        self.assertEqual(default_phrase("it-IT"), "ehi lari")
        self.assertEqual(default_phrase("en"), "hey lari")
        self.assertEqual(default_phrase(""), "hey lari")


class DerivedProviderTermsTests(unittest.TestCase):
    def test_vocabulary_terms_join_wake_terms_for_batch(self):
        cfg = build_wake_config(
            "ehi lari", vocab=("centro commerciale Aurora", "Aurora"),
        )
        self.assertEqual(
            cfg.batch_keyterms,
            ("Ehi Lari", "Hey Lari", "Lari", "centro commerciale Aurora", "Aurora"),
        )

    def test_realtime_keyterms_respect_the_20_char_provider_limit(self):
        cfg = build_wake_config(
            "ehi lari", vocab=("centro commerciale Aurora", "Aurora"),
        )
        self.assertEqual(cfg.realtime_keyterms,
                         ("Ehi Lari", "Hey Lari", "Lari", "Aurora"))
        self.assertTrue(all(len(term) <= 20 for term in cfg.realtime_keyterms))

    def test_without_vocabulary_only_wake_terms_are_sent(self):
        cfg = build_wake_config("ehi lari")
        self.assertEqual(cfg.batch_keyterms, cfg.keyterms)
        self.assertGreater(len(cfg.realtime_keyterms), 1)

    def test_style_prompt_mentions_the_configured_wake(self):
        cfg = build_wake_config("ehi lari", vocab=("Stazione Centrale",))
        self.assertIn("Ehi Lari / Hey Lari", cfg.style_prompt)
        self.assertIn("Stazione Centrale", cfg.style_prompt)
        plain = build_wake_config("ehi lari")
        self.assertNotIn("Nomi propri", plain.style_prompt)


if __name__ == "__main__":
    unittest.main()
