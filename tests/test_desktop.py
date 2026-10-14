"""The popup responds to conversational states, never ambient monitoring."""
import unittest
from types import SimpleNamespace
from lari.desktop import satellite_url, should_open_popup


class DesktopTests(unittest.TestCase):
    def test_monitoring_does_not_open_popup(self):
        for state in ("idle", "listening", "error"):
            self.assertFalse(should_open_popup("idle", state, True))
        self.assertFalse(should_open_popup("listening", "waking", False))

    def test_wake_and_followup_open_once_per_turn(self):
        self.assertTrue(should_open_popup("listening", "waking", True))
        self.assertTrue(should_open_popup("listening", "recording", True))
        self.assertFalse(should_open_popup("waking", "recording", True))
        self.assertFalse(should_open_popup("thinking", "speaking", True))

    def test_only_localhost_and_escaped_token(self):
        self.assertEqual(satellite_url(SimpleNamespace(port=8643, token="a/b?c")),
                         "http://127.0.0.1:8643/a%2Fb%3Fc/")
        with self.assertRaises(ValueError):
            satellite_url(SimpleNamespace(port=8643, token=""))

    def test_remote_bridge_needs_no_local_token(self):
        self.assertEqual(satellite_url(SimpleNamespace(desktop_url="https://private.example:8443/test-token", token="")),
                         "https://private.example:8443/test-token/")

    def test_reject_insecure_and_ambiguous_remote_urls_without_echoing_secrets(self):
        for url in ("http://remote.example/private-token/", "https://u:p@remote.example/private-token/",
                    "https://remote.example/", "https://remote.example/private-token/#fragment",
                    "https://remote.example:bad/private-token/", "file:///private-token/"):
            with self.subTest(url=url), self.assertRaises(ValueError) as error:
                satellite_url(SimpleNamespace(desktop_url=url))
            self.assertNotIn("private-token", str(error.exception))
