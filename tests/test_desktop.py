"""The popup responds to conversational states, never ambient monitoring."""
import unittest
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from lari.config import load_settings
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

    def test_remote_query_allows_only_device_and_raw_one(self):
        for query in ("device=desktop", "raw=1&device=desktop", "device=desktop&raw=1",
                      "raw=1", "device=" + "a" * 32):
            for path in ("/test-token", "/test-token/"):
                with self.subTest(query=query, path=path):
                    self.assertEqual(satellite_url(SimpleNamespace(
                        desktop_url=f"https://private.example{path}?{query}")),
                        f"https://private.example/test-token/?{query}")

    def test_remote_query_rejects_invalid_names_and_unknown_parameters(self):
        for query in ("device=../etc", "device=%2E%2E%2Fetc", "device=Desktop",
                      "device=", "device=-desktop", "device=" + "a" * 33,
                      "other=1", "raw=1&other=1", "device=desktop&other=1",
                      "raw=0", "raw=", "raw", "device=desktop&device=phone",
                      "raw=1&raw=1"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                satellite_url(SimpleNamespace(
                    desktop_url=f"https://private.example/test-token/?{query}"))

    def test_remote_without_query_keeps_normalized_trailing_slash(self):
        for path in ("/test-token", "/test-token/", "/test-token//"):
            with self.subTest(path=path):
                self.assertEqual(satellite_url(SimpleNamespace(
                    desktop_url=f"https://private.example{path}")),
                    "https://private.example/test-token/")

    def test_configured_device_is_added_to_local_and_remote_page_urls(self):
        for remote, expected in (
            ("", "http://127.0.0.1:8643/test-token/?device=desktop"),
            ("https://private.example/test-token", "https://private.example/test-token/?device=desktop"),
            ("https://private.example/test-token/?raw=1&device=phone",
             "https://private.example/test-token/?raw=1&device=desktop"),
        ):
            with self.subTest(remote=remote):
                settings = load_settings({"LARI_TOKEN": "test-token", "LARI_DEVICE": "desktop",
                                          "LARI_DESKTOP_URL": remote})
                self.assertEqual(satellite_url(settings), expected)

    def test_invalid_configured_device_is_ignored_without_startup_exception(self):
        for remote in ("", "https://private.example/test-token/?device=phone"):
            with self.subTest(remote=remote):
                env = {"LARI_TOKEN": "test-token", "LARI_DESKTOP_URL": remote}
                with self.assertLogs("lari.config", level="WARNING") as logs:
                    settings = load_settings({**env, "LARI_DEVICE": "../etc"})
                self.assertEqual(settings.device, "")
                self.assertEqual(satellite_url(settings), satellite_url(load_settings(env)))
                self.assertIn("LARI_DEVICE", logs.output[0])
                self.assertNotIn("../etc", logs.output[0])

    @unittest.skipUnless(shutil.which("node"), "Node is required for page WebSocket checks")
    def test_configured_device_reaches_real_page_websocket_on_local_and_remote_bridges(self):
        html = (Path(__file__).resolve().parent.parent / "static/index.html").read_text()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        script = script.rsplit("})();", 1)[0] + "globalThis.wsURL = WS_URL;})();"
        urls = [satellite_url(load_settings({
            "LARI_TOKEN": "test-token", "LARI_DEVICE": "desktop", "LARI_DESKTOP_URL": remote,
        })) for remote in ("", "https://private.example:8443/test-token/?raw=1&device=phone")]
        harness = r'''
const assert = require("assert/strict"), vm = require("vm");
for (const pageURL of __PAGE_URLS__) {
  const parsed = new URL(pageURL);
  const widget = () => ({textContent:"", style:{},
    classList:{add(){}, remove(){}, toggle(){}}, addEventListener(){}});
  const context = {
    location:parsed, URLSearchParams,
    document:{getElementById:widget, addEventListener(){}},
    window:{}, navigator:{}, setInterval(){}, clearInterval(){},
    setTimeout(){}, clearTimeout(){},
  };
  vm.createContext(context);
  vm.runInContext(SOURCE, context);
  assert.equal(context.wsURL, (parsed.protocol === "https:" ? "wss" : "ws") +
    "://" + parsed.host + "/test-token/ws?device=desktop");
}
'''
        result = subprocess.run([shutil.which("node"), "-e",
                                 harness.replace("__PAGE_URLS__", json.dumps(urls)).replace("SOURCE", json.dumps(script))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reject_insecure_and_ambiguous_remote_urls_without_echoing_secrets(self):
        for url in ("http://remote.example/private-token/", "https://u:p@remote.example/private-token/",
                    "https://remote.example/", "https://remote.example/private-token/#fragment",
                    "https://remote.example:bad/private-token/", "file:///private-token/"):
            with self.subTest(url=url), self.assertRaises(ValueError) as error:
                satellite_url(SimpleNamespace(desktop_url=url))
            self.assertNotIn("private-token", str(error.exception))
