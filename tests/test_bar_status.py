"""Bar status conversion and CLI failures, without network access."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError


SCRIPT = Path(__file__).resolve().parents[1] / "integrations/bar/lari-bar-status"
LOADER = importlib.machinery.SourceFileLoader("lari_bar_status", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
bar = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(bar)


def payload(*states):
    return {
        "version": 1, "phrase": "ciao Lari", "stt_backend": "vosk",
        "devices": [{"device": name, "connected": True, "state": state, "turn": 1}
                    for name, state in zip(("phone", "desktop"), states)],
    }


class BarStatusTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def run_config_cli(self, args=(), environment=None, expected_url=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, environment or {}), patch.object(
                bar, "fetch_status", return_value=payload("listening")) as fetch:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(bar.main(list(args)), 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        if expected_url is None:
            fetch.assert_not_called()
        else:
            fetch.assert_called_once_with(expected_url)
        return json.loads(stdout.getvalue())

    def test_configuration_precedence(self):
        url = "https://bridge.example/test-secret/status"
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("LARI_PORT=8643\nLARI_TOKEN=test-secret\n", encoding="utf-8")
            local_url = "http://127.0.0.1:8643/test-secret/status"
            cases = [
                (("--url", url, "--env", "missing.env"),
                 {"LARI_ENV": "missing.env", "LARI_STATUS_URL": "invalid"}, url),
                (("--env", str(env)),
                 {"LARI_ENV": "missing.env", "LARI_STATUS_URL": "invalid"}, local_url),
                ((), {"LARI_ENV": str(env), "LARI_STATUS_URL": "invalid"}, local_url),
                ((), {"LARI_STATUS_URL": url}, url),
            ]
            for args, environment, expected_url in cases:
                with self.subTest(args=args, environment=environment):
                    result = self.run_config_cli(args, environment, expected_url)
                    self.assertEqual(result["class"], "listening")

    def test_absent_configuration_is_offline(self):
        result = self.run_config_cli()
        self.assertEqual(result["class"], "offline")
        self.assertEqual(result["tooltip"],
                         "configurazione assente: usa --env, --url o LARI_ENV")

    def test_invalid_urls_are_offline(self):
        invalid_urls = ["", "not-a-url", "ftp://bridge.example/test-secret/status",
                        "https://bridge.example/status",
                        "https://user:password@bridge.example/test-secret/status",
                        "https://bridge.example/test-secret/status?query=1",
                        "https://bridge.example/test-secret/status#fragment",
                        "https://bridge.example:invalid/test-secret/status"]
        for url in invalid_urls:
            with self.subTest(url=url, source="--url"):
                result = self.run_config_cli(
                    ("--url", url, "--env", "missing.env"),
                    {"LARI_STATUS_URL": "https://bridge.example/test-secret/status"})
                self.assertEqual(result["class"], "offline")
            if url:
                with self.subTest(url=url, source="LARI_STATUS_URL"):
                    result = self.run_config_cli(environment={"LARI_STATUS_URL": url})
                    self.assertEqual(result["class"], "offline")

    def test_invalid_env_does_not_fall_back_to_url(self):
        for args, environment in [
                (("--env", "missing.env"),
                 {"LARI_STATUS_URL": "https://bridge.example/test-secret/status"}),
                ((), {"LARI_ENV": "missing.env",
                      "LARI_STATUS_URL": "https://bridge.example/test-secret/status"})]:
            with self.subTest(args=args, environment=environment):
                self.assertEqual(self.run_config_cli(args, environment)["class"], "offline")

    def run_cli(self, args=(), response=None, error=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(bar, "fetch_status", return_value=response,
                          side_effect=error) as fetch:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = bar.main(["--url", "http://127.0.0.1:8643/test-secret/status", *args])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        fetch.assert_called_once_with("http://127.0.0.1:8643/test-secret/status")
        return stdout.getvalue()

    def test_no_devices(self):
        result = bar.status_to_waybar(payload())
        self.assertEqual(result["class"], "idle")
        self.assertIn("Nessun satellite connesso", result["tooltip"])
        disconnected = payload("speaking")
        disconnected["devices"][0]["connected"] = False
        self.assertEqual(bar.status_to_waybar(disconnected), result)

    def test_listening(self):
        result = bar.status_to_waybar(payload("listening"))
        self.assertEqual(result["class"], "listening")
        self.assertEqual(result["text"], bar.DEFAULT_GLYPH)
        self.assertIn("Frase di wake: ciao Lari", result["tooltip"])
        self.assertIn("phone: listening", result["tooltip"])

    def test_speaking_and_other_busy_states(self):
        for state in bar.BUSY_STATES:
            with self.subTest(state=state):
                result = bar.status_to_waybar(payload(state))
                self.assertEqual(result["class"], "busy")

    def test_multiple_devices(self):
        result = bar.status_to_waybar(payload("listening", "speaking"), "mic")
        self.assertEqual(result["text"], "mic 2")
        self.assertEqual(result["class"], "busy")
        self.assertIn("phone: listening\ndesktop: speaking", result["tooltip"])

    def test_unknown_state(self):
        result = bar.status_to_waybar(payload("future-state"))
        self.assertEqual(result["class"], "error")
        self.assertIn("phone: stato sconosciuto", result["tooltip"])

    def test_idle_error_and_precedence(self):
        for states, expected in [(("idle",), "idle"), (("error",), "error"),
                                 (("listening", "error"), "error"),
                                 (("error", "thinking"), "busy")]:
            with self.subTest(states=states):
                self.assertEqual(bar.status_to_waybar(payload(*states))["class"], expected)

    def test_malformed_json(self):
        error = json.JSONDecodeError("Invalid JSON", "{", 1)
        result = json.loads(self.run_cli(error=error))
        self.assertEqual(result["class"], "offline")

    def test_malformed_payload(self):
        for value in [None, [], {}, {**payload(), "version": 2},
                      {**payload(), "devices": [None]}]:
            with self.subTest(value=value):
                self.assertEqual(json.loads(self.run_cli(response=value))["class"], "offline")

    def test_network_error_and_timeout(self):
        for error in (URLError("http://127.0.0.1:8643/test-secret/status"),
                      TimeoutError("test-secret")):
            with self.subTest(error=type(error).__name__):
                output = self.run_cli(error=error)
                self.assertEqual(json.loads(output)["class"], "offline")
                self.assertNotIn("test-secret", output)

    def test_token_never_appears_in_output(self):
        value = payload("listening")
        value["phrase"] = "test-secret"
        value["stt_backend"] = "test-secret"
        value["devices"][0]["device"] = "test-secret"
        for args in (("--glyph", "test-secret"), ("--plain", "--glyph", "test-secret")):
            self.assertNotIn("test-secret", self.run_cli(args, response=value))

    def test_plain_and_glyph(self):
        output = self.run_cli(("--plain", "--glyph", "mic"), response=payload("listening"))
        self.assertTrue(output.startswith("mic [listening] "))
        self.assertIn("phone: listening", output)
        self.assertEqual(json.loads(self.run_cli(("--glyph", "mic"),
                                               error=TimeoutError()))["text"], "mic")

    def test_tooltip_markup_is_escaped(self):
        value = payload("listening")
        value["devices"][0]["device"] = "<b>phone</b>"
        self.assertIn("&lt;b&gt;phone&lt;/b&gt;", json.loads(self.run_cli(response=value))["tooltip"])

    def test_env_and_escaped_token(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("# ignored\nexport LARI_PORT=8643 # port\nLARI_TOKEN='a/b?c'\n",
                           encoding="utf-8")
            self.assertEqual(bar.resolve_url(env, None),
                             ("http://127.0.0.1:8643/a%2Fb%3Fc/status", "a/b?c"))
            value = payload("listening")
            value["phrase"] = "a/b?c a%2Fb%3Fc"
            with patch.object(bar, "fetch_status", return_value=value):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(bar.main(["--env", str(env)]), 0)
            self.assertNotIn("a/b?c", output.getvalue())
            self.assertNotIn("a%2Fb%3Fc", output.getvalue())

    def test_http_boundary_timeout_and_json(self):
        with patch.object(bar, "urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = b'{"version": 1}'
            self.assertEqual(bar.fetch_status("http://127.0.0.1/status"), {"version": 1})
            opener.assert_called_once_with("http://127.0.0.1/status", timeout=2)

    def test_missing_env_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output), patch.object(bar, "fetch_status") as fetch:
                self.assertEqual(bar.main(["--env", str(Path(directory) / ".env")]), 0)
            fetch.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())["class"], "offline")


if __name__ == "__main__":
    unittest.main()
