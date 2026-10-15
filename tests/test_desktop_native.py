import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

from lari.config import load_settings
from lari.desktop import satellite_url
from lari.desktop_actions import application_catalog, command_catalog, execute_action
from lari.desktop_native import mascot_svg, move_dispatch
from lari.desktop_voice import PlaybackFrames, websocket_url


class NativePlaybackTests(unittest.TestCase):
    def test_segment_order_and_late_packets(self):
        frames = PlaybackFrames()
        self.assertFalse(frames.binary(b"late"))
        self.assertTrue(frames.metadata({"type": "audio_start", "turn": 3}))
        self.assertFalse(frames.metadata({"type": "audio_chunk", "turn": 2, "seq": 0}))
        self.assertFalse(frames.metadata({"type": "audio_chunk", "turn": 3, "seq": 1}))
        self.assertTrue(frames.metadata({"type": "audio_chunk", "turn": 3, "seq": 0}))
        self.assertFalse(frames.metadata({"type": "audio_end", "turn": 3}))
        self.assertTrue(frames.binary(b"MP3"))
        self.assertEqual(frames.queue.popleft(), b"MP3")
        self.assertTrue(frames.metadata({"type": "audio_end", "turn": 3}))
        self.assertEqual(frames.completed(), 3)
        self.assertFalse(frames.binary(b"late"))
        self.assertFalse(frames.metadata({"type": "audio_start", "turn": 3}))
        self.assertTrue(frames.metadata({"type": "audio_start", "turn": 4}))

    def test_single_packet_response_and_reset(self):
        frames = PlaybackFrames()
        self.assertTrue(frames.metadata({"type": "audio", "turn": 1}))
        self.assertTrue(frames.ended)
        self.assertTrue(frames.binary(b"MP3"))
        frames.reset()
        self.assertFalse(frames.queue)
        self.assertFalse(frames.binary(b"old"))

    def test_websocket_uses_private_secure_bridge(self):
        self.assertEqual(websocket_url("https://private.example/token/"), "wss://private.example/token/ws")
        self.assertEqual(websocket_url("http://127.0.0.1:8643/token/"), "ws://127.0.0.1:8643/token/ws")

    def test_device_identity_reaches_native_websocket(self):
        for remote, device, expected in (
            ("", "desktop", "ws://127.0.0.1:8643/token/ws?device=desktop"),
            ("https://private.example/token/?raw=1&device=phone", "desktop",
             "wss://private.example/token/ws?device=desktop"),
            ("https://private.example/token/?device=phone&raw=1", "",
             "wss://private.example/token/ws?device=phone"),
        ):
            with self.subTest(remote=remote, device=device):
                settings = load_settings({"LARI_TOKEN": "token", "LARI_DEVICE": device,
                                          "LARI_DESKTOP_URL": remote})
                self.assertEqual(websocket_url(satellite_url(settings)), expected)

    def test_invalid_device_falls_back_before_native_connection(self):
        with self.assertLogs("lari.config", level="WARNING"):
            settings = load_settings({"LARI_TOKEN": "token", "LARI_DEVICE": "../etc"})
        self.assertEqual(websocket_url(satellite_url(settings)),
                         "ws://127.0.0.1:8643/token/ws")


class LocalActionTests(unittest.TestCase):
    def test_app_launch_resolves_installed_id_not_a_shell_string(self):
        apps = {"editor.desktop": {"id": "editor.desktop", "name": "Editor", "path": "/tmp/Editor App.desktop"}}
        launch = Mock(return_value=SimpleNamespace(pid=42))
        result = execute_action({"operation": "open-app", "id": "editor.desktop"}, apps, {}, launch)
        self.assertTrue(result["ok"])
        launch.assert_called_once_with(["gio", "launch", "/tmp/Editor App.desktop"])
        self.assertFalse(execute_action({"operation": "open-app", "id": "editor; rm -rf /"}, apps, {}, launch)["ok"])

    def test_command_arguments_are_passed_literally(self):
        launch = Mock(return_value=SimpleNamespace(pid=9))
        result = execute_action({"operation": "exec", "argv": ["printf", "$(touch file); $HOME"]}, {}, {}, launch)
        self.assertTrue(result["ok"])
        launch.assert_called_once_with(["printf", "$(touch file); $HOME"])
        self.assertFalse(execute_action({"operation": "exec", "argv": "echo x"}, {}, {}, launch)["ok"])
        self.assertFalse(execute_action({"operation": "exec", "argv": []}, {}, {}, launch)["ok"])

    def test_user_app_override_can_hide_a_system_app(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user, system = root / "user", root / "system"
            user.mkdir(); system.mkdir()
            (system / "a.desktop").write_text("[Desktop Entry]\nType=Application\nName=Test\nExec=test\n")
            (user / "a.desktop").write_text("[Desktop Entry]\nType=Application\nHidden=true\n")
            self.assertFalse(application_catalog([user, system]))

    def test_named_commands_require_argument_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.json"
            path.write_text(json.dumps({"hello": ["printf", "hello"]}))
            self.assertEqual(command_catalog(path)["hello"], ["printf", "hello"])
            path.write_text(json.dumps({"hello": "printf hello"}))
            with self.assertRaises(ValueError):
                command_catalog(path)


class NativeArtTests(unittest.TestCase):
    def test_move_uses_installed_lua_dispatcher_and_an_exact_window(self):
        self.assertEqual(move_dispatch(10, 20), ["dispatch", 'hl.dsp.window.move({window="title:^Lari Mascot$", x=10, y=20, relative=false})'])

    def test_all_native_sprites_keep_vector_art_without_opaque_background(self):
        for state in ("idle", "listening", "thinking", "speaking", "error"):
            root = ET.fromstring(mascot_svg(state))
            self.assertTrue(any(c.tag.endswith("g") for c in root))
            self.assertFalse(any(c.tag.endswith("rect") and c.get("fill") == "#000000" for c in root))
