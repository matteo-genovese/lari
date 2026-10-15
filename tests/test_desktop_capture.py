"""Capture signals reach an explicitly typed slot without a display or audio device."""
import importlib.util
import unittest
from unittest.mock import Mock, patch

from lari.config import load_settings
from lari.desktop_voice import create_voice_client


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "optional desktop dependency PySide6")
class CaptureStateTests(unittest.TestCase):
    def test_capture_signal_reaches_typed_slot_and_reports_failure(self):
        from PySide6.QtCore import QCoreApplication, QObject, Signal
        from PySide6.QtMultimedia import QtAudio

        app = QCoreApplication.instance() or QCoreApplication([])

        class Capture(QObject):
            stateChanged = Signal(QtAudio.State)

        capture = Capture()
        capture.error = Mock(return_value=QtAudio.Error.NoError)
        capture.start = Mock(return_value=Mock())
        capture.setBufferSize = Mock()
        with patch("PySide6.QtMultimedia.QAudioSource", return_value=capture), \
                patch("PySide6.QtMultimedia.QMediaDevices") as devices, \
                patch("PySide6.QtMultimedia.QMediaPlayer"), \
                patch("PySide6.QtMultimedia.QAudioOutput"):
            devices.defaultAudioInput.return_value.isNull.return_value = False
            client = create_voice_client("http://localhost/token/", load_settings({}))
            self.addCleanup(client.files.cleanup)
            self.addCleanup(client.deleteLater)
            self.assertGreaterEqual(
                client.metaObject().indexOfSlot("on_capture_state(QAudio::State)"), 0
            )
            with patch.object(client, "connect"), patch.object(client, "stop") as stop:
                client.start()
                for state in QtAudio.State:
                    with self.subTest(state=state):
                        capture.stateChanged.emit(state)
                        self.assertIsNone(client.error)
                        stop.assert_not_called()

                capture.error.return_value = QtAudio.Error.IOError
                capture.stateChanged.emit(QtAudio.State.StoppedState)
                stop.assert_called_once_with()
                self.assertEqual(client.state, "error")
                self.assertEqual(client.error, "Il microfono ha interrotto la cattura")
            client.keepalive.stop()
            app.processEvents()
