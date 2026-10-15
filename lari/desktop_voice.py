"""Native PCM capture and turn-scoped MP3 playback over the satellite protocol."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from . import protocol


def websocket_url(page_url):
    parsed = urlsplit(page_url)
    return urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc,
                       parsed.path.rstrip("/") + "/ws",
                       urlencode([(name, value) for name, value in parse_qsl(parsed.query)
                                  if name == "device"]), ""))


class PlaybackFrames:
    """Validate metadata/binary pairing and reject stale turns before playback."""
    def __init__(self):
        self.last_turn = None
        self.reset()

    def reset(self):
        self.turn = None
        self.mode = None
        self.awaiting = False
        self.next_seq = 0
        self.ended = False
        self.queue = deque()

    def metadata(self, message):
        kind, turn = message.get("type"), message.get("turn")
        if kind in {"audio_start", "audio"}:
            if self.mode or turn is None or (self.last_turn is not None and turn <= self.last_turn):
                return False
            self.mode = "segmented" if kind == "audio_start" else "legacy"
            self.turn = turn
            self.awaiting = kind == "audio"
            self.ended = kind == "audio"
            return True
        if not self.mode or turn != self.turn or self.awaiting:
            return False
        if kind == "audio_chunk" and message.get("seq") == self.next_seq and not self.ended:
            self.awaiting = True
            return True
        if kind == "audio_end" and not self.ended:
            self.ended = True
            return True
        return False

    def binary(self, packet):
        if not self.mode or not self.awaiting or not packet:
            return False
        self.awaiting = False
        self.next_seq += 1
        self.queue.append(bytes(packet))
        return True

    def completed(self):
        turn = self.turn
        self.last_turn = turn
        self.reset()
        return turn


def create_voice_client(page_url, settings, parent=None):
    # Keep desktop dependencies optional for the server and its test suite.
    from PySide6.QtCore import QObject, Signal, QTimer, QUrl, QByteArray
    from PySide6.QtMultimedia import QAudioSource, QAudioFormat, QAudioOutput, QMediaDevices, QMediaPlayer, QtAudio
    from PySide6.QtNetwork import QAbstractSocket
    from PySide6.QtWebSockets import QWebSocket

    class VoiceClient(QObject):
        state_changed = Signal(str)
        text = Signal(str, str)
        status_changed = Signal()

        def __init__(self):
            super().__init__(parent)
            self.running = False
            self.connected = False
            self.state = "idle"
            self.phrase = settings.wake_phrase
            self.frames = 0
            self.turn = None
            self.error = None
            self.capture = None
            self.capture_buffer = bytearray()
            self.mute_until = 0
            self.playing = False
            self.active_path = None
            self.playback = PlaybackFrames()
            self.files = tempfile.TemporaryDirectory(prefix="lari-audio-")
            self.socket = QWebSocket()
            self.socket.connected.connect(self.on_connected)
            self.socket.disconnected.connect(self.on_disconnected)
            self.socket.textMessageReceived.connect(self.on_text)
            self.socket.binaryMessageReceived.connect(self.on_binary)
            self.socket.errorOccurred.connect(lambda *_: self.on_socket_error())
            self.retry = QTimer(self)
            self.retry.setSingleShot(True)
            self.retry.timeout.connect(self.connect)
            self.keepalive = QTimer(self)
            self.keepalive.timeout.connect(lambda: self.send(protocol.client_ping()))
            self.keepalive.start(5000)
            self.player = QMediaPlayer(self)
            self.output = QAudioOutput(self)
            self.player.setAudioOutput(self.output)
            self.player.mediaStatusChanged.connect(self.on_media_status)
            self.player.errorOccurred.connect(lambda *_: self.finish_playback("failed"))
            self.echo_timer = QTimer(self)
            self.echo_timer.setSingleShot(True)
            self.echo_timer.timeout.connect(self.ack_playback)
            self.media_watchdog = QTimer(self)
            self.media_watchdog.setSingleShot(True)
            self.media_watchdog.timeout.connect(lambda: self.finish_playback("failed"))

        def status(self):
            return {"running": self.running, "connected": self.connected, "state": self.state,
                    "phrase": self.phrase, "frames": self.frames, "error": self.error,
                    "client": "native"}

        def set_state(self, state):
            self.state = state
            self.state_changed.emit(state)
            self.status_changed.emit()

        def send(self, message):
            if self.connected:
                self.socket.sendTextMessage(json.dumps(message))

        def start(self):
            if self.running:
                return
            device = QMediaDevices.defaultAudioInput()
            fmt = QAudioFormat()
            fmt.setSampleRate(16000)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if device.isNull() or not device.isFormatSupported(fmt):
                self.error = "Il microfono non supporta PCM16 mono a 16 kHz"
                self.set_state("error")
                return
            self.capture = QAudioSource(device, fmt, self)
            self.capture.setBufferSize(12800)
            # Forward the enum in Python: PySide's dynamically registered QObject
            # slot can fail to convert the QAudio::State meta-function argument.
            self.capture.stateChanged.connect(lambda state: self.on_capture_state(state))
            self.device = self.capture.start()
            if self.device is None:
                self.error = "Microfono non disponibile"
                self.set_state("error")
                return
            self.device.readyRead.connect(self.read_microphone)
            self.running = True
            self.error = None
            self.connect()

        def stop(self):
            self.running = False
            self.retry.stop()
            self.stop_playback()
            if self.capture:
                self.capture.stop()
                self.capture.deleteLater()
                self.capture = None
            self.capture_buffer.clear()
            self.socket.close()
            self.connected = False
            self.set_state("idle")

        def connect(self):
            if self.running and self.socket.state() == QAbstractSocket.SocketState.UnconnectedState:
                self.socket.open(QUrl(websocket_url(page_url)))

        def on_connected(self):
            self.connected = True
            self.error = None
            self.playback.last_turn = None
            self.set_state("listening")

        def on_disconnected(self):
            self.connected = False
            self.stop_playback()
            self.capture_buffer.clear()
            self.status_changed.emit()
            if self.running:
                self.retry.start(3000)

        def on_socket_error(self):
            self.error = "Bridge non raggiungibile, riprovo…"
            self.status_changed.emit()
            if self.running and not self.retry.isActive():
                self.retry.start(3000)

        def on_capture_state(self, state):
            if state == QtAudio.State.StoppedState and self.capture and self.capture.error() != QtAudio.Error.NoError:
                self.error = "Il microfono ha interrotto la cattura"
                self.stop()
                self.set_state("error")

        def read_microphone(self):
            packet = bytes(self.device.readAll())
            if (not self.running or not self.connected or self.playing or self.state == "speaking"
                    or time.monotonic() < self.mute_until):
                self.capture_buffer.clear()
                return
            self.capture_buffer.extend(packet)
            while len(self.capture_buffer) >= 2560:
                if self.socket.bytesToWrite() > 64000:
                    self.capture_buffer.clear()
                    return
                frame = bytes(self.capture_buffer[:2560])
                del self.capture_buffer[:2560]
                self.socket.sendBinaryMessage(QByteArray(frame))
                self.frames += 1

        def on_text(self, raw):
            try:
                message = json.loads(raw)
            except ValueError:
                return
            if not isinstance(message, dict):
                return
            kind = message.get("type")
            if kind == "state":
                state = message.get("state")
                if state not in protocol.STATES:
                    return
                self.turn = message.get("turn", self.turn)
                self.phrase = message.get("phrase", self.phrase)
                self.error = message.get("error")
                self.set_state("speaking" if state == "listening" and self.playing else state)
            elif kind in {"partial_transcript", "transcript", "reply"}:
                self.text.emit("assistant" if kind == "reply" else "user", str(message.get("text", "")))
            elif kind in {"audio_start", "audio_chunk", "audio_end", "audio"}:
                if self.playback.metadata(message):
                    self.playing = True
                    self.set_state("speaking")
                    self.media_watchdog.start(45000)
                    self.play_next()
            elif kind == "interrupt_ack" and message.get("turn") == self.turn:
                self.playback.last_turn = self.turn
                self.stop_playback()
                self.mute_until = time.monotonic() + max(0, message.get("echo_tail_ms", 700)) / 1000
                self.set_state("listening")
            elif kind == "fatal":
                self.error = str(message.get("error", "Errore vocale"))
                self.set_state("error")

        def on_binary(self, packet):
            if self.playback.binary(packet):
                self.play_next()

        def play_next(self):
            if self.active_path:
                return
            if self.playback.queue:
                fh = tempfile.NamedTemporaryFile(dir=self.files.name, suffix=".mp3", delete=False)
                with fh:
                    fh.write(self.playback.queue.popleft())
                self.active_path = Path(fh.name)
                self.player.setSource(QUrl.fromLocalFile(str(self.active_path)))
                self.player.play()
                self.media_watchdog.start(45000)
            elif self.playback.mode and self.playback.ended and not self.playback.awaiting and not self.echo_timer.isActive():
                self.media_watchdog.stop()
                self.echo_timer.start(700)

        def clear_active(self):
            self.player.stop()
            self.player.setSource(QUrl())
            if self.active_path:
                self.active_path.unlink(missing_ok=True)
                self.active_path = None

        def on_media_status(self, status):
            if status == QMediaPlayer.MediaStatus.EndOfMedia and self.active_path:
                self.clear_active()
                self.play_next()

        def finish_playback(self, status):
            if not self.playback.mode:
                return
            turn = self.playback.completed()
            self.stop_playback()
            self.mute_until = time.monotonic() + 0.7
            self.send(protocol.client_playback_done(turn=turn, status=status))
            self.error = "Riproduzione audio non riuscita"
            self.set_state("error")

        def ack_playback(self):
            if not self.playback.mode:
                return
            turn = self.playback.completed()
            self.playing = False
            self.send(protocol.client_playback_done(turn=turn))
            self.set_state("listening")

        def stop_playback(self):
            self.echo_timer.stop()
            self.media_watchdog.stop()
            self.playback.reset()
            self.clear_active()
            self.playing = False

        def manual(self, phase):
            if self.running and self.connected and not self.playing:
                self.send(protocol.client_ptt(phase=phase))

        def interrupt(self):
            if self.turn is not None:
                self.send(protocol.client_interrupt(turn=self.turn))

        def shutdown(self):
            self.stop()
            self.files.cleanup()

    return VoiceClient()
