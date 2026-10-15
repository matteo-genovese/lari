"""Transparent desktop companion and a native conversation bubble."""
from __future__ import annotations

import math
import json
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
SPRITES = {"listening": "listening", "waking": "listening", "recording": "listening",
           "transcribing": "thinking", "thinking": "thinking", "speaking": "speaking", "error": "error"}


def move_dispatch(x, y, title="Lari Mascot"):
    selector = json.dumps("title:^" + title + "$")
    return ["dispatch", f"hl.dsp.window.move({{window={selector}, x={int(x)}, y={int(y)}, relative=false}})"]


def mascot_svg(state):
    """Adapt the existing vector art for native transparent rendering in memory."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring((ROOT / "static/assets" / f"lare-{SPRITES.get(state, 'idle')}.svg").read_bytes())
    for child in list(root):
        if child.tag.rsplit("}", 1)[-1] == "rect" and child.get("fill") == "#000000":
            root.remove(child)
        if child.tag.rsplit("}", 1)[-1] == "g" and child.get("class") == "state":
            child.attrib.pop("class", None)
    # Browser glow filters allocate very large intermediate buffers in Qt SVG.
    # The companion paints its own small glow instead.
    for element in root.iter():
        element.attrib.pop("filter", None)
        for child in list(element):
            if child.tag.rsplit("}", 1)[-1] == "filter":
                element.remove(child)
    return ET.tostring(root, encoding="utf-8")


def create_companion(app, voice, settings, menu):
    from PySide6.QtCore import Qt, QTimer, QRectF, QPoint, QProcess, QByteArray, QSettings
    from PySide6.QtGui import QPainter, QColor, QCursor, QRadialGradient
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser

    preferences = QSettings("Lari", "desktop")

    class Bubble(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Lari Conversation")
            self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.resize(340, 280)
            self.mover = QProcess(self)
            self.mover.setProgram("hyprctl")
            self.auto_hide = QTimer(self)
            self.auto_hide.setSingleShot(True)
            self.auto_hide.timeout.connect(self.hide)
            self.setStyleSheet("QWidget{color:#e8ecf4;font-size:13px;} QTextBrowser{background:#161a23;border:0;border-radius:12px;padding:8px;} QPushButton{background:#202734;border:0;border-radius:8px;padding:8px;color:#e8ecf4;} QLabel{color:#afb6c5;}")
            box = QVBoxLayout(self)
            top = QHBoxLayout()
            self.caption = QLabel("Lari")
            hide = QPushButton("×")
            hide.setFixedWidth(32)
            hide.clicked.connect(self.hide)
            top.addWidget(self.caption)
            top.addStretch()
            top.addWidget(hide)
            box.addLayout(top)
            self.chat = QTextBrowser()
            self.chat.setOpenExternalLinks(False)
            self.chat.setPlaceholderText("Richiamami con la tua frase vocale.")
            box.addWidget(self.chat)
            self.user_line = QLabel()
            self.user_line.setWordWrap(True)
            box.addWidget(self.user_line)
            buttons = QHBoxLayout()
            talk = QPushButton("Tieni premuto per parlare")
            talk.pressed.connect(lambda: voice.manual("down"))
            talk.released.connect(lambda: voice.manual("up"))
            interrupt = QPushButton("Interrompi")
            interrupt.clicked.connect(voice.interrupt)
            buttons.addWidget(talk)
            buttons.addWidget(interrupt)
            box.addLayout(buttons)

        def paintEvent(self, event):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(QColor("#11151d"))
            p.setPen(QColor("#303848"))
            p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

        def closeEvent(self, event):
            event.ignore()
            self.hide()

        def say(self, who, text):
            if who == "user":
                self.user_line.setText(text[-400:])
            else:
                self.chat.setPlainText(text)
            if text:
                self.reveal()

        def reveal(self):
            screen = app.screenAt(QCursor.pos()) or app.primaryScreen()
            area = screen.availableGeometry()
            x = min(max(companion.position.x() - self.width() + 100, area.left()), area.right() - self.width())
            y = max(area.top(), companion.position.y() - self.height() + 30)
            self.move(x, y)
            self.show()
            self.auto_hide.stop()
            if companion.hyprland:
                def place():
                    if self.mover.state() == QProcess.ProcessState.NotRunning:
                        self.mover.setArguments(move_dispatch(x, y, "Lari Conversation"))
                        self.mover.start()
                QTimer.singleShot(150, place)

    class Companion(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Lari Mascot")
            self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                                | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setFixedSize(150, 170)
            self.setMouseTracking(True)
            self.renderers = {s: QSvgRenderer(QByteArray(mascot_svg(s)), self)
                              for s in ("idle", "listening", "thinking", "speaking", "error")}
            for renderer in self.renderers.values():
                renderer.repaintNeeded.connect(self.update)
            self.roaming = preferences.value("roaming", settings.desktop_roam, type=bool)
            self.direction = 1
            self.pause_until = time.monotonic() + 2
            self.dragging = False
            self.pressed = None
            self.process = QProcess(self)
            self.process.setProgram("hyprctl")
            self.hyprland = bool(shutil.which("hyprctl")) and app.platformName().startswith("wayland")
            area = app.primaryScreen().availableGeometry()
            self.position = QPoint(area.right() - self.width() - 40, area.bottom() - self.height() - 10)
            saved = preferences.value("position")
            if isinstance(saved, QPoint) and area.contains(saved):
                self.position = saved
            self.move(self.position)
            self.animation = QTimer(self)
            self.animation.timeout.connect(self.update)
            self.animation.start(50)
            self.walk = QTimer(self)
            self.walk.timeout.connect(self.step)
            self.walk.start(350)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            state = SPRITES.get(voice.state, "idle") if voice.running else "idle"
            bob = 2 * math.sin(time.monotonic() * (6 if voice.state == "speaking" else 2.5))
            glow = QRadialGradient(75, 105, 62)
            glow.setColorAt(0, QColor(255, 140, 35, 32))
            glow.setColorAt(1, QColor(255, 140, 35, 0))
            painter.setBrush(glow)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(13, 43, 124, 124)
            if self.direction < 0:
                painter.translate(self.width(), 0)
                painter.scale(-1, 1)
            self.renderers[state].render(painter, QRectF(0, bob, 150, 150))
            painter.resetTransform()
            color = "#62d6a6" if voice.connected and voice.running else "#dba45c" if voice.running else "#7b8496"
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(68, 151, 8, 8)

        def move_native(self, point):
            self.position = point
            self.move(point)
            if self.hyprland and self.process.state() == QProcess.ProcessState.NotRunning:
                self.process.setArguments(move_dispatch(point.x(), point.y()))
                self.process.start()

        def step(self):
            if (not self.isVisible() or not self.roaming or self.dragging or bubble.isVisible()
                    or voice.state not in {"idle", "listening"} or time.monotonic() < self.pause_until):
                return
            if self.underMouse():
                return
            area = (app.screenAt(self.position) or app.primaryScreen()).availableGeometry()
            x = self.position.x() + self.direction * 5
            if x < area.left() + 12 or x > area.right() - self.width() - 12:
                self.direction *= -1
                self.pause_until = time.monotonic() + random.uniform(1, 3)
                return
            # Walk horizontally at the user's chosen height after a drag.
            y = min(max(self.position.y(), area.top()), area.bottom() - self.height())
            self.move_native(QPoint(x, y))
            if random.random() < 0.02:
                self.pause_until = time.monotonic() + random.uniform(1, 4)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.RightButton:
                menu.popup(event.globalPosition().toPoint())
            elif event.button() == Qt.MouseButton.LeftButton:
                self.pressed = event.globalPosition().toPoint()
                self.origin = QPoint(self.position)
                self.dragging = True

        def mouseMoveEvent(self, event):
            if self.dragging:
                delta = event.globalPosition().toPoint() - self.pressed
                self.move_native(self.origin + delta)

        def mouseReleaseEvent(self, event):
            if self.dragging and event.button() == Qt.MouseButton.LeftButton:
                self.dragging = False
                self.pause_until = time.monotonic() + 8
                if (event.globalPosition().toPoint() - self.pressed).manhattanLength() < 8:
                    bubble.reveal()
                preferences.setValue("position", self.position)

        def toggle_roaming(self):
            self.roaming = not self.roaming
            preferences.setValue("roaming", self.roaming)

        def show_companion(self):
            self.show()
            QTimer.singleShot(150, lambda: self.move_native(self.position))

        def closeEvent(self, event):
            event.ignore()
            self.hide()

    companion = Companion()
    bubble = Bubble()

    def state_changed(state):
        companion.update()
        names = {"listening": f"Dimmi «{voice.phrase}»", "waking": "Ti ascolto…", "recording": "Ti ascolto…",
                 "transcribing": "Sto trascrivendo…", "thinking": "Ci penso…", "speaking": "Ti rispondo…",
                 "error": "Serve attenzione", "idle": "Microfono sospeso"}
        bubble.caption.setText(names.get(state, state))
        companion.setToolTip(names.get(state, state))
        if state in {"waking", "recording", "thinking", "error"}:
            bubble.reveal()
        if state == "error" and voice.error:
            bubble.chat.setPlainText(voice.error)
        if state == "listening":
            bubble.auto_hide.start(15000)

    voice.state_changed.connect(state_changed)
    voice.text.connect(bubble.say)
    return companion, bubble
