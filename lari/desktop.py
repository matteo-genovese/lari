"""Persistent local satellite with a tray manager and wake-triggered window.

Speech, microphone capture and playback use the reference browser client.
Qt keeps the page alive even when the window is hidden.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from .config import load_settings
from .device import validate_device

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_STATES = {"waking", "recording", "transcribing", "thinking", "speaking"}


def should_open_popup(previous, current, connected):
    return connected and current in ACTIVE_STATES and previous not in ACTIVE_STATES


def satellite_url(settings):
    remote = getattr(settings, "desktop_url", "")
    device = getattr(settings, "device", "")
    if remote:
        try:
            parsed = urlsplit(remote)
            query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
            seen = set()
            for name, value in query:
                if name in seen:
                    raise ValueError("duplicate parameter")
                seen.add(name)
                if name == "device":
                    validate_device(value)
                elif name != "raw" or value != "1":
                    raise ValueError("invalid parameter")
            valid_port = parsed.port is None or 1 <= parsed.port <= 65535
            secure = parsed.scheme == "https" or (
                parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
            valid = (secure and parsed.hostname and valid_port and not parsed.username
                     and not parsed.password and not parsed.fragment
                     and len(parsed.path.strip("/").split("/")) == 1 and parsed.path.strip("/"))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("LARI_DESKTOP_URL: usa l’URL completo del satellite HTTPS con token.")
        if device:
            query = [(name, value) for name, value in query if name != "device"]
            query.append(("device", device))
        return urlunsplit(parsed._replace(
            path=parsed.path.rstrip("/") + "/",
            query=urlencode(query) if device else parsed.query,
        ))
    if not settings.token:
        raise ValueError("Imposta LARI_TOKEN nel file .env prima di avviare Lari.")
    url = f"http://127.0.0.1:{settings.port}/{quote(settings.token, safe='')}/"
    return url + "?" + urlencode({"device": device}) if device else url


def main():
    parser = argparse.ArgumentParser(description="Lari: assistente vocale nella barra del desktop")
    parser.add_argument("--background", action="store_true", help="Avvia senza aprire il popup")
    parser.add_argument("--toggle-listening", action="store_true", help="Sospendi/riprendi il microfono")
    parser.add_argument("--status", action="store_true", help="Stato del satellite, senza segreti")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    try:
        url = satellite_url(settings)
    except ValueError as exc:
        parser.error(str(exc))

    from PySide6.QtCore import QTimer, QUrl, Qt, QStandardPaths, QEvent
    from PySide6.QtGui import QAction, QIcon
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QSystemTrayIcon
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings, QWebEnginePermission, QWebEngineScript
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Lari")
    app.setDesktopFileName("lari")
    app.setQuitOnLastWindowClosed(False)
    # A per-user local socket makes launcher/tray invocations single-instance.
    socket_name = str(Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.RuntimeLocation)) / "lari-desktop.sock")
    client = QLocalSocket()
    client.connectToServer(socket_name)
    if client.waitForConnected(500):
        client.write(b"status" if args.status else b"toggle" if args.toggle_listening else b"show")
        client.waitForBytesWritten(1000)
        if args.status and client.waitForReadyRead(1000):
            print(bytes(client.readAll()).decode())
        return 0
    if args.status:
        print('{"running":false,"connected":false,"desktop":false}')
        return 1
    QLocalServer.removeServer(socket_name)
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(socket_name):
        raise RuntimeError("Impossibile avviare il gestore Lari")

    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self, target, kind, main_frame):
            # No external document can receive the automatic mic permission.
            return not main_frame or target.toString() == url

        def javaScriptConsoleMessage(self, level, message, line, source):
            # Chromium messages may contain the private token URL.
            pass

    class Popup(QMainWindow):
        def closeEvent(self, event):
            event.ignore()
            self.hide()

        def event(self, event):
            if event.type() == QEvent.Type.WindowDeactivate:
                self.hide()
            return super().event(event)

    window = Popup()
    window.setWindowTitle("Lari")
    window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                          | Qt.WindowType.WindowStaysOnTopHint)
    window.setFixedSize(360, 520)
    icon = QIcon(str(ROOT / "static/assets/icon-192.png"))
    window.setWindowIcon(icon)
    view = QWebEngineView(window)
    page = LocalPage(view)
    view.setPage(page)
    window.setCentralWidget(view)
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
    page.setLifecycleState(QWebEnginePage.LifecycleState.Active)
    # The remote bridge may predate the native API. Adapt its reference client
    # locally; no installation or configuration change is needed on the server.
    adapter = QWebEngineScript()
    adapter.setName("lari-desktop-host")
    adapter.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    adapter.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    adapter.setRunsOnSubFrames(False)
    adapter.setSourceCode((ROOT / "static/desktop-host.js").read_text())
    page.scripts().insert(adapter)
    trusted_origin = QUrl(url)

    def permission(request):
        origin = request.origin()
        if (request.permissionType() == QWebEnginePermission.PermissionType.MediaAudioCapture
                and origin.scheme() == trusted_origin.scheme()
                and origin.host() == trusted_origin.host()
                and origin.port() == trusted_origin.port()):
            request.grant()
        else:
            request.deny()

    page.permissionRequested.connect(permission)
    status = {"running": False, "state": "idle", "connected": False}
    desired_listening = True
    loaded = False
    last_state = "idle"
    pending_control = False

    def show():
        screen = app.screenAt(window.cursor().pos()) or app.primaryScreen()
        area = screen.availableGeometry()
        window.move(area.right() - window.width() - 16, area.top() + 42)
        window.show()
        window.raise_()
        window.activateWindow()

    tray = QSystemTrayIcon(icon, app)
    menu = QMenu()
    status_action = menu.addAction("Lari · connessione in corso…")
    status_action.setEnabled(False)
    menu.addSeparator()
    menu.addAction("Apri Lari", show)
    pause_action = QAction("Sospendi ascolto", menu)
    menu.addAction(pause_action)

    def toggle():
        nonlocal desired_listening, pending_control
        desired_listening = not desired_listening
        pending_control = True
        pause_action.setText("Sospendi ascolto" if desired_listening else "Riprendi ascolto")
        if loaded:
            page.runJavaScript("window.lariDesktop?." + ("start()" if desired_listening else "stop()"))

    pause_action.triggered.connect(toggle)
    menu.addAction("Nascondi popup", window.hide)
    menu.addSeparator()
    menu.addAction("Esci da Lari", app.quit)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: show() if reason in (
        QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
    tray.show()

    def receive():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            def dispatch(connection=connection):
                command = bytes(connection.readAll()).decode()
                if command == "status":
                    connection.write(json.dumps(status).encode())
                    connection.flush()
                elif command == "toggle":
                    toggle()
                else:
                    show()
                connection.disconnectFromServer()
            connection.readyRead.connect(dispatch)
            connection.disconnected.connect(connection.deleteLater)
            if connection.bytesAvailable():
                dispatch()
    server.newConnection.connect(receive)

    def on_load(ok):
        nonlocal loaded
        loaded = ok
        if ok:
            page.runJavaScript("window.lariDesktop?.compact()")
            if desired_listening:
                page.runJavaScript("window.lariDesktop?.start()")
        if not ok:
            status_action.setText("Lari · server non disponibile")
            tray.setToolTip(status_action.text())
            QTimer.singleShot(3000, lambda: view.load(QUrl(url)))
    view.loadFinished.connect(on_load)

    def update(raw):
        nonlocal last_state, desired_listening, pending_control
        if not raw:
            return
        status.update(json.loads(raw))
        if status.pop("hideRequested", False):
            window.hide()
        state = status.get("state", "idle")
        if pending_control:
            if status.get("running") == desired_listening and not status.get("starting"):
                pending_control = False
        elif not status.get("starting"):
            desired_listening = status.get("running", False)
        if should_open_popup(last_state, state, status.get("connected")):
            show()
        last_state = state
        text = ("microfono sospeso" if not desired_listening else
                "errore microfono/voce" if state == "error" else
                "connessione in corso…" if not status.get("connected") else
                f"in ascolto · {status.get('phrase') or settings.wake_phrase}" if state == "listening" else state)
        status_action.setText("Lari · " + text)
        tray.setToolTip(status_action.text())
        pause_action.setText("Sospendi ascolto" if desired_listening else "Riprendi ascolto")

    poll = QTimer(app)
    poll.timeout.connect(lambda: page.runJavaScript(
        "window.lariDesktop ? JSON.stringify(window.lariDesktop.status()) : ''", update) if loaded else None)
    poll.start(250)
    view.load(QUrl(url))
    if not args.background:
        show()
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(server.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
