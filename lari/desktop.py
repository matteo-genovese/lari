"""Native voice satellite with a roaming companion and local control socket."""
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


def send_control(path, request):
    """The same user socket works from SSH without a display or browser."""
    import socket
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(str(path))
        connection.sendall(json.dumps(request).encode() + b"\n")
        data = bytearray()
        while b"\n" not in data:
            chunk = connection.recv(65536)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise ValueError("Risposta desktop troppo grande")
        return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description="Lari: mascotte e assistente vocale sul desktop")
    parser.add_argument("--background", action="store_true", help="Avvio automatico del satellite")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--toggle-listening", action="store_true")
    controls.add_argument("--status", action="store_true")
    controls.add_argument("--apps", action="store_true", help="App disponibili su questo PC")
    controls.add_argument("--commands", action="store_true", help="Comandi desktop configurati")
    controls.add_argument("--open-app", metavar="ID", help="Apri una app del catalogo locale")
    controls.add_argument("--run-command", metavar="ID", help="Esegui un comando desktop configurato")
    controls.add_argument("--exec", dest="exec_argv", nargs=argparse.REMAINDER,
                          help="Avvia sul PC un programma con argomenti, senza shell implicita")
    controls.add_argument("--hide-mascot", action="store_true")
    controls.add_argument("--toggle-roaming", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    socket_path = settings.runtime_dir / "lari-desktop.sock"
    operation = ("status" if args.status else "apps" if args.apps else "commands" if args.commands else
                 "open-app" if args.open_app else "run-command" if args.run_command else "exec" if args.exec_argv is not None else
                 "toggle" if args.toggle_listening else "hide" if args.hide_mascot else
                 "roaming" if args.toggle_roaming else "show")
    request = {"operation": operation}
    if args.open_app or args.run_command:
        request["id"] = args.open_app or args.run_command
    if args.exec_argv is not None:
        request["argv"] = args.exec_argv
    try:
        result = send_control(socket_path, request)
        if operation in {"status", "apps", "commands", "open-app", "run-command", "exec"}:
            print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok", True) else 1
    except (OSError, ValueError):
        if operation != "show":
            print(json.dumps({"ok": False, "error": "Lari desktop non è in esecuzione"}))
            return 1
    try:
        url = satellite_url(settings)
    except ValueError as exc:
        parser.error(str(exc))

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon, QAction
    from PySide6.QtNetwork import QLocalServer
    from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
    from .desktop_voice import create_voice_client
    from .desktop_native import create_companion
    from .desktop_actions import application_catalog, command_catalog, execute_action

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Lari")
    app.setDesktopFileName("lari")
    app.setQuitOnLastWindowClosed(False)
    QLocalServer.removeServer(str(socket_path))
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(str(socket_path)):
        raise RuntimeError("Impossibile avviare il gestore Lari")
    voice = create_voice_client(url, settings, app)
    apps = application_catalog()
    try:
        commands = command_catalog(settings.desktop_commands_file)
    except (ValueError, OSError) as error:
        print("Configurazione comandi non valida:", type(error).__name__, file=sys.stderr)
        commands = command_catalog(Path("/nonexistent/lari-commands"))
    icon = QIcon(str(ROOT / "static/assets/icon-192.png"))
    tray = QSystemTrayIcon(icon, app)
    menu = QMenu()
    companion, bubble = create_companion(app, voice, settings, menu)
    status_action = menu.addAction("Lari · connessione in corso…")
    status_action.setEnabled(False)
    menu.addSeparator()
    menu.addAction("Mostra la fiammetta", companion.show_companion)
    menu.addAction("Parla con Lari", bubble.reveal)
    pause_action = menu.addAction("Sospendi ascolto")
    pause_action.triggered.connect(lambda: voice.stop() if voice.running else voice.start())
    roam_action = menu.addAction("Passeggia sullo schermo")
    roam_action.setCheckable(True)
    roam_action.setChecked(companion.roaming)
    roam_action.triggered.connect(companion.toggle_roaming)
    menu.addAction("Nascondi la fiammetta", companion.hide)
    apps_menu = menu.addMenu("Apri un’app")

    def act(request):
        result = execute_action(request, apps, commands)
        if result.get("ok") and result.get("dispatched"):
            bubble.say("assistant", "Avviato: " + result["dispatched"])
        return result

    for app_id, entry in sorted(apps.items(), key=lambda item: item[1]["name"].casefold()):
        apps_menu.addAction(entry["name"], lambda app_id=app_id: act({"operation": "open-app", "id": app_id}))
    commands_menu = menu.addMenu("Comandi del PC")
    for command in sorted(commands):
        commands_menu.addAction(command, lambda command=command: act({"operation": "run-command", "id": command}))
    menu.addSeparator()
    menu.addAction("Esci da Lari", app.quit)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: companion.show_companion() if reason in (
        QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
    tray.show()

    def status():
        result = voice.status()
        result.update({"mascot_visible": companion.isVisible(), "roaming": companion.roaming,
                       "position": [companion.position.x(), companion.position.y()]})
        return result

    def dispatch(request):
        if not isinstance(request, dict):
            return {"ok": False, "error": "Richiesta non valida"}
        operation = request.get("operation")
        if operation == "status":
            return status()
        if operation in {"apps", "commands", "open-app", "run-command", "exec"}:
            return act(request)
        if operation == "toggle":
            voice.stop() if voice.running else voice.start()
        elif operation == "show":
            companion.show_companion()
        elif operation == "hide":
            companion.hide()
            bubble.hide()
        elif operation == "roaming":
            companion.toggle_roaming()
            roam_action.setChecked(companion.roaming)
        else:
            return {"ok": False, "error": "Controllo sconosciuto"}
        return {"ok": True}

    def receive():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            connection.setProperty("request_buffer", b"")
            def read(connection=connection):
                data = connection.property("request_buffer") + bytes(connection.readAll())
                if len(data) > 8192:
                    connection.disconnectFromServer()
                    return
                if b"\n" not in data and data not in {b"show", b"status", b"toggle"}:
                    connection.setProperty("request_buffer", data)
                    return
                try:
                    request = ({"operation": data.decode()} if data in {b"show", b"status", b"toggle"}
                               else json.loads(data.split(b"\n", 1)[0]))
                    response = dispatch(request)
                except (ValueError, TypeError):
                    response = {"ok": False, "error": "Richiesta non valida"}
                connection.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                connection.flush()
                connection.disconnectFromServer()
            connection.readyRead.connect(read)
            connection.disconnected.connect(connection.deleteLater)
            if connection.bytesAvailable():
                read()
    server.newConnection.connect(receive)

    def update_status():
        text = ("microfono sospeso" if not voice.running else
                voice.error if voice.error else "connessione in corso…" if not voice.connected else
                f"in ascolto · {voice.phrase}" if voice.state == "listening" else voice.state)
        status_action.setText("Lari · " + text)
        tray.setToolTip(status_action.text())
        pause_action.setText("Sospendi ascolto" if voice.running else "Riprendi ascolto")
    voice.status_changed.connect(update_status)
    companion.show_companion()
    voice.start()
    # Let Python process termination signals even while the Qt event loop is idle.
    tick = QTimer(app)
    tick.timeout.connect(lambda: None)
    tick.start(500)
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(voice.shutdown)
    app.aboutToQuit.connect(server.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
