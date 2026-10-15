"""Local application catalog and named desktop actions, never shell strings."""
from __future__ import annotations

import configparser
import json
from pathlib import Path
import subprocess

BUILTIN_COMMANDS = {
    "lock": ["omarchy", "system", "lock"],
    "volume-up": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+", "--limit", "1"],
    "volume-down": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"],
    "mute": ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"],
}


def application_catalog(directories=None):
    directories = directories or [Path.home() / ".local/share/applications", Path("/usr/local/share/applications"),
                                   Path("/usr/share/applications")]
    catalog, seen = {}, set()
    for directory in directories:
        for path in sorted(Path(directory).rglob("*.desktop")):
            app_id = str(path.relative_to(directory)).replace("/", "-")
            if app_id in seen:
                continue
            seen.add(app_id)
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read(path, encoding="utf-8")
                entry = parser["Desktop Entry"]
                if (entry.get("Type") != "Application" or not entry.get("Exec")
                        or entry.get("Hidden", "false").lower() == "true"
                        or entry.get("NoDisplay", "false").lower() == "true"):
                    continue
                catalog[app_id] = {"id": app_id, "name": entry.get("Name", app_id), "path": str(path)}
            except (configparser.Error, KeyError, OSError, UnicodeError):
                continue
    return catalog


def command_catalog(custom_path):
    commands = {name: list(argv) for name, argv in BUILTIN_COMMANDS.items()}
    if custom_path.is_file():
        custom = json.loads(custom_path.read_text())
        if not isinstance(custom, dict):
            raise ValueError("commands.json deve contenere nomi associati a liste di argomenti")
        for name, argv in custom.items():
            if (not isinstance(name, str) or not name or not isinstance(argv, list) or not argv
                    or not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in argv)):
                raise ValueError("Ogni comando deve essere una lista di argomenti, non testo shell")
            commands[name] = argv
    return commands


def execute_action(request, apps, commands, launch=None):
    """Resolve installed IDs before spawning; results describe dispatch, not completion."""
    operation = request.get("operation")
    if operation == "apps":
        return {"ok": True, "apps": [{"id": app["id"], "name": app["name"]} for app in apps.values()]}
    if operation == "commands":
        return {"ok": True, "commands": sorted(commands)}
    if operation == "open-app":
        app = apps.get(request.get("id"))
        if app is None:
            return {"ok": False, "error": "App non presente nel catalogo del PC"}
        argv = ["gio", "launch", app["path"]]
        label = app["name"]
    elif operation == "run-command":
        argv = commands.get(request.get("id"))
        if argv is None:
            return {"ok": False, "error": "Comando non configurato sul PC"}
        label = request["id"]
    elif operation == "exec":
        argv = request.get("argv")
        if (not isinstance(argv, list) or not 1 <= len(argv) <= 256
                or not all(isinstance(arg, str) and "\x00" not in arg for arg in argv)
                or not argv[0]):
            return {"ok": False, "error": "Passa un programma e i suoi argomenti"}
        label = Path(argv[0]).name
    else:
        return {"ok": False, "error": "Azione desktop non riconosciuta"}
    if launch is None:
        def launch(argv):
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        process = launch(list(argv))
        return {"ok": True, "dispatched": label, "pid": process.pid}
    except OSError:
        return {"ok": False, "error": "Impossibile avviare il programma sul PC"}
