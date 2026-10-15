#!/usr/bin/env python3
"""Install user services and launcher; no root access or private values in files."""
from pathlib import Path
import secrets
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
HOME_DIR = Path.home()


def write(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def main():
    python = ROOT / ".venv/bin/python"
    if not python.exists():
        raise SystemExit("Crea prima .venv e installa requirements-desktop.txt")
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text((ROOT / ".env.example").read_text().replace(
            "LARI_TOKEN=\n", "LARI_TOKEN=" + secrets.token_urlsafe(32) + "\n", 1))
        env_file.chmod(0o600)
    from dotenv import dotenv_values
    values = dotenv_values(env_file)
    remote_client = bool(values.get("LARI_DESKTOP_URL"))
    if not remote_client and not values.get("LARI_TOKEN"):
        raise SystemExit("Imposta LARI_TOKEN nel file .env")
    port = int(values.get("LARI_PORT") or "8643")
    if not 1 <= port <= 65535:
        raise SystemExit("LARI_PORT non valido")
    # systemd quoting differs from shell quoting; protect paths and specifiers.
    def unit_path(path):
        return '"' + str(path).replace("%", "%%").replace('"', '\\"') + '"'
    units = HOME_DIR / ".config/systemd/user"
    write(units / "lari-server.service", f'''[Unit]
Description=Lari local voice bridge
After=network.target

[Service]
WorkingDirectory={str(ROOT).replace('%', '%%')}
ExecStart={unit_path(python)} -m uvicorn lari.server:app --host 127.0.0.1 --port {port} --env-file {unit_path(env_file)} --no-access-log
Restart=on-failure
RestartSec=3
UMask=0077

[Install]
WantedBy=default.target
''')
    write(units / "lari-desktop.service", f'''[Unit]
Description=Lari desktop voice satellite
After=graphical-session.target{'' if remote_client else ' lari-server.service'}
{'' if remote_client else 'Wants=lari-server.service'}
PartOf=graphical-session.target

[Service]
WorkingDirectory={str(ROOT).replace('%', '%%')}
ExecStart={unit_path(python)} -m lari.desktop --background
Restart=on-failure
RestartSec=3
UMask=0077

[Install]
WantedBy=graphical-session.target
''')
    launcher = HOME_DIR / ".local/bin/lari-desktop"
    write(launcher, f'''#!/bin/sh
cd {shlex.quote(str(ROOT))} || exit 1
exec {shlex.quote(str(python))} -m lari.desktop "$@"
''', executable=True)
    write(HOME_DIR / ".local/share/applications/lari.desktop", f'''[Desktop Entry]
Type=Application
Name=Lari
Comment=Assistente vocale: mascotte sul desktop e controlli del PC
Exec="{launcher}"
Icon={ROOT / "static/assets/icon-192.png"}
Terminal=false
Categories=Utility;Audio;
StartupWMClass=Lari
''')
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if remote_client:
        subprocess.run(["systemctl", "--user", "disable", "--now", "lari-server.service"], check=True)
    else:
        subprocess.run(["systemctl", "--user", "enable", "--now", "lari-server.service"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "lari-desktop.service"], check=True)
    subprocess.run(["systemctl", "--user", "restart", "lari-desktop.service"], check=True)
    print("Lari installato: avvio automatico, icona nella barra e popup vocale.")


if __name__ == "__main__":
    main()
