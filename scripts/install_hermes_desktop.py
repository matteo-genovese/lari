#!/usr/bin/env python3
"""Install the PC routing skill into an existing Hermes account over SSH."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="SSH account of the Hermes server")
    parser.add_argument("--pc", required=True, help="PC SSH account, reachable from the server")
    parser.add_argument("--launcher", required=True, help="Absolute PC path to lari-desktop")
    args = parser.parse_args()
    if not args.launcher.startswith("/") or args.server.startswith("-") or args.pc.startswith("-"):
        parser.error("Use SSH account names and an absolute launcher path")
    base = ROOT / "integrations/hermes-desktop"
    files = {name: (base / name).read_text() for name in ("SKILL.md", "scripts/pc.py")}
    files["scripts/config.json"] = json.dumps({"host": args.pc, "launcher": args.launcher}, indent=2)
    code = """
import json,sys
from pathlib import Path
root=Path.home()/'.hermes/skills/desktop/lari-desktop'
files=json.loads(sys.stdin.read())
for name,text in files.items():
    if name not in {'SKILL.md','scripts/pc.py','scripts/config.json'}:
        raise SystemExit('Unexpected skill file')
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:
        path.with_suffix(path.suffix+'.bak').write_bytes(path.read_bytes())
    path.write_text(text)
print('Hermes desktop skill installed.')
"""
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=6",
                    "--", args.server, "python3 -c " + shlex.quote(code)],
                   input=json.dumps(files), text=True, check=True, timeout=20)


if __name__ == "__main__":
    main()
