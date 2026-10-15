#!/usr/bin/env python3
"""Route desktop requests to the native satellite over an existing SSH account."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Controlli Lari sul PC")
    parser.add_argument("operation", choices=("status", "apps", "commands", "open-app", "run-command", "exec"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation in {"status", "apps", "commands"} and args.arguments:
        parser.error("Questa operazione non accetta argomenti")
    if args.operation in {"open-app", "run-command"} and len(args.arguments) != 1:
        parser.error("Passa un ID del catalogo")
    if args.operation == "exec" and not args.arguments:
        parser.error("Passa un programma e i suoi argomenti")
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    command = [config["launcher"], "--" + args.operation, *args.arguments]
    remote_command = shlex.join(command)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                             "-o", "ConnectTimeout=6", "--", config["host"], remote_command],
                            capture_output=True, text=True, timeout=20)
    print(result.stdout, end="")
    if result.returncode:
        print("Controllo PC non riuscito: " + result.stderr[-1000:], file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
