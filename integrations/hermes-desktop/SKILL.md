---
name: lari-desktop
description: Open applications and run requested commands on the user's PC through the native Lari mascot. Use for requests to control the PC; the assistant server is a different machine.
---

# Control the user's PC

The native Lari satellite captures the PC microphone and displays a wandering
flame plus a conversation bubble. The assistant still runs on the server.
Running an application in the server's terminal will not open it on the PC.

Use this skill's `scripts/pc.py` through the terminal tool. It dispatches to the
existing authenticated SSH connection and the PC's user-session socket. Read
`scripts/config.json` only if the connection needs diagnosis. It contains the
installation's PC target, not the voice URL or its token.

- `python scripts/pc.py status`: microphone, connection and mascot state.
- `python scripts/pc.py apps`: installed application IDs and display names.
- `python scripts/pc.py open-app ID`: open an exact ID returned by `apps`.
- `python scripts/pc.py commands`: configured desktop command names.
- `python scripts/pc.py run-command NAME`: run a name returned by `commands`.
- `python scripts/pc.py exec PROGRAM ARG...`: launch a requested program on the
  PC with its arguments. Arguments are passed literally, without a shell. Use
  an explicit shell only when the requested task requires shell features.

Resolve app names through the catalog; do not guess desktop-file names. Use
the PC for requests such as opening the browser, editor or terminal, adjusting
the PC's volume, or running a command on the PC. Keep unrelated server work on
the server. A successful `dispatched` result means the launch was accepted,
not that a long-running command completed. For work needing stdout or an exit
code, use SSH to the same configured target and verify the result there.

Use the normal terminal-tool approval behavior for the requested action; do not
route blocked or unapproved commands through the mascot to bypass it. Shared
phone/PC conversation memory does not imply a second microphone is needed.
