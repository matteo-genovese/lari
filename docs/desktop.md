# Desktop satellite

The desktop satellite is a native Qt application. A transparent flame wanders
along the screen edge without taking focus; drag it to reposition it. It stops
walking during a conversation and shows transcripts/replies in a native bubble.
Click the flame to talk, or right-click it for the same manager offered in the
system tray. The microphone remains active when the flame or bubble is hidden.
**Suspend listening** releases the microphone. A roaming toggle is available in
the menu; position and roaming preference persist across restarts.

Audio goes directly from Qt/PipeWire to the bridge over WebSocket: PCM16 mono at
16 kHz, with native MP3 playback, turn-scoped interruption and muted capture
through playback and the echo tail. No browser or web-app window is needed.
The bridge still supplies the wake phrase, STT, TTS and Hermes conversation.

See [installation and services](#installation-and-services) for setup.


For an existing bridge on another machine, put its complete HTTPS satellite URL
(including its private token) in `LARI_DESKTOP_URL` in `.env`. The installer then
disables the unused local bridge service. Your PC provides the microphone and
speaker. Clients with the same device identity share a conversation; distinct device
names have separate persistent sessions. Avoid two microphones listening to
the same room to prevent duplicate turns.

The tray and flame menus can open installed apps and run configured desktop
commands. CLI actions go through a same-user socket so they run inside the
active desktop session, including when invoked over an existing SSH account:

```bash
lari-desktop                          # Show the flame (one instance)
lari-desktop --status                 # Connection, native capture and movement
lari-desktop --toggle-listening       # Suspend/resume microphone capture
lari-desktop --hide-mascot            # Hide the flame and bubble; keep listening
lari-desktop --toggle-roaming         # Pause/resume walking
lari-desktop --apps                   # Installed desktop application IDs
lari-desktop --open-app firefox.desktop
lari-desktop --commands               # lock, mute, volume-up, volume-down, custom names
lari-desktop --run-command volume-up
lari-desktop --exec PROGRAM ARG...     # Launch literal argv; no implicit shell
```

Custom named commands live in `~/.config/lari/commands.json`: a JSON object
mapping names to argument arrays, for example `{"terminal": ["foot"]}`. Set
`LARI_DESKTOP_COMMANDS` for another path. A dispatch result confirms process
launch, not completion. The local control socket is accessible only to your
Unix user; remote control uses that user's existing SSH access.

To let Hermes route requested PC actions correctly, install the bundled
[integration skill](../integrations/hermes-desktop/SKILL.md) in its existing account:

```bash
.venv/bin/python scripts/install_hermes_desktop.py \
  --server user@assistant-server --pc user@your-pc \
  --launcher /home/user/.local/bin/lari-desktop
```

This writes only the routing skill and target configuration. It requires an
existing authenticated SSH connection from the assistant server to the PC;
it creates no public endpoint and copies no voice credentials. Hermes must
still apply its normal tool approval rules to requested commands.

On Hyprland, use floating, pinned rules for class `lari` and titles `Lari Mascot`
and `Lari Conversation`; the mascot rule should disable initial focus. Native
movement uses Hyprland's IPC because Wayland does not let ordinary clients set
arbitrary window positions. Use your compositor's current rule syntax.

```bash
systemctl --user restart lari-desktop
systemctl --user disable --now lari-desktop lari-server  # Disable autostart
```

Unit tests cannot calibrate your room or confirm a real wake → Hermes reply →
audible TTS. Verify that flow on the actual microphone after installation.

## Installation and services

Use Python 3.11/3.12. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-desktop.txt
# Complete the model and private .env setup in installation.md, then:
.venv/bin/python scripts/install_desktop.py
```

`requirements-desktop.txt` includes the bridge dependencies, PySide6 and python-dotenv. The installer writes user units `lari-server.service` and `lari-desktop.service`, a launcher at `~/.local/bin/lari-desktop`, and an application entry. It reloads systemd and enables/starts the services; the desktop is also restarted. Changed generated files are backed up with a `.bak` suffix.

In local mode the server unit binds `127.0.0.1` with the configured port and loads `.env` using Uvicorn's `--env-file`. The desktop unit starts `python -m lari.desktop --background` with the graphical session and depends on the server in local mode. The installer preserves existing `.env` values; if absent, it copies the template and generates a token. It does not select a paid STT backend or download speech models.

## Conversation controls

The native bubble offers **Hold to talk** and **Interrupt**. Holding starts a manual turn on press and sends it on release, bypassing wake detection. Interrupt cancels the current turn; capture remains muted through the echo tail. The browser/PWA also retains its visible hold and tap conversation controls.

`--status` uses `$XDG_RUNTIME_DIR/lari-desktop.sock` (default `/run/user/<uid>/lari-desktop.sock`) and is separate from the bridge HTTP `/<token>/status` inventory. Controls work over SSH without a display when the desktop host is already running; if absent they exit with status 1. `--background` is used by the service; startup shows the companion and starts listening.

## Remote client mode

Set `LARI_DESKTOP_URL` in the private `.env` to the complete satellite page URL of the existing bridge, then rerun the installer. It disables/stops the local `lari-server` unit and enables/restarts the desktop. This PC supplies microphone and speaker; wake, STT, TTS and Hermes access run on the remote bridge. That bridge supplies the wake configuration and conversation memory.

The URL validator accepts HTTPS or loopback HTTP, a single token path segment, no username/password, and no fragment. It accepts `device` (validated like the page's `?device=`) and `raw=1` as query parameters, and rejects unknown or duplicated ones. `LARI_DEVICE`, when set, is added to the URL and overrides a `device` parameter carried by `LARI_DESKTOP_URL`; without it the host would use that parameter or fall back to `default`. See [device identity and status](protocol.md#device-identity-and-status).

The native client connects directly to the token-scoped WebSocket and forwards the selected device identity. `raw=1` is accepted in the page URL for compatibility; it is a browser diagnostic setting and is omitted from the native WebSocket. If Hermes is on another machine, run the [API tunnel](integrations.md#hermes-api-tunnel) on the bridge machine.

## Known limits

- The tray needs StatusNotifierItem support. Arbitrary positioning on Wayland currently uses Hyprland IPC; other compositors may place windows differently.
- Hiding or closing the conversation bubble does not pause the microphone.
- Remote operation requires a compatible bridge and network access.
- Microphone routing, acoustics, wake calibration and credentials need a real end-to-end test.
- Tool approvals cannot be granted over the voice protocol.
