# Desktop satellite

`lari/desktop.py` is a Qt/PySide6 tray host for the reference browser satellite. Qt WebEngine keeps microphone capture and playback alive while the popup is hidden. Hermes remains the assistant; the host does not run another assistant backend.

## Installation and services

Use Python 3.11/3.12. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-desktop.txt
# Complete the model and private .env setup in installation.md, then:
.venv/bin/python scripts/install_desktop.py
```

`requirements-desktop.txt` includes the bridge dependencies, PySide6 and python-dotenv. The installer writes user units `lari-server.service` and `lari-desktop.service`, a launcher at `~/.local/bin/lari-desktop`, and an application entry. It reloads systemd and enables/starts the services; the desktop is also restarted. Changed generated files are backed up with a `.bak` suffix.

In local mode the server unit binds `127.0.0.1` with the configured port and loads `.env` using Uvicorn's `--env-file`. The desktop unit starts `python -m lari.desktop --background` with the graphical session, depends on the server in local mode, and disables Chromium background throttling. The installer preserves existing `.env` values; if absent, it copies the template and generates a token. It does not select a paid STT backend or download speech models.

## Tray, popup and controls

The satellite starts listening when its page loads. Entering an active conversation state opens the compact flame popup, including follow-up turns. Clicking outside, closing the popup or its close control hides it while capture continues. The tray offers open/hide, suspend/resume listening and quit. Suspending stops capture and releases the microphone.

**Hold to talk** starts a manual turn on press and sends it on release. **Tap to talk** starts it on the first tap and closes it on the second. Both use `ptt` down/up and bypass wake detection. The interrupt control cancels the current turn; capture remains muted through the echo tail. These controls do not implement voice barge-in.

With `~/.local/bin` in `PATH`:

```bash
lari-desktop                    # Show the popup; the host is single-instance
lari-desktop --toggle-listening  # Suspend/resume the running host
lari-desktop --status            # Local host status, connection and frame count
systemctl --user restart lari-server lari-desktop
systemctl --user disable --now lari-desktop lari-server
```

`--status` uses the local per-user socket and is separate from the bridge's HTTP `/<token>/status` inventory. If no host is running it prints a disconnected result and exits with status 1. Use `--background` to start without opening the popup.

## Remote client mode

Set `LARI_DESKTOP_URL` in the private `.env` to the complete satellite page URL of the existing bridge, then rerun the installer. It disables/stops the local `lari-server` unit and enables/restarts the desktop. This PC supplies microphone and speaker; wake, STT, TTS and Hermes access run on the remote bridge. That bridge supplies the wake configuration and conversation memory.

The URL validator accepts HTTPS or loopback HTTP, a single token path segment, no username/password, and no fragment. It accepts `device` (validated like the page's `?device=`) and `raw=1` as query parameters, and rejects unknown or duplicated ones. `LARI_DEVICE`, when set, is added to the URL and overrides a `device` parameter carried by `LARI_DESKTOP_URL`; without it the host would use that parameter or fall back to `default`. See [device identity and status](protocol.md#device-identity-and-status).

Microphone permission is granted only to the configured origin; other permissions are denied. Main-frame navigation is restricted to the configured page. A bundled adapter adds the native desktop controls to older reference pages. If Hermes is on a different machine, the [API tunnel](integrations.md#hermes-api-tunnel) must run on the bridge machine, not automatically on this client.

## Known limits

- The tray needs StatusNotifierItem support; on Hyprland configure a floating rule for app class `Lari` using the compositor's installed syntax.
- The popup is a fixed-size tool window; closing it is not a microphone pause.
- Remote operation requires network access and a compatible reference page. Qt does not supply an independent voice pipeline.
- Microphone routing, room acoustics, wake calibration and assistant credentials require a real end-to-end test.
- Tool approvals cannot be granted over the voice protocol.
