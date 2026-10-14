# Lari status in the bar

`lari-bar-status` uses Python 3 and only the standard library. It polls the bridge
with a 2-second timeout and prints one JSON line with `text`, `tooltip` and `class`.
It requires no QML plugins or changes to the bridge.

## Installation

From the repository:

```sh
mkdir -p ~/.local/bin
cp integrations/bar/lari-bar-status ~/.local/bin/lari-bar-status
chmod +x ~/.local/bin/lari-bar-status
```

Make sure `~/.local/bin` is in `PATH`. Configuration is selected
in this order:

1. `--url <url>`: complete URL of the status endpoint.
2. `--env <file>`: dotenv file from which to read `LARI_PORT` and `LARI_TOKEN`.
3. `LARI_ENV`: environment variable containing the dotenv file path.
4. `LARI_STATUS_URL`: environment variable containing the complete endpoint URL,
   validated with the same rules as `--url`.

There is no default path. For example, use `--env '<repo>/.env'`,
replacing `<repo>` with the repository path, or set
`LARI_ENV=~/.config/lari/env` in the bar's environment.
The file is read as data, without executing shell commands.
`LARI_PORT` is optional: if it is absent or empty, the default is `8643`.
The token is not shown in output or errors.

For a remote bridge, `--url` replaces reading `.env`: it must contain
the complete endpoint URL, in the form `https://bridge.example/<token>/status`.
Do not save authenticated URLs in the repository's public configuration.
`--glyph 'mic'` replaces the Nerd Font microphone glyph, including with
plain text.

## Omarchy 4 / Quickshell

Add this module to the `bar.layout.right` array in
`~/.config/omarchy/shell.json`, keeping the other modules:

```json
{
  "id": "lari",
  "type": "command",
  "interval": 3,
  "exec": "~/.local/bin/lari-bar-status --env '<repo>/.env'",
  "onClick": "lari-desktop"
}
```

Replace `<repo>` before use. `lari-desktop` must be installed and
available in `PATH`: clicking opens the satellite popup.
After the first customization, `shell.json` is canonical: it is not
deep-merged with the defaults. Keep the sections and modules
you want to retain. The `command` format and this rule are documented
in the [official Omarchy documentation](https://github.com/omacom/omarchy/blob/quattro/docs/omarchy-shell.md).

## Waybar

Add `custom/lari` to `modules-right` in the Waybar configuration and
insert this block among the configuration properties:

```json
"custom/lari": {
  "exec": "~/.local/bin/lari-bar-status --env '<repo>/.env'",
  "interval": 3,
  "return-type": "json",
  "on-click": "lari-desktop"
}
```

Replace `<repo>` and reload the bar after editing the configuration.

## Verification and states

```sh
lari-bar-status --plain --env '<repo>/.env'
```

With `LARI_ENV` or `LARI_STATUS_URL` in the environment, `lari-bar-status --plain` is enough.
Without configuration, the script returns the `offline` class, the tooltip
«configurazione assente: usa --env, --url o LARI_ENV» and exit code 0.
Example output when the bridge is available:

```text
 2 [busy] Frase di wake: ciao Lari STT: vosk phone: speaking desktop: listening
```

The classes are `idle` (no connected satellites, or inactive satellites),
`listening` (listening), `busy` (waking, recording, transcribing, thinking or
speaking), `error` (error or unknown state) and `offline` (bridge
unavailable, timeout, invalid configuration or response). With multiple connected
satellites, precedence is `busy`, `error`, `listening`, `idle`; the text shows
the glyph and connected count. The tooltip lists each connected satellite and its
state on a separate line, together with the wake phrase and STT backend.

Operational errors still produce output and exit code 0.
Use an interval of **2–5 seconds**: updates occur on the next
poll, so very brief states may not appear.

## Removal

Remove the `lari` module from `bar.layout.right`, or `custom/lari` from the
Waybar block and `modules-right`. Reload the bar, then remove the script:

```sh
rm ~/.local/bin/lari-bar-status
```
