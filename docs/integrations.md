# Integrations

## Status bar

See [integrations/bar/](../integrations/bar/README.md) for the standard-library Python poller, Omarchy/Quickshell and Waybar configuration. It reads the version-1 [status payload](protocol.md#device-identity-and-status). Both the poller and the token-protected bridge status route are implemented in this checkout.

Keep authenticated status URLs in private configuration. The bar's `lari-desktop` click action requires the desktop launcher installed and available in `PATH`.

## Hermes API tunnel

The machine running the bridge must be able to reach `LARI_HERMES_API`. If Hermes runs on another machine with its API listening only on loopback, run this on the bridge machine, replacing the SSH target with your own:

```bash
ssh -N -L 127.0.0.1:8642:127.0.0.1:8642 -o ExitOnForwardFailure=yes <user>@<hermes-host>
```

Keep the tunnel running and set private `LARI_HERMES_API=http://127.0.0.1:8642`. This forwards the bridge machine's loopback port to port `8642` on the Hermes machine; change ports to match your actual API. Supply `LARI_HERMES_KEY` if Hermes requires authentication. SSH forwarding does not configure or start the Hermes API server. If the local port is already occupied, choose another local port and update `LARI_HERMES_API` accordingly.

For the remote desktop topology, the desktop connects to the bridge's HTTPS page while the bridge uses this tunnel to Hermes. The desktop cannot make its own loopback API reachable to a remote bridge merely by setting `LARI_DESKTOP_URL`.

## Another satellite client

Use the token-scoped WebSocket and [protocol](protocol.md). Capture/resample to 16 kHz mono PCM16 little-endian, send JSON controls, consume semantic state and transcripts, and play binary MP3 frames after their metadata. Preserve turn IDs and sequence order, acknowledge actual playback, and wait through interruption echo tails. Manual controls use the same down/up pair as the reference page.

Use different valid device names for separate Hermes conversation keys. Direct WebSocket clients must send already valid names; the page's normalization/fallback is not a WebSocket behavior. Keep the token private. There is no voice tool-approval input.

Usage is available at `/<token>/usage`, with optional estimates from your configured rate. The local Realtime cap is not an account spending limit.
