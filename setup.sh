#!/usr/bin/env bash
# Lari setup: virtualenv, dependencies, Italian Vosk model, URL token, .env.
# Idempotent: safe to re-run; existing files, models and .env are kept.
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
MODEL_DIR=models/vosk-model-small-it-0.22
MODEL_URL=https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet -r requirements.txt

if [ ! -f "$MODEL_DIR/am/final.mdl" ]; then
    echo "Downloading the Italian Vosk model (~45 MB)..."
    mkdir -p models
    curl -fsSL "$MODEL_URL" -o models/vosk-it.zip
    unzip -q models/vosk-it.zip -d models
    rm models/vosk-it.zip
    [ -f "$MODEL_DIR/am/final.mdl" ] || { echo "Unexpected model layout in $MODEL_DIR" >&2; exit 1; }
fi

if [ ! -f .env ]; then
    cp .env.example .env
    TOKEN="$("$VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    sed -i "s|^#\{0,1\}LARI_TOKEN=.*|LARI_TOKEN=$TOKEN|" .env
    chmod 600 .env
    echo "Created .env with a generated LARI_TOKEN — keep it secret."
fi

echo "Setup complete. Next: edit .env (Hermes endpoint, providers), then run:"
echo "  .venv/bin/uvicorn lari.server:app --host 127.0.0.1 --port \${LARI_PORT:-8643}"
