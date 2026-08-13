#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KEY_FILE="${AI_ROUTER_LAN_KEY_FILE:-$PROJECT_DIR/.router-lan.key}"

if [[ ! -r "$KEY_FILE" ]]; then
    echo "Missing LAN API key file: $KEY_FILE" >&2
    echo "Create it with: umask 077; openssl rand -hex 32 > $KEY_FILE" >&2
    exit 1
fi

export AI_ROUTER_BIND_HOST="${AI_ROUTER_BIND_HOST:-0.0.0.0}"
export AI_ROUTER_PORT="${AI_ROUTER_PORT:-8000}"
export AI_ROUTER_API_KEY="$(< "$KEY_FILE")"
export AI_ROUTER_CONFIG="${AI_ROUTER_CONFIG:-config/router.full.yaml}"

exec "$PROJECT_DIR/start.sh" serve
