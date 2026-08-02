#!/usr/bin/env bash
set -eu

BASE_URL="${AI_ROUTER_DEMO_URL:-http://127.0.0.1:8000}"
API_KEY="${AI_ROUTER_API_KEY:-local-demo-key}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_case() {
    expected="$1"
    message="$2"
    response="$(curl --fail --silent --show-error \
        "$BASE_URL/route" \
        -H "X-API-Key: $API_KEY" \
        -H 'Content-Type: application/json' \
        --data "{\"message\":\"$message\"}")"
    printf '%s' "$response" | "$PYTHON_BIN" -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
actual = payload["route"]
source = payload["source"]
confidence = payload["confidence"]
latency_ms = payload["latency_ms"]
degraded = payload["degraded"]
print(
    f"expected={expected:<6} actual={actual:<6} "
    f"source={source:<23} "
    f"confidence={confidence!s:<6} "
    f"latency_ms={latency_ms:<8} "
    f"degraded={degraded}"
)
if actual != expected:
    raise SystemExit(1)
' "$expected"
}

run_case code "Write a Python function that validates an email address"
run_case reason "Why does inflation reduce household purchasing power?"
run_case chat "Hello"
