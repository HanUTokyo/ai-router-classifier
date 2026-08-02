#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMMAND="${1:-serve}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

case "$COMMAND" in
setup)
    if [[ -e "$SCRIPT_DIR/.venv" ]] && ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
        BROKEN_VENV="$SCRIPT_DIR/.venv.broken.$(date +%Y%m%d%H%M%S)"
        mv "$SCRIPT_DIR/.venv" "$BROKEN_VENV"
        echo "Moved broken environment to $BROKEN_VENV"
    fi
    if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        python3 -m venv "$SCRIPT_DIR/.venv"
    fi
    "$SCRIPT_DIR/.venv/bin/python" -m pip install --upgrade pip
    "$SCRIPT_DIR/.venv/bin/python" -m pip install -r requirements.lock
    "$SCRIPT_DIR/.venv/bin/python" -m pip install -e . --no-deps
    "$SCRIPT_DIR/.venv/bin/python" -m pip check
    ;;
serve)
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" serve
    ;;
review)
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" review
    ;;
doctor)
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" doctor
    ;;
evaluate)
    shift
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" evaluate "$@"
    ;;
benchmark)
    shift
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" benchmark "$@"
    ;;
rules)
    shift
    exec "$PYTHON_BIN" -m router --config "${AI_ROUTER_CONFIG:-config/router.yaml}" rules "$@"
    ;;
test)
    exec "$PYTHON_BIN" -m pytest
    ;;
*)
    echo "Usage: bash start.sh [setup|serve|review|doctor|evaluate|benchmark|rules|test]"
    exit 2
    ;;
esac
