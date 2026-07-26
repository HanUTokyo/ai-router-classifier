#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MLX_AGENT_PORT="${MLX_AGENT_PORT:-8888}"
LOG_DIR="$SCRIPT_DIR/router/logs"
PID_FILE="$SCRIPT_DIR/.mlx_agent.pid"

mkdir -p "$LOG_DIR"

port_in_use() { ss -ltnp 2>/dev/null | grep -Eq ":$1\\b"; }

kill_port() {
    local port=$1
    local pid
    pid=$(ss -ltnp 2>/dev/null | grep -E ":$port\\b" | grep -oP 'pid=\K[0-9]+' | head -1)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}

CMD="${1:-start}"
case "$CMD" in
start)
    if port_in_use "$MLX_AGENT_PORT"; then
        kill_port "$MLX_AGENT_PORT"
    fi
    python3 -m uvicorn mlx_openai_agent:app \
        --host 0.0.0.0 --port "$MLX_AGENT_PORT" \
        --log-level warning \
        >> "$LOG_DIR/mlx-agent.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo "mlx-openai-agent started on :$MLX_AGENT_PORT"
    ;;
stop)
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    kill_port "$MLX_AGENT_PORT"
    echo "mlx-openai-agent stopped"
    ;;
restart)
    bash "$0" stop
    sleep 1
    bash "$0" start
    ;;
status)
    if port_in_use "$MLX_AGENT_PORT"; then
        echo "mlx-openai-agent :$MLX_AGENT_PORT running"
    else
        echo "mlx-openai-agent :$MLX_AGENT_PORT stopped"
    fi
    ;;
*)
    echo "Usage: bash start_mlx_agent.sh [start|stop|restart|status]"
    ;;
esac
