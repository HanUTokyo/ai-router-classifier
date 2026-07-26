#!/usr/bin/env bash
# Router local-agent 快速启动器
# 启动分发器（:8001）和 local-agent（:8000）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置（可通过环境变量覆盖）──────────────────────────────
# 分发器模型走本机 Ollama（qwen2.5:0.2b 用于路由分类）
ROUTER_OLLAMA_URL="${ROUTER_OLLAMA_URL:-http://127.0.0.1:11434/api/chat}"
OLLAMA_URL="${OLLAMA_URL:-http://192.168.31.216:11434/api/chat}"
ROUTER_MODEL="${ROUTER_MODEL:-qwen2.5:0.2b}"
CHAT_MODEL="${CHAT_MODEL:-gemma4:e4b}"
REASON_MODEL="${REASON_MODEL:-deepseek-r1:8b}"
CODE_MODEL="${CODE_MODEL:-deepseek-coder:6.7b}"
TOOL_CALL_MODEL="${TOOL_CALL_MODEL:-$CHAT_MODEL}"
TOOL_CALL_SUPPORTED_MODELS="${TOOL_CALL_SUPPORTED_MODELS:-}"
UPSTREAM_MEMORY_MODE="${UPSTREAM_MEMORY_MODE:-local}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-120}"

GEMMA_NUM_CTX="${GEMMA_NUM_CTX:-131072}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-1.0}"
GEMMA_TOP_P="${GEMMA_TOP_P:-0.95}"
GEMMA_TOP_K="${GEMMA_TOP_K:-64}"
DISPATCHER_PORT="${DISPATCHER_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
LOG_DIR="$SCRIPT_DIR/router/logs"

# ── 工具函数 ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

port_in_use() { ss -ltnp 2>/dev/null | grep -q ":$1 "; }

kill_port() {
    local port=$1
    local pid
    pid=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null && info "已终止端口 $port 上的进程 (PID $pid)" || true
        sleep 1
    fi
}

export ROUTER_OLLAMA_URL OLLAMA_URL ROUTER_MODEL CHAT_MODEL REASON_MODEL CODE_MODEL TOOL_CALL_MODEL TOOL_CALL_SUPPORTED_MODELS UPSTREAM_MEMORY_MODE OLLAMA_TIMEOUT
export GEMMA_NUM_CTX GEMMA_TEMPERATURE GEMMA_TOP_P GEMMA_TOP_K
export ROUTER_ROUTE_URL="http://127.0.0.1:${DISPATCHER_PORT}/route"

# ── 子命令处理 ──────────────────────────────────────────────
CMD="${1:-start}"

case "$CMD" in

# ────────────────────────────────────────────────────────────
start)
    info "===== Router local-agent 启动器 ====="
    mkdir -p "$LOG_DIR"

    # 检查端口冲突
    for port in "$DISPATCHER_PORT" "$GATEWAY_PORT"; do
        if port_in_use "$port"; then
            warn "端口 $port 已被占用，尝试终止旧进程..."
            kill_port "$port"
        fi
    done

    # 启动分发器
    info "启动分发器 → http://0.0.0.0:${DISPATCHER_PORT}/route"
    python3 -m uvicorn router.app:app \
        --host 0.0.0.0 --port "$DISPATCHER_PORT" \
        --log-level warning \
        >> "$LOG_DIR/dispatcher.log" 2>&1 &
    DISPATCHER_PID=$!
    echo "$DISPATCHER_PID" > "$SCRIPT_DIR/.dispatcher.pid"

    # 等待分发器就绪
    info "等待分发器就绪..."
    for i in $(seq 1 15); do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${DISPATCHER_PORT}/docs')" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    # 启动 local-agent
    info "启动 local-agent → http://0.0.0.0:${GATEWAY_PORT}/api/chat"
    python3 -m uvicorn local_agent:app \
        --host 0.0.0.0 --port "$GATEWAY_PORT" \
        --log-level warning \
        >> "$LOG_DIR/local-agent.log" 2>&1 &
    GATEWAY_PID=$!
    echo "$GATEWAY_PID" > "$SCRIPT_DIR/.gateway.pid"

    sleep 1

    # 健康检查
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${GATEWAY_PORT}/health')" 2>/dev/null; then
        info "local-agent 健康检查 ✓"
    else
        warn "local-agent 暂未响应，请稍候或查看日志：$LOG_DIR/local-agent.log"
    fi

    echo ""
    echo -e "  ${GREEN}分发器${NC}  PID $DISPATCHER_PID  →  http://0.0.0.0:${DISPATCHER_PORT}"
    echo -e "  ${GREEN}local-agent${NC} PID $GATEWAY_PID →  http://0.0.0.0:${GATEWAY_PORT}"
    echo ""
    echo -e "  用法示例（在容器内）："
    echo -e "    POST /api/chat  (Ollama 兼容)"
    echo ""
    echo -e "  停止服务：  bash start.sh stop"
    echo -e "  查看状态：  bash start.sh status"
    echo -e "  实时日志：  bash start.sh logs"
    ;;

# ────────────────────────────────────────────────────────────
stop)
    info "停止所有服务..."
    for pidfile in .dispatcher.pid .gateway.pid; do
        if [[ -f "$SCRIPT_DIR/$pidfile" ]]; then
            pid=$(cat "$SCRIPT_DIR/$pidfile")
            if kill "$pid" 2>/dev/null; then
                info "已停止 PID $pid"
            fi
            rm -f "$SCRIPT_DIR/$pidfile"
        fi
    done
    kill_port "$DISPATCHER_PORT"
    kill_port "$GATEWAY_PORT"
    info "服务已全部停止"
    ;;

# ────────────────────────────────────────────────────────────
restart)
    bash "$0" stop
    sleep 1
    bash "$0" start
    ;;

# ────────────────────────────────────────────────────────────
status)
    echo "── 服务状态 ──────────────────────────────"
    if port_in_use "$DISPATCHER_PORT"; then
        echo -e "  分发器 :${DISPATCHER_PORT}  ${GREEN}运行中${NC}"
    else
        echo -e "  分发器 :${DISPATCHER_PORT}  ${RED}未运行${NC}"
    fi
    if port_in_use "$GATEWAY_PORT"; then
        echo -e "  local-agent :${GATEWAY_PORT}  ${GREEN}运行中${NC}"
    else
        echo -e "  local-agent :${GATEWAY_PORT}  ${RED}未运行${NC}"
    fi

    # 快速健康检查
    if port_in_use "$GATEWAY_PORT"; then
        result=$(python3 -c "
import json, urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:${GATEWAY_PORT}/health', timeout=3)
    data = json.loads(r.read())
    print('health:', data.get('status','?'))
except Exception as e:
    print('health: unreachable -', e)
" 2>&1)
        echo "  $result"
    fi
    echo "──────────────────────────────────────────"
    ;;

# ────────────────────────────────────────────────────────────
logs)
    echo "── 实时日志（Ctrl+C 退出）─────────────────"
    tail -f "$LOG_DIR/dispatcher.log" "$LOG_DIR/local-agent.log" 2>/dev/null \
        || warn "日志文件不存在，请先启动服务"
    ;;

# ────────────────────────────────────────────────────────────
test)
    info "===== 端到端快速测试 ====="
    echo ""

    # 1. 分发器测试
    info "1. 分发器 /route 测试..."
    python3 - <<'PY'
import json, urllib.request, sys
try:
    data = json.dumps({"message": "帮我写一个 Python SQL 查询"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8001/route",
        data=data, headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=15)
    body = json.loads(r.read())
    print(f"   route={body['route']}  confidence={body['confidence']}  source={body['source']}")
except Exception as e:
    print(f"   FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY

    # 2. local-agent /api/chat 测试（chat 类问题，模型推理快）
    info "2. local-agent /api/chat 完整链路测试（可能需要 30-120 秒）..."
    python3 - <<'PY'
import json, urllib.request, sys
try:
    data = json.dumps({
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": "用一句话介绍 Python"}],
        "stream": False
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat",
        data=data, headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=120)
    body = json.loads(r.read())
    print(f"   model={body.get('model', '?')}  done={body.get('done', False)}")
    ans = body.get("message", {}).get("content", "")
    print(f"   answer_len={len(ans)}")
    print(f"   preview: {ans[:120]}")
except Exception as e:
    print(f"   FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY
    echo ""
    info "测试通过 ✓"
    ;;

# ────────────────────────────────────────────────────────────
*)
    echo "用法: bash start.sh [start|stop|restart|status|logs|test]"
    echo ""
    echo "  start    启动分发器(:${DISPATCHER_PORT}) 和 local-agent(:${GATEWAY_PORT})"
    echo "  stop     停止所有服务"
    echo "  restart  重启所有服务"
    echo "  status   查看运行状态"
    echo "  logs     实时查看日志"
    echo "  test     运行端到端快速测试"
    ;;

esac
