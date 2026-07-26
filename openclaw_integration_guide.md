# openclaw 通过网关访问 Ollama 改造指南

## 场景
- **NUC 主机**：运行网关服务（端口 8000/8001）
- **LXC 虚拟机**：运行 openclaw 应用
- **目标**：openclaw 改成调用网关接口而非直接调 Ollama

## 网络配置

### 1. NUC 侧：确认 local-agent 在 0.0.0.0 监听
当前 local-agent 已配置为 `--host 0.0.0.0 --port 8000`，可被容器访问。

### 2. LXC 容器侧：获取 NUC 的可访问地址
容器内访问 NUC 主机的三种方式（任选其一）：

#### 方式 A：使用 LXC bridge 网关 IP（推荐）
```bash
# 在容器内测试连通性
ping 10.0.3.1
curl http://10.0.3.1:8000/health
```
若能连通，后续用 `http://10.0.3.1:8000` 作为网关地址。

#### 方式 B：使用 NUC 实际 LAN IP
```bash
# 在 NUC 上查看 IP（假设网卡是 eth0）
ip addr show eth0
# 假设输出 192.168.1.100

# 在容器内测试
curl http://192.168.1.100:8000/health
```

#### 方式 C：使用 localhost（仅当容器与 host 共享网络时）
```bash
curl http://localhost:8000/health
```

**选择推荐**：使用方式 A（10.0.3.1）最稳定。

---

## openclaw 改造步骤

### 步骤 1：复制网关客户端库到 openclaw 项目
```bash
# 在 NUC 上：
cp /opt/ai/Router/gateway_client.py /path/to/openclaw/

# 或在 openclaw 项目中执行：
wget http://10.0.3.1:8000/../gateway_client.py
```

### 步骤 2：修改 openclaw 的模型调用代码（只改 URL）

#### 旧代码示例（直接调 Ollama）
```python
import requests

OLLAMA_URL = "http://192.168.1.66:11434/api/chat"

def ask_llm(message: str) -> str:
    payload = {
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
    data = resp.json()
    return data["message"]["content"]
```

#### 新代码（只改 URL，其余一行不动）
```python
import requests

OLLAMA_URL = "http://10.0.3.1:8000/api/chat"  # ← 只改这一行

def ask_llm(message: str) -> str:
    payload = {
        "model": "gemma4:e4b",          # 网关会忽略此字段，自动路由选模型
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
    data = resp.json()
    return data["message"]["content"]   # 返回结构与 Ollama 完全兼容
```

### 步骤 3：无需设置任何额外环境变量
由于接口已与 Ollama 完全兼容，不需要引入任何新库或环境变量。

---

## 网关接口文档

### OpenAI 兼容端点（推荐给 Open WebUI）

网关新增 OpenAI 风格协议：

- `GET /v1/models`
- `POST /v1/chat/completions`

其中 `POST /v1/chat/completions` 支持：

- 非流式：`"stream": false`
- 流式 SSE：`"stream": true`

说明：`Authorization: Bearer ...` 请求头当前只做协议兼容，不做强校验。

Open WebUI 建议配置：

- Base URL: `http://10.0.3.1:8000/v1`
- API Key: 任意非空字符串
- Model: 固定使用 `local-agent`

### POST /ask
问答接口，返回分类路由后的大模型回答。

**请求：**
```json
{
  "message": "帮我写一个任务管理系统"
}
```

**响应：**
```json
{
  "route": "code",           // 分类结果: code | reason | chat
  "model": "deepseek-coder:6.7b",  // 选中的模型
  "confidence": 0.9,         // 分类置信度
  "source": "rule_strong",   // 分类源信息
  "answer": "..."            // 大模型回答
}
```

**超时**：默认 120 秒。可通过 `OLLAMA_TIMEOUT` 环境变量在网关端调整。

### GET /health
健康检查，用于联通性测试。

**响应：**
```json
{
  "status": "ok",
  "service": "local-agent"
}
```

---

## 故障排查

### 问题 1：容器无法连接网关
```bash
# 在容器内测试
ping 10.0.3.1
curl -v http://10.0.3.1:8000/health

# 若无法 ping，检查 LXC 网络配置
cat /etc/lxc/net.conf
```

### 问题 2：网关返回超时
容器内请求超时，可能原因：
- Ollama 响应慢：增加 `OLLAMA_TIMEOUT` 环境变量（单位秒）
- 网络延迟：检查 NUC 与容器网络连接质量

```bash
# 在 NUC 上临时调大超时
export OLLAMA_TIMEOUT=180
```

### 问题 3：分类模型无法达到（model_error_fallback）
若网关返回 `source: "model_error_fallback"`，说明分类模型（qwen2.5:0.2b）失败，已自动回退到 chat 模型。
- 检查 Ollama 模型是否正常加载：`curl http://192.168.1.66:11434/api/tags`

---

## 性能建议

1. **连接池复用**：gateway_client 内部使用 requests 会话，建议应用级保持单例
```python
# 推荐：单例模式
gateway = get_gateway_client()
```

2. **超时策略**：
   - 容器级 requests 超时：120 秒
   - 网关级 Ollama 超时：可通过 `OLLAMA_TIMEOUT` 配置

3. **错误处理**：网关会自动回退到 chat 模型，但需在客户端捕获异常
```python
try:
    answer = gateway.ask("...")
except requests.RequestException as e:
    # 处理network error
    print(f"Network error: {e}")
```

---

## 验证：完整测试流程

### 在 NUC 上启动网关
```bash
cd /opt/ai/Router

# 启动分发器（port 8001）
python3 -m uvicorn router.app:app --host 0.0.0.0 --port 8001 &

# 启动 local-agent（port 8000）
python3 -m uvicorn local_agent:app --host 0.0.0.0 --port 8000 &
```

### 在容器内测试
```bash
export GATEWAY_URL="http://10.0.3.1:8000"

python3 - <<'EOF'
from gateway_client import ask
answer = ask("帮我写一个 Python 函数")
print(answer)
EOF
```

若输出完整回答，则改造成功。

---

## openclaw 集成检查清单
- [ ] 复制 gateway_client.py 到 openclaw 项目
- [ ] 修改模型调用代码，改用 GatewayClient
- [ ] 设置 GATEWAY_URL 环境变量（或代码中指定网关地址）
- [ ] 测试网络连通性（`curl http://10.0.3.1:8000/health`）
- [ ] 运行完整问答测试
- [ ] 监控网关日志（router/logs/router.log）
