# 快速参考：openclaw 网关集成

## 一句话说明
把 Ollama 的 URL 从 `http://192.168.1.66:11434` 改成网关地址即可，其余代码**一行不动**。

## 最小化改动（只改 URL）

### 前（直连 Ollama）
```python
import requests
response = requests.post("http://192.168.1.66:11434/api/chat",
    json={"model": "qwen", "messages": [{"role": "user", "content": msg}]})
answer = response.json()["message"]["content"]
```

### 后（通过网关，自动智能路由）
```python
import requests
response = requests.post("http://10.0.3.1:8000/api/chat",
    json={"model": "qwen", "messages": [{"role": "user", "content": msg}]})
answer = response.json()["message"]["content"]
```

> `model` 字段由网关自动决策，填什么都会被覆盖为路由后的实际模型。

## 环境变量设置（在 LXC 容器启动前）
```bash
# 仅在需要使用 /ask 原生接口时设置
export GATEWAY_URL="http://10.0.3.1:8000"
export OLLAMA_TIMEOUT="120"  # 可选，default 120s
```

## 网络快速测试
```bash
# 在容器内执行
curl http://10.0.3.1:8000/health

# 若返回 {"status":"ok","service":"local-agent"}，则网络正常
```

## 返回结构
```json
POST /ask 返回：
{
  "route": "code",           // 智能路由决策
  "model": "deepseek-coder", // 使用的模型
  "answer": "..."            // 回答内容
}

简化调用时只需取 answer 字段。
```

## OpenAI 兼容接口（用于 Open WebUI）

网关现已支持 OpenAI 风格端点：

- `GET /v1/models`
- `POST /v1/chat/completions`

流式兼容策略：

- 只有当请求头包含 `Accept: text/event-stream` 且 `stream=true` 时，网关才返回 SSE。
- 其他情况（即使 `stream=true`）会返回标准 JSON，避免客户端将 `data: ...` 误当 JSON 解析。

### Open WebUI 配置

- API Base URL: `http://10.0.3.1:8000/v1`
- API Key: 任意值（例如 `test-key`，当前仅兼容 Bearer 头，不强校验）
- Model: 固定使用 `local-agent`

### 快速验证（Python）
```bash
/opt/ai/Router/.venv/bin/python - <<'PY'
import json, urllib.request

with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=10) as r:
    print(r.read().decode())

payload = json.dumps({
    'model': 'local-agent',
    'stream': False,
    'messages': [{'role': 'user', 'content': 'Hello'}]
}).encode()

req = urllib.request.Request(
    'http://127.0.0.1:8000/v1/chat/completions',
    data=payload,
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer test-key'},
)

with urllib.request.urlopen(req, timeout=120) as r:
    print(r.read().decode())
PY
```

## 错误处理  
```python
try:
    answer = ask("question")
except Exception as e:
    print(f"网关调用失败: {e}")
    # 实现本地备用方案或重试逻辑
```

## 文件位置
- 客户端库：`/opt/ai/Router/gateway_client.py`
- 详细指南：`/opt/ai/Router/openclaw_integration_guide.md`
