# 🤖 大模型监控面板快速入门

## 📊 三个核心指标

### 1️⃣ `llm_latency_seconds` - 延迟直方图
👉 **什么是**：记录每次 LLM 调用的延迟分布

📈 **Prometheus 存储**：
```
<0.1s:   50次
<0.5s:   120次
<1s:     200次
<2s:     250次
<5s:     290次
<10s:    300次
```

🎯 **用处**：
- 发现"突然变慢"的问题（模型卡住）
- 识别高峰期的延迟增加
- 计算 P95/P99 延迟

---

### 2️⃣ `llm_calls_total` - 总调用次数
👉 **什么是**：记录总共调用了多少次 LLM

📊 **PromQL 查询**：
```promql
# QPS (每秒请求数)
rate(llm_calls_total[1m])

# 总调用数
llm_calls_total
```

🎯 **用处**：
- 看系统压力
- 是否过载
- 业务流量变化

---

### 3️⃣ `llm_errors_total` - 错误次数
👉 **什么是**：记录超时、异常、空返回等错误

🔴 **错误类型**：
- `timeout` - 超时
- `ConnectionError` - 连接错误
- `ValueError` - 值错误
- `exception` - 其他异常

📊 **PromQL 查询**：
```promql
# 错误率 (%)
rate(llm_errors_total[1m]) / rate(llm_calls_total[1m]) * 100

# 按错误类型分类
rate(llm_errors_total{error_type="timeout"}[1m])
```

🎯 **用处**：
- 模型是否不稳定
- 网络是否有问题
- 需要报警的阈值

---

## 🚀 快速启动

### 1️⃣ 安装依赖
```bash
pip install prometheus-client fastapi
```

### 2️⃣ 启动监控服务
```bash
# 启动 Prometheus + Grafana
docker-compose up -d

# 检查状态
docker-compose ps
```

### 3️⃣ 访问面板
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
  - 用户名: admin
  - 密码: admin

---

## 📊 Grafana 面板说明

### 📈 1️⃣ 延迟曲线 - "模型有没有变慢"
```
时间 → latency
展示：
✓ P95 延迟 (95% 的请求快于这个时间)
✓ P99 延迟 (99% 的请求快于这个时间)
```

**看什么**：
- 📉 曲线上升 = 模型变慢
- 🔴 突然尖峰 = 模型可能卡住
- ⚠️ 高峰期变化 = 负载影响

### 📊 2️⃣ QPS (每秒请求数) - "系统有多忙"
```
QPS = calls/second
```

**看什么**：
- 📈 上升趋势 = 业务增长
- 🔴 突然下降 = 可能出错了
- ⚠️ 超过阈值 = 系统过载

### 🚨 3️⃣ 错误率 - "模型稳不稳定"
```
错误率 = errors/total
```

**看什么**：
- 🟢 < 1% = 正常
- 🟡 1-5% = 注意
- 🔴 > 5% = 需要处理

**常见错误**：
- 🕐 `timeout` = 模型响应太慢
- 🌐 `ConnectionError` = 网络问题
- 💔 其他异常 = 代码问题

---

## 💡 PromQL 查询示例

### 查询 P99 延迟
```promql
histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))
```

### 查询特定模型的错误率
```promql
rate(llm_errors_total{model="qwen"}[1m]) / rate(llm_calls_total{model="qwen"}[1m])
```

### 查询超时错误的速率
```promql
rate(llm_errors_total{error_type="timeout"}[1m])
```

### 查询活跃请求数
```promql
llm_active_requests
```

---

## 🔧 代码集成

### 在你的 FastAPI 应用中

```python
from metrics import (
    llm_calls_total,
    llm_latency_seconds,
    llm_errors_total,
    record_llm_tokens,
    get_metrics,
)
import time

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    start = time.time()
    model = req.model
    
    # 记录调用
    llm_calls_total.labels(model=model, endpoint="chat").inc()
    
    try:
        # 你的 LLM 调用逻辑
        result = call_llm(req.messages)
        
        # 记录成功 + 延迟
        elapsed = time.time() - start
        llm_latency_seconds.labels(
            model=model,
            endpoint="chat",
            status="success"
        ).observe(elapsed)
        
        # 记录 token 数
        record_llm_tokens(
            model=model,
            input_tokens=result['usage']['prompt_tokens'],
            output_tokens=result['usage']['completion_tokens']
        )
        return result
        
    except Exception as e:
        # 记录错误
        elapsed = time.time() - start
        llm_errors_total.labels(
            model=model,
            endpoint="chat",
            error_type=type(e).__name__
        ).inc()
        raise

@app.get("/metrics")
def metrics():
    return get_metrics()
```

---

## 📈 建议的告警规则

### 🔴 高延迟告警
```promql
# P95 延迟 > 5 秒
histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m])) > 5
```

### 🔴 高错误率告警
```promql
# 错误率 > 5%
rate(llm_errors_total[1m]) / rate(llm_calls_total[1m]) > 0.05
```

### 🔴 超时错误告警
```promql
# 超时速率 > 10/分钟
rate(llm_errors_total{error_type="timeout"}[1m]) > 10/60
```

---

## 🛠️ 故障排查

### 问题：看不到指标
**检查**：
```bash
# 1. 确认服务运行
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:8080/metrics

# 2. 检查 prometheus.yml 配置
cat prometheus.yml

# 3. 重启 Prometheus
docker-compose restart prometheus
```

### 问题：Grafana 看不到数据
**检查**：
1. Prometheus 是否连接正常 → Grafana → Configuration → Data Sources
2. 仪表板是否正确导入 → Dashboards → Import

---

## 📚 更多资源

- [Prometheus 文档](https://prometheus.io/docs/)
- [Grafana 文档](https://grafana.com/docs/)
- [PromQL 查询指南](https://prometheus.io/docs/prometheus/latest/querying/basics/)
