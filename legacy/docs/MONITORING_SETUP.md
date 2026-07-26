# 🤖 LLM 监控系统集成总结

## ✅ 已完成的改动

### 1️⃣ 核心监控模块
- ✅ 创建 `metrics.py` - Prometheus 指标定义和收集
  - `llm_latency_seconds` - 延迟直方图
  - `llm_calls_total` - 总调用计数器
  - `llm_errors_total` - 错误计数器
  - `llm_active_requests` - 活跃请求数指标
  - `llm_tokens_total` - Token 统计
  - `llm_queue_length` - 队列长度

### 2️⃣ 应用集成
✅ **mlx_openai_agent.py**
- 添加 `/metrics` 端点 (Prometheus 格式)
- 集成 `openai_chat_completions` 中的指标收集
- 记录成功/失败/超时的延迟
- 记录 token 使用量

✅ **main.py (Local Agent)**
- 添加 `/metrics` 端点
- 集成 `openai_chat_completions` 中的指标
- 支持流式响应和非流式响应的指标
- 记录各类错误

✅ **router/app.py (Classifier)**
- 添加 `/metrics` 端点
- 集成 `/route` 端点的指标
- 记录分类器的延迟和错误

### 3️⃣ 基础设施配置
✅ **docker-compose.yml**
- Prometheus 服务 (端口 9090)
- Grafana 服务 (端口 3000)
- 持久化存储

✅ **prometheus.yml**
- 配置三个 job 采集指标：
  - mlx-openai-agent (port 8000)
  - local-agent (port 8001)
  - router-classifier (port 8080)
- 采集间隔：5 秒

✅ **Grafana 配置**
- 数据源配置 (prometheus.yml)
- 仪表板预配置 (dashboards.yml)
- 完整的 LLM 监控仪表板 (llm-dashboard.json)

✅ **requirements.txt**
- 添加 `prometheus-client==0.19.0` 依赖

---

## 🚀 如何使用

### 第一步：安装依赖
```bash
pip install -r requirements.txt
```

### 第二步：启动你的应用
确保三个服务运行在：
- 8000 - mlx-openai-agent
- 8001 - local-agent / gateway
- 8080 - router-classifier

```bash
# 例如
python -m uvicorn mlx_openai_agent:app --port 8000 &
python -m uvicorn main:app --port 8001 &
python -m uvicorn router.app:app --port 8080 &
```

### 第三步：启动监控系统
```bash
docker-compose up -d
```

### 第四步：访问 Grafana
打开浏览器访问 http://localhost:3001
- 用户名: admin
- 密码: admin

---

## 📊 指标端点

| 服务 | 端口 | Metrics 端点 |
|------|------|------------|
| MLX OpenAI Agent | 8000 | http://localhost:8000/metrics |
| Local Agent | 8001 | http://localhost:8001/metrics |
| Router Classifier | 8080 | http://localhost:8080/metrics |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3001 | http://localhost:3001 |

---

## 📈 监控仪表板面板

### 1️⃣ LLM 延迟分布 (P95/P99)
- 展示延迟的百分位数
- 识别"突然变慢"的问题
- 显示高峰期的影响

### 2️⃣ QPS (每秒请求数)
- 实时吞吐量
- 系统负载变化
- 业务流量趋势

### 3️⃣ 错误率
- 总体错误率趋势
- 不同错误类型的分布
- 错误速率 (errors/min)

### 4️⃣ 活跃请求数
- 当前进行中的请求
- 并发度监控
- 资源利用率指示

---

## 🔍 常见查询

### 查看 P99 延迟
```promql
histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))
```

### 查看错误率
```promql
rate(llm_errors_total[1m]) / rate(llm_calls_total[1m]) * 100
```

### 查看超时错误
```promql
rate(llm_errors_total{error_type="timeout"}[1m])
```

### 查看 QPS
```promql
rate(llm_calls_total[1m])
```

---

## 🔧 故障排查

### 看不到指标？
```bash
# 验证端点是否可达
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:8080/metrics

# 查看 Prometheus 状态
curl http://localhost:9090/api/v1/targets
```

### Grafana 中看不到数据？
1. 检查 Prometheus 数据源是否正常连接
2. 确认仪表板是否导入了
3. 重启 Prometheus: `docker-compose restart prometheus`

### Docker 容器无法启动？
```bash
# 查看日志
docker-compose logs prometheus
docker-compose logs grafana

# 重新启动
docker-compose down
docker-compose up -d
```

---

## 📚 相关文档

- 📖 [监控使用指南](MONITORING_GUIDE.md) - 详细的指标说明和 PromQL 示例
- 🏗️ [Prometheus 官方文档](https://prometheus.io/docs/)
- 📊 [Grafana 官方文档](https://grafana.com/docs/)

---

## 💡 下一步

### 推荐配置
1. **设置告警规则** - 在 Prometheus 中定义高延迟/高错误率告警
2. **集成 AlertManager** - 配置邮件/钉钉通知
3. **自定义仪表板** - 根据业务需求调整 Grafana 面板
4. **长期存储** - 配置 Prometheus 远程存储以保留更长的历史数据

### 可视化改进
- 添加自定义告警面板
- 集成业务指标 (如用户满意度)
- 创建每日/每周的报告

---

## 📋 文件清单

新增文件：
- `metrics.py` - Prometheus 指标定义
- `MONITORING_GUIDE.md` - 监控使用指南
- `docker-compose.yml` - Docker 容器编排
- `prometheus.yml` - Prometheus 配置
- `requirements.txt` - Python 依赖
- `grafana-provisioning/datasources/prometheus.yml` - 数据源配置
- `grafana-provisioning/dashboards/dashboards.yml` - 仪表板预配置
- `grafana-provisioning/dashboards/llm-dashboard.json` - 主监控仪表板

修改的文件：
- `mlx_openai_agent.py` - 添加指标集成
- `main.py` - 添加指标集成
- `router/app.py` - 添加指标集成

---

## 🎯 监控目标

✅ 实时追踪 LLM 调用延迟
✅ 监控系统吞吐量 (QPS)
✅ 检测错误和异常情况
✅ 识别性能瓶颈
✅ 支持业务决策和容量规划
