# 🤖 LLM 监控面板 - 完整集成文档

## 📋 概述

已为你的项目成功集成了完整的 **Prometheus + Grafana** 监控系统，用于实时监控大模型（LLM）的性能指标。

### 🎯 监控目标
- ✅ 实时追踪 LLM 调用延迟
- ✅ 监控系统吞吐量 (QPS)
- ✅ 检测错误和异常情况
- ✅ 识别性能瓶颈
- ✅ 支持业务决策和容量规划

---

## 📊 三个核心指标

### 1️⃣ `llm_latency_seconds` - 延迟分布
**是什么**：Histogram，记录每次 LLM 调用的耗时

📈 **Prometheus 存储方式**（不是每条记录）：
```
延迟分布桶：
  <0.05s: 10次
  <0.1s:  50次
  <0.5s:  120次
  <1s:    200次
  <2s:    250次
  <5s:    290次
  <10s:   300次
```

🎯 **用处**：
- 发现"突然变慢"（模型卡住）
- 识别高峰期延迟增加
- 计算 P95/P99 延迟

---

### 2️⃣ `llm_calls_total` - 总调用次数
**是什么**：Counter，记录总共调用了多少次 LLM

📊 **PromQL 查询**：
```promql
rate(llm_calls_total[1m])  # QPS - 每秒请求数
llm_calls_total            # 累计总数
```

🎯 **用处**：
- 看系统压力
- 是否过载
- 业务流量变化

---

### 3️⃣ `llm_errors_total` - 错误次数
**是什么**：Counter，记录超时/异常/空返回等错误

🔴 **错误类型分类**：
- `timeout` - 模型响应超时
- `ConnectionError` - 网络连接失败
- `ValueError` - 数据验证失败
- 其他异常类型

📊 **PromQL 查询**：
```promql
rate(llm_errors_total[1m]) / rate(llm_calls_total[1m])  # 错误率
rate(llm_errors_total{error_type="timeout"}[1m])       # 超时错误
```

🎯 **用处**：
- 模型是否不稳定
- 网络是否有问题
- 需要告警的阈值

---

## 📁 新增文件

### 核心模块
- **`metrics.py`** - Prometheus 指标定义和收集
  - 定义 5 个主要指标
  - 提供装饰器和辅助函数
  - 暴露 `/metrics` endpoint

### 配置文件
- **`prometheus.yml`** - Prometheus 配置
  - 配置 3 个 job（MLX Agent、Local Agent、Router）
  - 采集间隔 5 秒
  
- **`docker-compose.yml`** - 容器编排
  - Prometheus (端口 9090)
  - Grafana (端口 3000)
  - 持久化存储卷

- **`requirements.txt`** - Python 依赖
  - 添加 `prometheus-client==0.19.0`

### Grafana 配置
- **`grafana-provisioning/datasources/prometheus.yml`** - 数据源配置
- **`grafana-provisioning/dashboards/dashboards.yml`** - 仪表板预配置
- **`grafana-provisioning/dashboards/llm-dashboard.json`** - 主监控仪表板
  - 延迟分布图 (P95/P99)
  - QPS 曲线
  - 错误率图
  - 活跃请求数

### 文档
- **`MONITORING_GUIDE.md`** - 详细的监控使用指南
- **`MONITORING_SETUP.md`** - 集成总结和注意事项
- **`README_MONITORING.md`** - 本文档

### 工具
- **`verify_monitoring.py`** - 监控系统验证脚本
- **`start_monitoring.sh`** - 快速启动脚本

---

## ✏️ 修改的文件

### 1. `mlx_openai_agent.py`
```python
# 添加的内容：
- import 指标模块
- /metrics endpoint
- 在 openai_chat_completions 中集成指标收集
  • 记录调用计数
  • 记录延迟
  • 记录错误
  • 记录 token 使用
```

### 2. `main.py` (Local Agent)
```python
# 添加的内容：
- import 指标模块
- /metrics endpoint
- 在 openai_chat_completions 中集成指标
  • 支持流式和非流式响应
  • 记录各类错误
  • 流式响应的延迟在完成时记录
```

### 3. `router/app.py`
```python
# 添加的内容：
- import 指标模块 (处理相对导入)
- /metrics endpoint
- /health endpoint
- 在 /route 端点集成指标
  • 分类器的延迟
  • 分类错误
```

---

## 🚀 快速开始

### 第 1 步：安装依赖
```bash
pip install -r requirements.txt
```

### 第 2 步：启动你的应用
```bash
# 分别在不同终端启动
python -m uvicorn mlx_openai_agent:app --port 8000 &
python -m uvicorn main:app --port 8001 &
python -m uvicorn router.app:app --port 8080 &
```

### 第 3 步：启动监控系统
```bash
bash start_monitoring.sh

# 或者手动
docker-compose up -d
```

### 第 4 步：访问 Grafana
打开浏览器访问 **http://localhost:3001**
- 用户名: `admin`
- 密码: `admin`

### 第 5 步：验证系统
```bash
python verify_monitoring.py
```

---

## 📊 监控指标详解

### 延迟监控
```
展示：P95 和 P99 延迟曲线

看什么：
• 📈 曲线上升 = 模型变慢
• 🔴 突然尖峰 = 模型可能卡住
• ⚠️ 高峰期变化 = 负载影响性能
```

### QPS 监控
```
展示：每秒请求数

看什么：
• 📈 上升趋势 = 业务增长
• 🔴 突然下降 = 可能出错了
• ⚠️ 超过阈值 = 系统过载
```

### 错误率监控
```
展示：错误率百分比

看什么：
• 🟢 < 1% = 正常
• 🟡 1-5% = 注意
• 🔴 > 5% = 需要处理

错误类型：
• 🕐 timeout = 模型响应太慢
• 🌐 ConnectionError = 网络问题
• 💔 其他异常 = 代码问题
```

---

## 💡 常用 PromQL 查询

### 查看 P99 延迟（最多99%的请求快于该时间）
```promql
histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))
```

### 查看 QPS（每秒请求数）
```promql
rate(llm_calls_total[1m])
```

### 查看错误率百分比
```promql
rate(llm_errors_total[1m]) / rate(llm_calls_total[1m]) * 100
```

### 查看特定模型的错误率
```promql
rate(llm_errors_total{model="qwen"}[1m]) / rate(llm_calls_total{model="qwen"}[1m]) * 100
```

### 查看超时错误的速率
```promql
rate(llm_errors_total{error_type="timeout"}[1m])
```

### 查看活跃请求数
```promql
llm_active_requests
```

---

## 🔧 故障排查

### 问题 1：看不到指标数据
**检查步骤**：
```bash
# 1. 验证服务端点
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:8080/metrics

# 2. 查看 Prometheus 目标状态
http://localhost:9090/targets

# 3. 查看 Prometheus 日志
docker-compose logs prometheus

# 4. 重启 Prometheus
docker-compose restart prometheus
```

### 问题 2：Grafana 无法连接 Prometheus
**检查步骤**：
1. 进入 Grafana → Configuration → Data Sources
2. 检查 Prometheus 连接状态
3. 确保 URL 是 `http://prometheus:9090`（内部网络）
4. 或使用 `http://localhost:9090`（如果 Grafana 在主机上）

### 问题 3：Docker 容器无法启动
**检查步骤**：
```bash
# 查看错误日志
docker-compose logs

# 清理旧容器
docker-compose down -v
docker-compose up -d

# 检查端口占用
lsof -i :9090
lsof -i :3000
```

---

## 📈 推荐配置

### 告警规则
```yaml
- alert: HighLatencyP99
  expr: histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m])) > 5
  annotations:
    summary: "P99 延迟超过 5 秒"

- alert: HighErrorRate
  expr: rate(llm_errors_total[1m]) / rate(llm_calls_total[1m]) > 0.05
  annotations:
    summary: "错误率超过 5%"
```

### 长期存储
- 配置 Prometheus 远程存储以保留更长的历史数据
- 建议保留至少 30 天的数据用于分析

### 自定义仪表板
- 根据业务需求创建额外的面板
- 集成业务指标（如用户满意度）
- 创建定期报告

---

## 📚 参考资源

- 📖 [Prometheus 官方文档](https://prometheus.io/docs/)
- 📖 [Grafana 官方文档](https://grafana.com/docs/)
- 📖 [PromQL 查询指南](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- 📖 [Python Prometheus 客户端](https://github.com/prometheus/client_python)

---

## ✅ 验证清单

- [ ] 安装了依赖 (`pip install -r requirements.txt`)
- [ ] 应用服务运行在指定端口 (8000, 8001, 8080)
- [ ] Docker 和 Docker Compose 已安装
- [ ] `docker-compose up -d` 成功启动
- [ ] Prometheus 可以访问：http://localhost:9090
- [ ] Grafana 可以访问：http://localhost:3001
- [ ] 至少生成了一些 LLM 调用来创建指标数据
- [ ] Grafana 仪表板中显示了数据
- [ ] `verify_monitoring.py` 验证通过

---

## 🎉 完成！

你的 LLM 监控系统已完全集成！

### 下一步建议：
1. **生成流量**：发送一些 LLM 调用来填充指标
2. **探索 Grafana**：查看不同的图表和指标
3. **自定义仪表板**：根据需求调整面板
4. **设置告警**：配置业务关键指标的告警规则
5. **定期监控**：建立监控和告警的日常检查流程

---

有问题？查看：
- 📖 `MONITORING_GUIDE.md` - 详细指南
- 📖 `MONITORING_SETUP.md` - 集成说明
- 🔍 `verify_monitoring.py` - 系统验证脚本
