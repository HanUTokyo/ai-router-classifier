# AI Router 分类器

[![CI](https://github.com/HanUTokyo/ai-router-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/HanUTokyo/ai-router-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md)

一个由评测驱动的本地 AI Router：对请求分类并按需调度到专用模型，具备经过校准的
规则、显式 fallback 和接近生产工程的质量门禁。

## 为什么做这个项目

- 代码、推理和通用对话任务适合不同类型的模型。
- 所有请求都交给大模型，会增加本地算力成本与延迟。
- 纯关键词或自由 Prompt 路由难以校准、审计和安全变更。

## 我完成了什么

我设计并实现了混合路由链路、OpenAI/Ollama 兼容 API、评测框架、校准 fallback、
规则生命周期、人工审核流程和容器化部署。

### 三项核心证据

- 在一次性的 **90 条冻结 locked test** 上达到 **0.9889 Macro-F1**。样本量较小且
  测试集已经公开，这是明确限制，不代表开放域准确率。
- 同一次验收中 **hard-rule precision 为 1.0**。
- 支持 **OpenAI SSE 和 Ollama NDJSON 真流式输出**；校准后的分类工件不可用时会
  显式返回 degraded 结果。

## 30 秒理解系统

```text
请求
  ↓
经过校准的 Hard Rules ── 或 ── 结构化小模型分类器
  ↓
RouteDecision { route, source, confidence, degraded }
  ↓
code | reason | chat
  ↓ 可选
Ollama 回答模型
```

项目目标不是堆叠架构复杂度。在线路由链路刻意保持简单，主要复杂性放在评测、校准
和规则安全变更上。Planner、DAG、多 Agent、训练闭环、LangGraph 和搜索编排均不属于
v1。

### 工程亮点

- Hard rule + 小模型混合路由
- 对 Prompt、few-shot、schema、生成参数和实际模型做内容寻址
- OpenAI 与 Ollama 兼容 API，并支持真实流式协议
- 冻结数据集与可复现的 baseline/current 对比
- 显式 degraded 模式与校准 fallback
- 影子规则发布、哈希绑定审批与回滚
- Docker 本地部署
- Prometheus 指标与结构化日志
- 推理、健康检查、审核/控制路由在同进程内形成明确代码边界

## 运行 Demo

最短路径只有四条命令：

```bash
git clone https://github.com/HanUTokyo/ai-router-classifier.git
cd ai-router-classifier
docker compose up --build --detach
bash scripts/demo.sh
```

三个只读分类请求的输出示例：

```text
expected=code   actual=code   source=hard_rule               confidence=1.0    latency_ms=0.227    degraded=False
expected=reason actual=reason source=hard_rule               confidence=1.0    latency_ms=0.222    degraded=False
expected=chat   actual=chat   source=hard_rule               confidence=None   latency_ms=0.171    degraded=False
```

延迟因机器而异。Docker 首次启动还会下载数 GB 的 Ollama 镜像，以及约 397MB 的
`qwen2.5:0.5b` 模型。Demo 只绑定 `127.0.0.1:8000`，默认使用有意公开的本地 key
`local-demo-key`；任何共享部署都必须覆盖 `AI_ROUTER_API_KEY`。

也可以直接调用只分类接口：

```bash
curl http://127.0.0.1:8000/route \
  -H 'X-API-Key: local-demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我修复这个 Python 函数"}'
```

## 结果与可信度

下面两项开发集结果都使用相同的 329 条 verified `router_dev_v4` 数据和相同的
`qwen2.5:0.5b` 模型 digest：

| 策略 | Macro-F1 | code recall | reason recall | chat recall | hard-rule coverage | P95 | fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 substring/fixed-confidence | 0.2256 | 0.9369 | 0.0741 | 0.0273 | 0% | 358.669ms | 0% |
| 当前校准 Router | **0.9970** | **1.0000** | **1.0000** | **0.9909** | 31.91% | 362.524ms | 0% |

最初的 substring baseline 虽然 code recall 很高，但几乎无法区分 reason 与 chat。
这次失败推动了混合设计：范围窄且经过验证的规则提供可审计快路径，小模型负责语义
歧义和规则冲突。

locked-test-v4 的唯一一次验收通过全部发布门禁。由于 90 条数据现已公开，它不能再
为未来校准授权。模型、Prompt 或 hard rule 变化后，必须使用一套全新、未曝光且冻结
的测试集。详见[验证与错误切片分析](VALIDATION.md)和
[机器可读结果](docs/results/)。

## 关键设计选择

**为什么不用纯关键词？** “What does API mean?” 包含技术词，但并未请求实现。
substring 规则会过度路由这类请求，也难以划清 reason/chat 边界。

**为什么不用大模型自由路由？** 它会提高延迟和资源消耗，将回答行为与分类耦合，
并让输出稳定性和校准更难验证。

**为什么没有多 Agent？** v1 聚焦可测、可审计、低延迟的分类与可选调度。未来编排
应该放在 `RouteDecision` 之后，而不是侵入分类器核心。

## 本机运行

需要 Python 3.11+ 和 [Ollama](https://ollama.com/)：

```bash
bash start.sh setup
ollama pull qwen2.5:0.5b
bash start.sh doctor
bash start.sh serve
```

默认配置让所有回答角色共用轻量模型。可选的 `config/router.full.yaml` 会把不同路由
映射到专用模型，额外下载量约 19GB。

## API

- `POST /route`：只分类，不调用回答模型。
- `POST /route/feedback`：保存人工反馈，不自动训练或修改规则。
- `POST /v1/chat/completions`：OpenAI Chat Completions 兼容层，支持 SSE。
- `POST /api/chat`：Ollama Chat 兼容层，支持 NDJSON。
- `GET /health/live`、`GET /health/ready`、`GET /metrics`。
- `POST /ask`：已弃用的兼容接口。

OpenAI 请求中的 `model` 是逻辑公共 ID，实际回答模型由服务端选择。Responses API
和通用 `response_format` 尚未实现。完整说明见 [API 文档](docs/API.md)。

## 复现评测

```bash
# 当前策略
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --output artifacts/current-dev.json

# 同一数据与已安装模型 digest 上的旧 baseline
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --legacy \
  --output artifacts/legacy-dev.json
```

报告包含 Macro-F1、分类指标、混淆矩阵、hard-rule coverage/precision、路径延迟、
fallback rate、结构化输出错误、关键切片、Prompt/模型 digest 和错误样本 ID。

不得对公开的 locked-test-v4 运行 `--write-calibration`。

## 仓库结构

```text
router/       分类器、Prompt 资产、API 分区、Gateway 与评测
config/       Demo/完整配置、活动规则、校准收据与 baseline 规则
data/         已审核开发集、回归集与已公开 locked-test 数据
docs/         API、架构、生命周期、错误分析与发布结果
scripts/      人工审核与只读 Demo 工具
tests/        使用 Mock Ollama 的行为和集成契约测试
```

详细设计见[架构与权衡](ARCHITECTURE.md)、[中文代码架构总结](docs/architecture.zh-CN.md)
和[规则生命周期](docs/RULE_LIFECYCLE.md)。

## 已知限制

- 运行期状态使用本地 JSONL，面向单进程，不支持多实例共同写入。
- 当前只有 Ollama 模型后端。
- 审核/控制面与推理面已有明确代码边界，但仍共享同一 FastAPI 进程和安全边界。
- OpenAI token usage 是估算值，并非 tokenizer 精确计数。
- 90 条 locked test 规模较小且已经公开；现有结果不能证明开放域准确率或线上抗漂移能力。

## License

[MIT](LICENSE)
