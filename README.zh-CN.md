# AI Router 分类器

[English](README.md)

这是一个由评测驱动的本地 AI Router：先把请求分类为 `code`、`reason` 或
`chat`，再按需转发给不同的 Ollama 回答模型。

核心链路保持小而可审计：

```text
输入
  ↓
经过验证的 Hard Rules
  ↓
结构化小模型分类器
  ↓
校准置信度 / 显式降级
  ↓
code | reason | chat
```

项目附带人工金标数据、可复现 legacy baseline、质量与延迟门禁、错误切片分析、
OpenAI/Ollama 兼容 API 和受控规则发布流程。Planner、DAG、多 Agent、定期训练、
LangGraph 和搜索编排明确不属于 v1。

## 结果

下表的两项开发集结果使用相同的 329 条 verified dev-v4 数据和相同的
`qwen2.5:0.5b` 模型 digest 重新运行：

| 策略 | Macro-F1 | code recall | reason recall | chat recall | hard-rule coverage | P95 | fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 substring/fixed-confidence | 0.2256 | 0.9369 | 0.0741 | 0.0273 | 0% | 358.669ms | 0% |
| 当前校准 Router | **0.9970** | **1.0000** | **1.0000** | **0.9909** | 31.91% | 362.524ms | 0% |

90 条 locked-test-v4 的唯一一次验收达到 **0.9889 Macro-F1**、hard-rule
precision 1.0，并通过全部发布门禁。该测试集已经曝光，不能再为未来变更授权校准。
详见[评测说明](VALIDATION.md)与[机器可读结果](docs/results/)。

延迟受机器环境影响。本次运行的 small-model 路径 P95 为 378.119ms。

## Docker 快速开始

Compose 会启动 Ollama、把 397MB 的 `qwen2.5:0.5b` 下载到持久卷，并启动 Router：

```bash
docker compose up --build
```

Demo 只绑定 `127.0.0.1:8000`，使用公开的本地演示 key `local-demo-key`：

```bash
bash scripts/demo.sh
```

也可以直接调用分类接口：

```bash
curl http://127.0.0.1:8000/route \
  -H 'X-API-Key: local-demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我修复这个 Python 函数"}'
```

任何共享或远程部署都必须覆盖 `AI_ROUTER_API_KEY`。

## 本机运行

需要 Python 3.11+ 和 [Ollama](https://ollama.com/)：

```bash
bash start.sh setup
ollama pull qwen2.5:0.5b
bash start.sh doctor
bash start.sh serve
```

默认配置让所有回答角色共用轻量模型，只需一次下载。专业多模型 Gateway 是可选项：

```bash
ollama pull gemma4:e4b
ollama pull deepseek-r1:8b
ollama pull deepseek-coder:6.7b
AI_ROUTER_CONFIG=config/router.full.yaml bash start.sh serve
```

三个可选回答模型合计约需 19GB。

## API

- `POST /route`：只分类，不调用回答模型。
- `POST /route/feedback`：保存人工反馈，不自动训练或修改规则。
- `POST /v1/chat/completions`：OpenAI Chat Completions 兼容层，支持 SSE。
- `POST /api/chat`：Ollama Chat 兼容层，支持 NDJSON。
- `GET /health/live`、`GET /health/ready`、`GET /metrics`。
- `POST /ask`：已弃用的兼容接口。

OpenAI 请求的 `model` 是逻辑公共 ID，实际回答模型由服务端 Router 选择。
Responses API 和通用 `response_format` 尚未实现。完整说明见 [API 文档](docs/API.md)。

## 评测

```bash
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --output artifacts/current-dev.json

bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --legacy \
  --output artifacts/legacy-dev.json

bash start.sh benchmark \
  --dataset data/router_dev_v4.jsonl \
  --split dev
```

报告包含 Macro-F1、分类指标、混淆矩阵、hard-rule coverage/precision、路径延迟、
fallback rate、结构化输出错误、关键切片和错误样本 ID。

不得对公开的 locked-test-v4 再次运行 `--write-calibration`。模型、Prompt 或 hard
rule 发生变化后，必须使用一套全新、未曝光且已冻结的锁定测试集。

## 关键设计

- Hard rule 只有在校准收据完全匹配时才能短路。
- 不同标签的 hard rule 冲突交给小模型。
- Weak rule 是证据，不是正常路径的直接决策者。
- 规则和模型判断前会移除操纵路由标签的子句。
- 只有指代型追问才携带最近上下文，记忆永不参与分类。
- 失败会返回显式 degraded 来源，不会伪造置信度。
- 规则上线需要人工证据、影子评测、哈希收据、人工批准和可回滚快照。

详见[架构与权衡](ARCHITECTURE.md)、[中文代码架构总结](docs/architecture.zh-CN.md)
和[规则生命周期](docs/RULE_LIFECYCLE.md)。

## 已知限制

- JSONL 状态面向本地单进程，不支持多实例共同写入。
- 当前只有 Ollama 模型后端。
- Prompt/few-shot 内容和模型 digest 尚未在运行时完全内容寻址。
- OpenAI token usage 是估算值。
- 审核控制面与在线 API 仍在同一进程。

## License

[MIT](LICENSE)
