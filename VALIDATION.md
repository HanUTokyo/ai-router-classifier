# Validation status

本文件区分“实现完成”“工程验证通过”和“上线质量门槛通过”，避免用模型预测或
未复核种子数据冒充正式准确率。

## 已验证

- Python 3.14 虚拟环境可安装，锁定依赖通过 `pip check`。
- 38 项自动化测试覆盖规则边界、冲突、上下文、降级、错误状态码、两种流式协议、
  不完整流的明确终止、反馈文件权限和评测数据闸门。
- 使用本机 Ollama 做过真实链路检查：结构化分类在 `think=false` 下可返回严格
  JSON；OpenAI non-stream/SSE 和 Ollama NDJSON 均能输出可见回答。
- 默认校准文件为 `validated=false`、`hard_rules_enabled=false`。生产服务在
  完成人工金标验收前不会把 provisional hard rule 当作可信短路，也不会让
  未验证 weak-rule 阈值在模型故障时接管分类。

## 尚未宣称通过

- `data/router_cases.jsonl` 的 300 条样本仍是待人工复核素材，不是金标。
- 三个候选模型已下载并完成未复核开发集诊断；仍需在人工复核后的开发集重跑
  五折比较，再用唯一胜者进入锁定测试集验收。
- 未完成连续 7 天、至少 200 次真实分类的观察期。
- Router 默认只绑定 `127.0.0.1`，但检查时发现现有 Ollama 进程监听
  `*:11434`。本次没有擅自重启该外部服务；部署前应将 Ollama 也收口到 loopback
  或使用主机防火墙限制访问。

## 2026-07-26 诊断快照

以下结果来自 `verified=false` 的 210 条开发集，只用于调试评测管线：

| 策略 | Macro-F1 | code recall | reason recall | chat recall | small-model P95 |
|---|---:|---:|---:|---:|---:|
| legacy substring/fixed confidence | 0.4436 | 0.9714 | 0.0571 | 0.5571 | 328.118ms |
| rules 1.2.0 + prompt v5 + qwen2.5:0.5b | 0.9857 | 0.9857 | 1.0000 | 0.9714 | 239.916ms |
| rules 1.2.0 + prompt v5 + qwen3:0.6b | 0.9471 | 0.9714 | 1.0000 | 0.8714 | 267.088ms |
| rules 1.2.0 + prompt v5 + gemma3:1b | 0.9905 | 0.9857 | 1.0000 | 0.9857 | 541.772ms |

新策略在这份种子集上通过总体开发门槛，但上下文指代和提示注入切片仍各有误判；
其中 qwen3 未通过 chat recall 门槛。Gemma3 与 Qwen2.5 的五折 Macro-F1
差值为 0.0048，小于 0.005，因此诊断排序按计划选择延迟更低的
`qwen2.5:0.5b`。样本尚未经人工确认，所以它只是 provisional winner，
`quality_eligible_candidate` 仍为 `null`，也没有写入
`config/calibration.json`。

本次记录的 Ollama digest：qwen2.5:0.5b 为
`a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67`，
qwen3:0.6b 为
`7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`，
gemma3:1b 为
`8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc`。

## 验收命令

```bash
.venv/bin/python scripts/review_dataset.py \
  --dataset data/router_cases.jsonl \
  --reviewer your-name

bash start.sh benchmark --split dev
# 先把 <winner> 写入 config/router.yaml 的 classifier.model
bash start.sh evaluate --split test --write-calibration
```

锁定测试未通过全部质量门槛时，`--write-calibration` 命令会拒绝写入校准文件。
