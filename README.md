# AI Router Classifier

这是一个固定输出 `code`、`reason`、`chat` 的本地分类器。分类器是系统核心：

1. 经过评测验证的 hard rule 可以直接判定。
2. 其余请求交给本地小模型结构化分类。
3. weak rule 只提供证据、冲突记录和模型不可用时的保守回退。
4. 薄 Gateway 根据分类结果调用回答模型。

旧实现完整保存在 `legacy/`，运行时不再依赖它。
当前已验证内容与尚未完成的上线闸门见 [VALIDATION.md](VALIDATION.md)。

## 快速开始

```bash
bash start.sh setup
ollama pull qwen3:0.6b
ollama pull qwen2.5:0.5b
ollama pull gemma3:1b
bash start.sh doctor
bash start.sh serve
```

默认监听 `127.0.0.1:8000`。绑定非本机地址时必须设置
`AI_ROUTER_API_KEY`。

## 分类接口

```bash
curl http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我修复这段 Python 代码"}'
```

返回字段包括 `route`、经验证后才会出现数值的 `confidence`、
`source`、`rule_hits`、分类器/错误类型、版本、降级状态和延迟。未完成
锁定测试校准前，生产服务不会启用 provisional hard rule；评测命令会启用它们，
以便测出 hard-rule precision。

## Gateway 接口

- `POST /v1/chat/completions`：OpenAI Chat Completions，支持 stream。
- `POST /api/chat`：Ollama Chat，支持 NDJSON stream。
- `POST /ask`：旧接口兼容层，响应带弃用头。
- `GET /health/live`、`GET /health/ready`、`GET /metrics`。

上游失败返回 502，超时返回 504，不会再以正常回答返回错误字符串。

## 数据与评测

`data/router_cases.jsonl` 包含 300 条双语种子样本：三类各 100 条，
开发集 210 条、锁定测试集 90 条。初始状态均为
`pending_human_review`，不得作为正式金标冒充质量结果。

逐条人工复核：

```bash
.venv/bin/python scripts/review_dataset.py \
  --dataset data/router_cases.jsonl \
  --reviewer your-name
```

诊断性评测：

```bash
bash start.sh evaluate --split dev --allow-unverified
bash start.sh evaluate --split dev --allow-unverified --legacy
bash start.sh benchmark --split dev --allow-unverified
```

人工复核完成并选出开发集胜者后，先把胜者固定到
`config/router.yaml` 的 `classifier.model`，再运行锁定测试并写入校准：

```bash
bash start.sh evaluate --split test --write-calibration
```

正式闸门：Macro-F1 ≥ 0.92、每类召回率 ≥ 0.88、hard rule 精确率
≥ 0.98、hard-rule 路径 P95 < 10ms、小模型路径 P95 ≤ 2 秒、
结构化输出错误率 < 0.5%。只有完整通过后才允许写入校准文件。

## 隐私与存储

运行日志默认只记录内容哈希、长度、分类来源、版本和延迟。人工反馈与规则冲突
队列保存在 `data/`，权限设置为 `0600`。本地记忆默认关闭，而且永不参与分类。
