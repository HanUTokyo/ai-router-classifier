# Architecture

```text
Client
  ├─ POST /route ───────────────┐
  └─ OpenAI/Ollama adapter      │
                               v
                     RouterClassifier
                ┌──────────────┴──────────────┐
                │ hard rules (short circuit) │
                │ weak evidence              │
                │ small Ollama classifier    │
                └──────────────┬──────────────┘
                               v
                         RouteDecision
                               │
                               v
                     Thin model gateway
                               │
                               v
                         Ollama answer model
```

分类器和 Gateway 在同一个 FastAPI 进程中。分类器不读取记忆、不执行 RAG、不决定
工具细节，也不通过第二个 HTTP 服务调用自身。

## 决策不变量

- hard rule 必须先在锁定测试集达到 0.98 precision，且只有唯一标签命中时
  才短路；未校准规则和不同标签冲突都必须交给小模型。
- 英文技术词使用 token boundary，不允许 `api` 命中 `capitalism`。
- 小模型只能返回结构化三分类结果，非法结果最多重试一次。
- 未通过锁定测试集的校准文件时，`confidence` 为 `null` 且结果标记为未校准。
- 模型不可用时，只有校准文件批准且阈值/边际配置完全匹配时才能走 weak-rule
  回退；否则显式降级到 chat。
- 预测日志不能成为训练真值；只有人工反馈和复核数据可以。
