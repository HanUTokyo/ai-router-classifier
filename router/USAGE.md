# Usage Guide

## 1. Architecture Split
- `main.py`: the only gateway entrypoint, responsible for question answering.
- `router/`: dispatcher only, responsible for route classification (`code/reason/chat`).

## 2. Run Single Gateway Entrypoint
Run from workspace root:

```bash
export OLLAMA_URL="http://192.168.1.66:11434/api/chat"
export ROUTER_ROUTE_URL="http://localhost:8001/route"
export ROUTER_MODEL="qwen2.5:0.2b"
export CHAT_MODEL="gemma4:e4b"
export REASON_MODEL="deepseek-r1:7b"
export CODE_MODEL="deepseek-coder:6.7b"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Ask (gateway does: route -> select model -> answer):

```bash
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个任务管理系统"}'
```

Health check:

```bash
curl http://localhost:8000/health
```

Optional: run dispatcher only service for diagnostics:

```bash
uvicorn router.app:app --host 0.0.0.0 --port 8001
curl http://localhost:8001/route \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个任务管理系统"}'
```

## 3. Generate Evolution Artifacts (Analyzer v3)
Default run (reads logs/router.log):

```bash
python3 analyzer.py
```

Custom paths:

```bash
python3 analyzer.py \
  --log-path logs/router.log \
  --rules-path rules.yaml \
  --output-dir . \
  --top-k 5 \
  --min-freq 3
```

Generated files:
- rule_missing.jsonl
- tfidf_scores.json
- rules_patch.txt
- rule_candidates.yaml
- training_data.jsonl

## 4. Review Patch Proposal
Open rules_patch.txt and decide approved keywords per label.

Example rules_patch.txt section:

```text
[code]
+ - spring
+ - 系统
```

## 5. Apply Approved Keywords (Semi-auto)
Option A: Inline approval

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approve "code:spring,系统"
```

Option B: Approval file

approval.yaml:

```yaml
code:
  - spring
  - 系统
reason:
  - 归纳
```

Apply:

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approval-file approval.yaml
```

## 6. Dry Run (No Write)
Preview without modifying rules.yaml:

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approve "code:spring" \
  --dry-run
```

## 7. Backup Strategy
When not using --dry-run, applier will:
1) create backup file: rules.YYYYMMDDHHMMSS.yaml
2) atomically write updated rules.yaml

## 8. Recommended Workflow
1) run router and accumulate logs
2) run analyzer.py
3) review rules_patch.txt
4) apply approved keywords via apply_rules_patch.py
5) re-run validation traffic and observe logs/router.log

---

# 使用说明（中文翻译）

## 1. 架构拆分
- `main.py`：唯一网关入口，负责最终问答。
- `router/`：仅分发器，负责输出 `code/reason/chat`。

## 2. 启动单一网关入口
在项目根目录启动：

```bash
export OLLAMA_URL="http://192.168.1.66:11434/api/chat"
export ROUTER_ROUTE_URL="http://localhost:8001/route"
export ROUTER_MODEL="qwen2.5:0.2b"
export CHAT_MODEL="gemma4:e4b"
export REASON_MODEL="deepseek-r1:7b"
export CODE_MODEL="deepseek-coder:6.7b"
uvicorn main:app --host 0.0.0.0 --port 8000
```

问答接口（网关流程：先分发，再选模型回答）：

```bash
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个任务管理系统"}'
```

健康检查：

```bash
curl http://localhost:8000/health
```

可选：单独启动分发器服务用于诊断：

```bash
uvicorn router.app:app --host 0.0.0.0 --port 8001
curl http://localhost:8001/route \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个任务管理系统"}'
```

## 3. 生成进化产物（Analyzer v3）
默认运行（读取 logs/router.log）：

```bash
python3 analyzer.py
```

自定义路径：

```bash
python3 analyzer.py \
  --log-path logs/router.log \
  --rules-path rules.yaml \
  --output-dir . \
  --top-k 5 \
  --min-freq 3
```

生成文件：
- rule_missing.jsonl
- tfidf_scores.json
- rules_patch.txt
- rule_candidates.yaml
- training_data.jsonl

## 4. 审阅 Patch 建议
打开 rules_patch.txt，按标签决定批准的关键词。

rules_patch.txt 示例片段：

```text
[code]
+ - spring
+ - 系统
```

## 5. 应用已批准关键词（半自动）
方案 A：命令行内联审批

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approve "code:spring,系统"
```

方案 B：审批文件

approval.yaml：

```yaml
code:
  - spring
  - 系统
reason:
  - 归纳
```

执行应用：

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approval-file approval.yaml
```

## 6. Dry Run（不写入）
仅预览，不修改 rules.yaml：

```bash
python3 apply_rules_patch.py \
  --rules-path rules.yaml \
  --patch-path rules_patch.txt \
  --approve "code:spring" \
  --dry-run
```

## 7. 备份策略
在未使用 --dry-run 的情况下，应用器将：
1) 创建备份文件：rules.YYYYMMDDHHMMSS.yaml
2) 以原子方式写入更新后的 rules.yaml

## 8. 推荐工作流
1) 运行路由服务并积累日志
2) 运行 analyzer.py
3) 审阅 rules_patch.txt
4) 通过 apply_rules_patch.py 应用已批准关键词
5) 重新跑验证流量并观察 logs/router.log
