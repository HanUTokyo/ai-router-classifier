# Router System Architecture

## 1. Goal
This system routes user input into three labels:
- code
- reason
- chat

The router combines rule-based matching and model-based classification, then fuses both results into a final decision.

## 2. Components
- app.py
Purpose: FastAPI entrypoint, HTTP middleware, trace_id injection, request validation.

- router_service.py
Purpose: Main orchestration layer.
Responsibilities:
1) call RuleEngine
2) call model router
3) fuse decisions
4) emit final router logs

- rule_engine.py
Purpose: Rule matching engine based on rules.yaml.
Responsibilities:
1) load rules with validation
2) compute keyword hits and score
3) produce rule-side confidence and matched keywords

- router_model.py
Purpose: LLM classifier wrapper.
Responsibilities:
1) prompt/model invocation
2) normalize and map model output
3) fallback to rule engine if model fails

- logger.py
Purpose: JSON logging to logs/router.log.

- analyzer.py (v3)
Purpose: Offline evolution pipeline.
Responsibilities:
1) detect rule missing samples
2) compute TF-IDF keyword scores
3) generate patch proposal, not auto-apply

- apply_rules_patch.py
Purpose: Semi-automatic patch applier with human approval gate.
Responsibilities:
1) read rules_patch.txt
2) apply only approved keywords
3) backup current rules.yaml
4) write updated rules.yaml atomically

## 3. Data Flow
Runtime flow:
1) Client -> /route
2) app.py validates request and sets trace_id
3) router_service.py runs rule + model
4) fuse_decision picks final label/confidence/source
5) logger.py writes events to logs/router.log

Evolution flow:
1) logs/router.log -> analyzer.py
2) analyzer.py outputs:
- rule_missing.jsonl
- tfidf_scores.json
- rules_patch.txt
- rule_candidates.yaml
- training_data.jsonl
3) Human reviews rules_patch.txt
4) apply_rules_patch.py applies approved keywords
5) rules.yaml is updated with timestamp backup

## 4. Guardrails
- rules.yaml is the single source of truth for online rule config.
- analyzer.py only proposes patch; it never overwrites rules.yaml.
- apply_rules_patch.py requires explicit approval input.
- rules.yaml write is atomic and always backed up before overwrite.

## 5. Output Artifacts
- logs/router.log: runtime event stream
- rule_missing.jsonl: uncovered rule samples
- tfidf_scores.json: keyword score distribution
- rules_patch.txt: proposal diff for human review
- rule_candidates.yaml: coarse candidate set
- training_data.jsonl: high-value training samples

---

# 路由系统架构（中文翻译）

## 1. 目标
该系统将用户输入路由到三个标签之一：
- code
- reason
- chat

路由器结合规则匹配与模型分类，再将两者融合为最终决策。

## 2. 组件
- app.py
作用：FastAPI 入口，负责 HTTP 中间件、trace_id 注入、请求校验。

- router_service.py
作用：主编排层。
职责：
1) 调用 RuleEngine
2) 调用模型路由
3) 融合决策
4) 输出最终路由日志

- rule_engine.py
作用：基于 rules.yaml 的规则匹配引擎。
职责：
1) 加载并校验规则
2) 计算关键词命中与分数
3) 输出规则侧置信度与命中关键词

- router_model.py
作用：LLM 分类器封装层。
职责：
1) 提示词与模型调用
2) 规范化并映射模型输出
3) 模型失败时回退到规则引擎

- logger.py
作用：将 JSON 日志写入 logs/router.log。

- analyzer.py (v3)
作用：离线进化流水线。
职责：
1) 识别规则缺失样本
2) 计算 TF-IDF 关键词分数
3) 生成 patch 建议（不自动应用）

- apply_rules_patch.py
作用：带人工审批门控的半自动 patch 应用器。
职责：
1) 读取 rules_patch.txt
2) 仅应用已批准关键词
3) 备份当前 rules.yaml
4) 原子写入更新后的 rules.yaml

## 3. 数据流
在线运行流：
1) Client -> /route
2) app.py 校验请求并设置 trace_id
3) router_service.py 执行 rule + model
4) fuse_decision 产出最终 label/confidence/source
5) logger.py 将事件写入 logs/router.log

离线进化流：
1) logs/router.log -> analyzer.py
2) analyzer.py 输出：
- rule_missing.jsonl
- tfidf_scores.json
- rules_patch.txt
- rule_candidates.yaml
- training_data.jsonl
3) 人工审阅 rules_patch.txt
4) apply_rules_patch.py 应用已批准关键词
5) rules.yaml 以带时间戳备份后完成更新

## 4. 安全与约束
- rules.yaml 是在线规则配置的唯一事实源。
- analyzer.py 只生成建议 patch，不会覆盖 rules.yaml。
- apply_rules_patch.py 需要显式审批输入。
- rules.yaml 写入采用原子方式，并在覆盖前强制备份。

## 5. 输出产物
- logs/router.log：运行时事件流
- rule_missing.jsonl：规则未覆盖样本
- tfidf_scores.json：关键词评分分布
- rules_patch.txt：人工审阅用提案 diff
- rule_candidates.yaml：粗粒度候选集合
- training_data.jsonl：高价值训练样本
