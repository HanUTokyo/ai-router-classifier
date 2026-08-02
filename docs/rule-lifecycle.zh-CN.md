# Rule 生命周期

Hard/weak rule 与分类模型一样，必须经过候选、影子评测、人工批准、版本发布和回滚。
任何反馈或脚本都不能直接修改 `config/rules.yaml`。

## 状态流

```text
人工反馈
   │
   v
candidate proposal
   │  已审核开发集 + 已曝光回归集
   v
shadow_evaluated
   │  离线门禁通过；线上只记录差异
   v
promoted rules version
   │  新锁定测试集重新校准
   v
hard short-circuit enabled

任一阶段 ──> disable / demote / rollback
```

`/review` 仍然只有“正确 / 错误”两个按钮，不承担规则编辑。`/route/feedback`
会保存人工标签，并自动附加当前规则命中、规则版本、模型版本和以下信号：

- `hard_rule_positive`：当前唯一 hard rule 与人工标签一致。
- `hard_rule_negative`：当前唯一 hard rule 与人工标签冲突。
- `candidate_rule_gap`：现有分类错误，且没有唯一 hard rule 接管。
- `human_label_only`：只有人工标签，没有直接的规则升级信号。

这些信号只进入证据队列，不会自动产生或启用规则。

## 1. 建立候选

新增或修改规则时，先准备一个只包含单条规则的 YAML：

```yaml
id: code-explicit-tool-call
label: code
kind: hard
pattern_type: regex
pattern: '(?i)\b(write|implement)\b.{0,40}\b(tool call|function calling)\b'
exclusions: []
weight: 1.0
enabled: true
```

然后生成候选变更：

```bash
bash start.sh rules propose \
  --action add \
  --rule-id code-explicit-tool-call \
  --target-version 1.4.0 \
  --reason "人工反馈中重复出现明确的工具调用实现请求" \
  --created-by Kai \
  --evidence-ref feedback-request-id \
  --rule-spec /path/to/rule-spec.yaml \
  --output config/rule_candidates/code-explicit-tool-call-v1.4.0.yaml
```

支持的动作：

- `add`：新增规则，必须提供 `--rule-spec`。
- `update`：修改规则或排除模式，必须提供 `--rule-spec`。
- `disable`：停用规则，必须提供至少一个 `--evidence-ref`。
- `demote`：hard 降为 weak，必须提供至少一个 `--evidence-ref`。
- `promote`：weak 升为 hard。这里的动作名指规则类型升级，不是发布命令。

每项变更都必须至少引用一条人工审核证据。候选版本必须高于当前版本；阈值可以
提高，但不能低于系统固定门槛。`update`
也不能暗中改变 `kind` 或 `enabled`，这两类变化必须使用专门动作并留下证据。

## 2. 影子评测

```bash
bash start.sh rules evaluate \
  --candidate config/rule_candidates/code-explicit-tool-call-v1.4.0.yaml
```

默认在以下两套已审核数据上运行：

- `data/router_dev_v4.jsonl`
- `data/router_regression_v2.jsonl`

报告默认写入 `config/rule_shadow.json`。它绑定以下 SHA-256 收据：

- 当前活动规则文件
- 候选变更文件
- 完整候选规则集
- 每个评测数据集

门禁要求：

- 候选必须实际改变至少一个规则证据命中。
- 总体 hard-rule precision 不低于 `0.98`。
- 不得新增 hard-rule 错误。
- 不得新增不同标签的 hard-rule 冲突。
- 已曝光回归集不得新增 hard-rule 错误。
- 新增、修改或升级为 hard 的目标规则 precision 不低于 `0.98`。
- 目标 hard rule 至少命中 20 条已审核样本。
- 所有新增、修改、停用、升降级都必须具备人工证据引用。

weak rule 只改变交给小模型的结构化证据，不取得短路权；它仍需确保没有引入
新的 hard 错误或冲突。若未来要把 weak 升级为 hard，必须重新满足 hard rule
的精确率与最小支持门槛。

服务重启后会读取有效的 `config/rule_shadow.json`。影子规则只把与当前规则不同的
命中写入 `data/review_queue.jsonl`，绝不影响实际路由结果。

## 3. 人工批准与发布

只有门禁全部通过、且所有文件哈希仍与评测时一致，才能发布：

```bash
bash start.sh rules promote \
  --candidate config/rule_candidates/code-explicit-tool-call-v1.4.0.yaml \
  --report config/rule_shadow.json \
  --approved-by Kai
```

发布过程会：

1. 将旧 `rules.yaml` 保存到 `config/rule_history/`。
2. 冻结候选文件和完整影子报告。
3. 原子替换活动规则文件。
4. 写入包含批准人、时间和全部哈希的发布收据。
5. 清除已经完成使命的活动影子报告；冻结副本仍保留在历史目录。

规则版本变化会自动使旧校准失效。因此新版本即使已经发布，也不会立即获得
hard-rule 短路权；必须用一套未曝光的新锁定测试集重新校准。

## 4. 回滚

```bash
bash start.sh rules rollback \
  --snapshot config/rule_history/rules-v1.3.0-xxxxxxxxxxxx.yaml \
  --approved-by Kai \
  --reason "线上影子反馈出现新的反例"
```

回滚前的当前版本也会自动生成快照，并写入独立的回滚收据。

## 数据隔离

- 开发集和已曝光回归集用于反复升级规则。
- 当前锁定测试集只能用于最终验收，不能用来反复调整规则。
- 一旦根据锁定测试结果修改了规则，该测试集立即降级为回归集；下一次验收必须
  新建锁定测试集。
- 模型预测、规则预测和影子结果都不能直接成为真值。
