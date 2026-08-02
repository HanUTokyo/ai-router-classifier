# Rule lifecycle

Active rules are never changed directly from feedback or generated output.

```text
verified human evidence
        ↓
candidate proposal
        ↓
dev-v4 + regression-v2 shadow evaluation
        ↓
quality gates + unchanged SHA-256 receipts + human approval
        ↓
atomic active-rule promotion and history snapshot
        ↓
new private locked test before hard-rule short circuits are re-enabled
```

## Propose

Prepare one `RuleSpec` YAML and create a candidate:

```bash
bash start.sh rules propose \
  --action add \
  --rule-id code-explicit-tool-call \
  --target-version 1.16.0 \
  --reason "Repeated verified implementation requests" \
  --created-by reviewer \
  --evidence-ref feedback-case-id \
  --rule-spec /path/to/rule.yaml \
  --output config/rule_candidates/code-explicit-tool-call-v1.16.0.yaml
```

Supported actions are `add`, `update`, `disable`, `demote`, and `promote`.
Every change requires at least one human evidence reference. `update` cannot
silently change hard/weak kind or enabled state.

## Shadow evaluate

```bash
bash start.sh rules evaluate \
  --candidate config/rule_candidates/code-explicit-tool-call-v1.16.0.yaml \
  --dev-dataset data/router_dev_v4.jsonl \
  --regression-dataset data/router_regression_v2.jsonl
```

The report binds the active rules, proposal, candidate rule file, and datasets
with SHA-256 receipts. A hard rule must have at least 20 verified matches and at
least 0.98 precision. The candidate may not introduce a hard error, hard-label
conflict, or regression error.

When configured, the service can load the candidate as a shadow engine. It
records active/shadow differences but never changes the real route.

## Promote

```bash
bash start.sh rules promote \
  --candidate config/rule_candidates/code-explicit-tool-call-v1.16.0.yaml \
  --report config/rule_shadow.json \
  --approved-by reviewer
```

Promotion revalidates every receipt, snapshots the old rule file, archives the
candidate/report, atomically replaces active rules, and writes an approval
receipt. A rule-version change invalidates the current calibration, so new hard
rules do not gain production short-circuit authority until a new locked test is
accepted.

## Roll back

```bash
bash start.sh rules rollback \
  --snapshot config/rule_history/rules-v1.15.0-<digest>.yaml \
  --approved-by reviewer \
  --reason "New counterexample in reviewed traffic"
```

The current rules are snapshotted before rollback. Candidate and history
directories are intentionally ignored in the public repository because local
receipts may contain reviewer identities and absolute paths.

The original Chinese operating guide is available at
[rule-lifecycle.zh-CN.md](rule-lifecycle.zh-CN.md).
