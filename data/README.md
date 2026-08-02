# Release datasets

The public v1 tree contains only the current reviewed datasets:

| Dataset | Role | Cases | Status |
|---|---|---:|---|
| `router_dev_v4.jsonl` | Development and model comparison | 329 | reviewed/frozen |
| `router_regression_v2.jsonl` | Exposed regression coverage | 270 | exposed |
| `router_locked_test_v4.jsonl` | Final acceptance evidence | 90 | exposed/passed |

Every row has a human-verified label. Each dataset is paired with a manifest
containing its SHA-256 digest, lifecycle status, counts, and provenance receipts.

The locked test is intentionally public and therefore exposed. It is retained
as evidence for v1.0.0 and must never be used to recalibrate a changed model,
prompt, or rule set. A changed classifier artifact requires a new private,
frozen locked test.

Model predictions, active rule hits, and shadow reports are never treated as
ground truth. Runtime feedback and review queues are local-only files ignored by
Git.
