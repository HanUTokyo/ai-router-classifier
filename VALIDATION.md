# Evaluation and release evidence

## Release result

The public v1 classifier is frozen as:

```text
qwen2.5:0.5b
classifier version 1.0.0
prompt classifier-v11
rules 1.15.0
calibration router-locked-test-v4
```

All development comparisons use the same 329 verified `router_dev_v4` cases,
the same model name, and model digest
`a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67`.

| Metric | Legacy policy | Current policy |
|---|---:|---:|
| Macro-F1 | 0.2256 | **0.9970** |
| code recall | 0.9369 | **1.0000** |
| reason recall | 0.0741 | **1.0000** |
| chat recall | 0.0273 | **0.9909** |
| hard-rule coverage | 0% | 31.91% |
| hard-rule precision | n/a | 1.0000 |
| overall P95 | 358.669ms | 362.524ms |
| small-model P95 | 358.669ms | 378.119ms |
| fallback rate | 0% | 0% |
| structured-output error rate | 0% | 0% |
| release gate | failed | **passed** |

The current run had one error (`r2-rc-chat-002`, classified as `code`). The
reported latency is environment-dependent; compare policies only on the same
machine and model state.

Machine-readable reports:

- [current dev-v4](docs/results/current-dev-v4.json)
- [legacy dev-v4](docs/results/legacy-dev-v4.json)
- [locked-test-v4 release summary](docs/results/locked-test-v4-summary.json)

## Locked-test acceptance

The frozen candidate was evaluated exactly once on 90 new, verified cases.

| Metric | Result | Gate |
|---|---:|---:|
| Macro-F1 | 0.9889 | >= 0.92 |
| code recall | 1.0000 | >= 0.88 |
| reason recall | 0.9667 | >= 0.88 |
| chat recall | 1.0000 | >= 0.88 |
| hard-rule precision | 1.0000 | >= 0.98 |
| hard-rule P95 | 1.456ms | < 10ms |
| small-model P95 | 1075.672ms | <= 2000ms |
| format error rate | 0 | < 0.5% |

Prompt injection, rule conflict, technical/nontechnical language, and
implementation-after-analysis slices were all 1.0. Context reference accuracy
was 0.9167. The only error was `v4-reason-022` (`reason` -> `chat`).

The test manifest is now `exposed_passed`. Its data remains public evidence but
cannot authorize another calibration. Any classifier artifact change requires a
new private locked test.

## Metric definitions

- `fallback.rate` counts only `weak_rule_fallback` and `default_fallback`.
- `degraded_rate` is broader: it also identifies uncalibrated/provisional paths.
- Hard-rule precision is measured only on unique hard-rule decisions.
- Small-model latency includes every non-hard source for release-gate purposes.
- Critical slice accuracy is computed from human-authored dataset tags.

## Reproduction

Start Ollama with `qwen2.5:0.5b`, then run:

```bash
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --output artifacts/current-dev-v4.json

bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --legacy \
  --output artifacts/legacy-dev-v4.json
```

Latency varies by hardware and cold-start state. Accuracy should be compared
only when the model digest, dataset hash, prompt version, and rule digest match.
