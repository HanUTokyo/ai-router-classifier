# AI Router Classifier

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[中文说明](README.zh-CN.md)

A local, evaluation-driven AI router that classifies requests as `code`,
`reason`, or `chat`, then optionally forwards them to specialized Ollama models.

The classifier is deliberately small and auditable:

```text
Input
  ↓
Validated hard rules
  ↓
Structured small-model classifier
  ↓
Calibrated confidence / explicit fallback
  ↓
code | reason | chat
```

It includes human-reviewed gold data, a reproducible legacy baseline, latency
and quality gates, error-slice analysis, OpenAI/Ollama-compatible APIs, and a
controlled rule release lifecycle. Planner, DAG, multi-agent, training-loop,
LangGraph, and search orchestration are intentionally outside v1.

## Results

Both development results below were reproduced on the same 329 verified
`router_dev_v4` cases with the same `qwen2.5:0.5b` model digest.

| Policy | Macro-F1 | code recall | reason recall | chat recall | hard-rule coverage | P95 | fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy substring/fixed-confidence | 0.2256 | 0.9369 | 0.0741 | 0.0273 | 0% | 358.669ms | 0% |
| Current calibrated router | **0.9970** | **1.0000** | **1.0000** | **0.9909** | 31.91% | 362.524ms | 0% |

The one-time, 90-case locked-test-v4 acceptance reached **0.9889 Macro-F1**,
1.0 hard-rule precision, and passed every release gate. The test is now exposed
and cannot authorize future recalibration. See [evaluation details](VALIDATION.md)
and [machine-readable results](docs/results/).

Latency is environment-dependent. Path-level P95 for the small-model route was
378.119ms in this run.

## Quick start with Docker

Docker Compose starts Ollama, downloads the 397MB `qwen2.5:0.5b` model into a
persistent volume, and starts the Router:

```bash
docker compose up --build
```

The demo is bound to `127.0.0.1:8000` and uses the intentionally public local
key `local-demo-key`.

```bash
bash scripts/demo.sh
```

Or call the classifier directly:

```bash
curl http://127.0.0.1:8000/route \
  -H 'X-API-Key: local-demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Help me fix this Python function"}'
```

Override `AI_ROUTER_API_KEY` before any shared or remote deployment.

## Native setup

Requirements: Python 3.11+ and [Ollama](https://ollama.com/).

```bash
bash start.sh setup
ollama pull qwen2.5:0.5b
bash start.sh doctor
bash start.sh serve
```

The default profile maps every answer role to the same lightweight model so a
new contributor needs only one download. For the optional specialized gateway:

```bash
ollama pull gemma4:e4b
ollama pull deepseek-r1:8b
ollama pull deepseek-coder:6.7b
AI_ROUTER_CONFIG=config/router.full.yaml bash start.sh serve
```

Those optional answer models require roughly 19GB in total.

## API surface

- `POST /route` — classification only.
- `POST /route/feedback` — verified human feedback; never auto-trains rules.
- `POST /v1/chat/completions` — OpenAI Chat Completions adapter, including SSE.
- `POST /api/chat` — Ollama Chat adapter, including NDJSON streaming.
- `GET /health/live`, `GET /health/ready`, `GET /metrics`.
- `POST /ask` — deprecated compatibility endpoint.

The OpenAI `model` field is a logical public model ID; server-side routing picks
the actual model. The Responses API and general `response_format` are not
implemented. See the complete [API contract](docs/API.md).

## Evaluation

```bash
# Current policy
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --output artifacts/current-dev.json

# Reproducible legacy baseline on the same data
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --legacy \
  --output artifacts/legacy-dev.json

# Candidate model comparison
bash start.sh benchmark \
  --dataset data/router_dev_v4.jsonl \
  --split dev
```

Reports include Macro-F1, per-class metrics, confusion matrix, hard-rule
coverage and precision, path latency, fallback rate, structured-output errors,
critical slices, and misclassified case IDs.

Do not rerun `--write-calibration` on the public locked-test-v4 data. Any model,
prompt, or hard-rule change requires a new private frozen locked test.

## Design highlights

- Hard rules short-circuit only when a matching calibration receipt is valid.
- Conflicting hard rules are delegated to the small classifier.
- Weak rules are evidence, not normal-path decision makers.
- Route-label injection clauses are removed before rules and model inference.
- Conversation context is included only for referential follow-ups.
- Optional memory is answer-time only and never affects classification.
- Runtime failures return explicit degraded sources instead of fake confidence.
- Rule proposals require reviewed evidence, shadow evaluation, immutable hashes,
  human approval, atomic promotion, and rollback receipts.

See [architecture and trade-offs](ARCHITECTURE.md), the
[detailed Chinese architecture](docs/architecture.zh-CN.md), and the
[rule lifecycle](docs/RULE_LIFECYCLE.md).

## Repository layout

```text
router/       Runtime classifier, gateway, adapters, storage, and evaluation
config/       Demo/full profiles, active rules, calibration, baseline rules
data/         Current reviewed development, regression, and exposed test data
docs/         API, architecture, lifecycle, and release results
scripts/      Human-review and read-only demo utilities
tests/        Unit and integration-contract tests with mocked Ollama transport
```

## Known limitations

- Runtime state is local JSONL and is designed for one process, not horizontal
  multi-writer deployment.
- Ollama is the only model backend implementation.
- Prompt/few-shot content and the installed model digest are documented in
  receipts but are not yet fully content-addressed at runtime.
- OpenAI token usage is an estimate, not tokenizer-exact accounting.
- The review/control plane is co-located with the serving process.

## License

[MIT](LICENSE)
