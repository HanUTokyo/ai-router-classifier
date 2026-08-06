# AI Router Classifier

[![CI](https://github.com/HanUTokyo/ai-router-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/HanUTokyo/ai-router-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[中文说明](README.zh-CN.md)

An evaluation-driven local AI router that classifies requests and optionally
dispatches them to specialized models, with calibrated rules, explicit
fallbacks, and production-style quality gates.

## Why this project exists

- Code, reasoning, and general-chat workloads benefit from different models.
- Sending every request to a large model increases local compute cost and
  latency.
- Keyword-only and free-form prompt routers are difficult to calibrate, audit,
  and change safely.

## What I built

I designed and implemented the hybrid routing pipeline, OpenAI/Ollama-compatible
APIs, evaluation framework, calibrated fallback behavior, rule lifecycle,
review workflow, and containerized deployment.

### Evidence at a glance

- **0.9889 Macro-F1** on a one-time **90-case frozen locked test**. The small
  sample and now-exposed test set are stated limitations, not an open-domain
  accuracy claim.
- **1.0 hard-rule precision** on that acceptance run.
- **OpenAI-compatible SSE and Ollama NDJSON streaming**, with explicit degraded
  results when the calibrated classifier artifact is unavailable.

## System in 30 seconds

```text
Request
  ↓
Calibrated hard rules ── or ── structured small-model classifier
  ↓
RouteDecision { route, source, confidence, degraded }
  ↓
code | reason | chat
  ↓ optional
Ollama answer model
```

The goal was not to maximize architectural complexity. The online routing path
stays small; most complexity lives in evaluation, calibration, and safe rule
changes. Planner, DAG, multi-agent, training-loop, LangGraph, and search
orchestration are intentionally outside v1.

### Engineering highlights

- Hybrid rule + small-model routing
- Content-addressed prompt, few-shot, schema, generation settings, and model
  identity
- OpenAI and Ollama compatible APIs with real streaming
- Frozen datasets and reproducible baseline/current evaluation
- Explicit degraded-mode behavior and calibrated fallbacks
- Shadow rule rollout, hash-bound approval, and rollback
- Dockerized local deployment
- Prometheus metrics and structured logging
- Same-process code boundaries for inference, health, and review/control routes

## Run the demo

The shortest path is four commands:

```bash
git clone https://github.com/HanUTokyo/ai-router-classifier.git
cd ai-router-classifier
docker compose up --build --detach
bash scripts/demo.sh
```

Example output from the three read-only classification requests:

```text
expected=code   actual=code   source=hard_rule               confidence=1.0    latency_ms=0.227    degraded=False
expected=reason actual=reason source=hard_rule               confidence=1.0    latency_ms=0.222    degraded=False
expected=chat   actual=chat   source=hard_rule               confidence=None   latency_ms=0.171    degraded=False
```

Latency varies by machine. On first startup Docker also downloads the
multi-gigabyte Ollama image and the approximately 397MB `qwen2.5:0.5b` model.
The demo binds only to `127.0.0.1:8000` and uses the intentionally public local
key `local-demo-key`; override `AI_ROUTER_API_KEY` before any shared deployment.

To call the classification-only endpoint directly:

```bash
curl http://127.0.0.1:8000/route \
  -H 'X-API-Key: local-demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Help me fix this Python function"}'
```

## Results and credibility

Both development results below were reproduced on the same 329 verified
`router_dev_v4` cases with the same `qwen2.5:0.5b` model digest.

| Policy | Macro-F1 | code recall | reason recall | chat recall | hard-rule coverage | P95 | fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy substring/fixed-confidence | 0.2256 | 0.9369 | 0.0741 | 0.0273 | 0% | 358.669ms | 0% |
| Current calibrated router | **0.9970** | **1.0000** | **1.0000** | **0.9909** | 31.91% | 362.524ms | 0% |

The first substring baseline achieved high code recall but almost failed to
distinguish reasoning and chat. That failure motivated the hybrid design:
narrow verified rules provide auditable fast paths, while the small model
handles semantic ambiguity and rule conflicts.

The one-time locked-test-v4 acceptance passed every release gate. Because its
90 cases are now public, it cannot authorize future recalibration. Any model,
prompt, or hard-rule change requires a new private frozen test set. See the
[validation and error-slice analysis](VALIDATION.md) and
[machine-readable results](docs/results/).

## Design decisions

**Why not keywords alone?** A phrase such as “What does API mean?” contains a
technical term but does not request implementation. Substring rules over-route
these cases and perform poorly on reasoning/chat boundaries.

**Why not let a large model route freely?** It raises latency and resource use,
couples answer behavior to routing, and makes output stability and calibration
harder to verify.

**Why no multi-agent framework?** The v1 scope is measurable, auditable,
low-latency classification and optional dispatch. Future orchestration belongs
after `RouteDecision`, not inside the classifier.

## Native setup

Requirements: Python 3.11+ and [Ollama](https://ollama.com/).

```bash
bash start.sh setup
ollama pull qwen2.5:0.5b
bash start.sh doctor
bash start.sh serve
```

The default profile maps all answer roles to the same lightweight model. The
optional `config/router.full.yaml` maps routes to specialized models and needs
roughly 19GB of additional downloads.

## API surface

- `POST /route` — classification only.
- `POST /route/feedback` — verified human feedback; never auto-trains rules.
- `POST /v1/chat/completions` — OpenAI Chat Completions adapter, including SSE.
- `POST /api/chat` — Ollama Chat adapter, including NDJSON streaming.
- `GET /health/live`, `GET /health/ready`, `GET /metrics`.
- `POST /ask` — deprecated compatibility endpoint.

The OpenAI `model` field is a logical public ID; server-side routing chooses the
actual answer model. The Responses API and general `response_format` are not
implemented. See the complete [API contract](docs/API.md).

## Reproduce the evaluation

```bash
# Current policy
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --output artifacts/current-dev.json

# Legacy baseline on the same data and installed model digest
bash start.sh evaluate \
  --dataset data/router_dev_v4.jsonl \
  --split dev \
  --legacy \
  --output artifacts/legacy-dev.json
```

Reports include Macro-F1, per-class metrics, confusion matrix, hard-rule
coverage and precision, path latency, fallback rate, structured-output errors,
critical slices, prompt/model digests, and misclassified case IDs.

Do not run `--write-calibration` against the public locked-test-v4 data.

## Repository layout

```text
router/       Classifier, prompt assets, API planes, gateway, and evaluation
config/       Demo/full profiles, active rules, calibration, baseline rules
data/         Reviewed development, regression, and exposed locked-test data
docs/         API, architecture, lifecycle, analysis, and release results
scripts/      Human-review and read-only demo utilities
tests/        Behavior and integration-contract tests with mocked Ollama
```

Detailed design: [Architecture and trade-offs](ARCHITECTURE.md),
[Chinese code architecture](docs/architecture.zh-CN.md), and
[rule lifecycle](docs/RULE_LIFECYCLE.md).

## Known limitations

- Runtime state uses local JSONL and targets one process, not horizontal
  multi-writer deployment.
- Ollama is the only model backend implementation.
- The review/control and inference planes have explicit code boundaries but
  still share one FastAPI process and security boundary.
- OpenAI token usage is estimated rather than tokenizer-exact.
- The 90-case locked test is small and exposed; reported scores do not establish
  open-domain accuracy or production drift resistance.

## License

[MIT](LICENSE)
