# Architecture and trade-offs

## System shape

```text
Client
  ├─ POST /route ───────────────────────────────┐
  ├─ OpenAI Chat Completions adapter ──────────┤
  └─ Ollama Chat adapter ──────────────────────┤
                                                v
                                      RouterClassifier
                              ┌─────────────────┼─────────────────┐
                              │ calibrated hard rules             │
                              │ weak evidence + conflict metadata │
                              │ structured local small model       │
                              └─────────────────┬─────────────────┘
                                                v
                                         RouteDecision
                                                │
                                                v
                                           ThinGateway
                                                │
                                                v
                                      Ollama answer model
```

The classifier, gateway, and local review UI share one FastAPI process. Their
code boundaries are explicit: `router/api/inference.py` owns public inference
and compatibility routes, `router/api/control.py` owns feedback/review routes,
`router/api/health.py` owns readiness and metrics, and `router/app.py` only
assembles dependencies, lifespan, middleware, and routers. `RuntimeServices`
passes dependencies into router factories instead of relying on module globals.

Ollama is the only remote runtime dependency. The classifier does not perform
retrieval, call tools, read memory, or execute tasks. Future orchestration
belongs after `RouteDecision`, not inside the classification core.

## Online decision flow

1. Remove clauses whose only purpose is forcing `code`, `reason`, or `chat`.
2. Evaluate active rules and compare an optional shadow rule set.
3. Short-circuit only when exactly one hard label matches and calibration
   authorizes hard rules.
4. Otherwise call the configured small model with a strict JSON schema and at
   most one retry.
5. On model failure, use weak-rule fallback only when the accepted calibration
   enables the exact configured threshold and margin; otherwise return an
   explicit degraded `chat` fallback.
6. The gateway maps the decision to an answer model, with an optional tool-model
   override, and forwards either a normal or streaming Ollama request.

`confidence` is empirical source/label precision from the accepted locked test.
It is not a model logit or a request-specific probability.

## Content-addressed classifier identity

Prompt assets live outside the orchestration code in `router/prompts/`.
`router/prompting.py` strictly validates their schema at startup and builds the
same ordered message payload used for classification. A canonical SHA-256 digest
covers the system prompt, three ordered few-shot groups, structured-output JSON
schema, retry instruction, and generation options. YAML formatting changes do
not affect the digest; semantic content or order changes do.

The calibration receipt binds that prompt digest to the active rule digest and
the installed Ollama model digest. Startup and evaluation query the installed
model identity before calibration can activate. In strict mode, a missing or
mismatched model prevents startup. In non-strict mode, the service remains
available but hard-rule short circuits stay disabled and decisions explicitly
report degraded behavior.

## Control and data planes

The online data plane reads immutable configuration, rules, and calibration.
The local control plane provides:

```text
candidate data -> human review -> verified dev/regression/locked datasets
                                      │
                                      ├─ model benchmark and release gates
                                      └─ rule candidate shadow evaluation

feedback -> rule proposal -> hash-bound shadow report -> human promotion
```

Model/rule predictions never become labels automatically. Rule promotion uses
SHA-256 receipts, minimum support and precision, atomic replacement, snapshots,
and rollback receipts. A rule digest change invalidates the prior calibration.

## Reliability and security invariants

- Hard-rule precision must be at least 0.98 before short-circuit eligibility.
- English token rules use boundaries and exclusions to avoid substring traps.
- Invalid structured classifier output is retried once, then typed as a failure.
- Referential context is bounded; optional memory is answer-time only.
- Upstream failures map to 502/504, including explicit mid-stream termination.
- Non-loopback serving requires an API key.
- Content logging and memory are disabled by default; local state uses 0600.
- Prometheus metrics distinguish route source, degradation, upstream status,
  stream status, and path latency.

## Trade-offs

### Rules plus a small model

Rules provide sub-millisecond, explainable decisions for narrow, validated
patterns. The small model handles ambiguity and conflicts. This is more complex
than model-only routing, but creates measurable fast paths and safer outages.

### Local files instead of a database

JSONL/YAML keeps the project auditable and easy to run locally. Locks are
process-local, so this design intentionally targets one worker. Multi-instance
deployment would require transactional storage for review and feedback state.

### One process for serving and review

Co-location keeps the portfolio demo simple. A production multi-tenant system
should separate control-plane review/rule administration from the read-only
serving data plane.

### Ollama-specific backend

Direct Ollama integration minimizes adapter code for a local-first project.
Supporting cloud or redundant providers should introduce narrow classifier/chat
backend interfaces before adding provider-specific branches.

### Compatibility, not full protocol identity

The project supports the common Chat Completions and Ollama Chat paths, but not
the Responses API or every OpenAI option. Usage accounting is estimated.

## Residual architectural risks

- Review/control and inference code have separate dependency boundaries but
  remain co-located in one process and security boundary.
- Runtime JSONL stores are process-local and unsuitable for multi-writer
  horizontal scaling.
- The receipt binds prompt, rule, and model content. Changes elsewhere in the
  Python classification policy still rely on reviewed version discipline and
  require a new private locked test before recalibration.

For a detailed Chinese code walkthrough, see
[docs/architecture.zh-CN.md](docs/architecture.zh-CN.md).
