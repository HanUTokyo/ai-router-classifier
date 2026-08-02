# API contract

The service exposes one classifier API and two answer-model adapters. FastAPI
also serves an interactive schema at `/docs` when the caller is authorized.

## Authentication

Loopback is the secure default. Binding to any other host requires
`AI_ROUTER_API_KEY`. Supply it as either:

```text
Authorization: Bearer <key>
X-API-Key: <key>
```

Only `/health` and `/health/live` are unauthenticated when a key is configured.
The service generates a fresh `X-Request-ID` for every response.

## POST /route

Classifies a request without calling an answer model.

```json
{
  "message": "Help me fix this Python function",
  "context": [
    {"role": "user", "content": "The parser drops empty values."},
    {"role": "assistant", "content": "The filter may be too broad."}
  ]
}
```

- `message` is required, non-blank, and limited to 10,000 characters.
- `context` is optional and limited to six messages.
- Context reaches the classifier only when the current task contains a
  referential hint; memory never reaches the classifier.

Example response:

```json
{
  "route": "code",
  "confidence": 1.0,
  "source": "hard_rule",
  "rule_hits": [
    {"rule_id": "code-explicit-en", "label": "code", "kind": "hard", "weight": 1.0}
  ],
  "classifier_model": null,
  "classifier_error_type": null,
  "classifier_version": "1.0.0+rules-1.15.0-...+prompt-classifier-v11+cal-router-locked-test-v4",
  "degraded": false,
  "latency_ms": 0.2
}
```

`confidence` is the empirical precision for the decision source and predicted
label on the accepted locked test. It is not a per-request model probability.
It is `null` when calibration metadata is missing or no precision exists for
that source/label.

Decision sources are `hard_rule`, `small_model`, `small_model_rule_agree`,
`weak_rule_fallback`, `default_fallback`, and `forced` (internal only).

## POST /route/feedback

Stores a human-verified label and the active classifier/rule metadata.

```json
{
  "message": "Help me fix this Python function",
  "expected_route": "code",
  "actual_route": "chat",
  "tags": ["manual-review"],
  "comment": "Implementation action was missed."
}
```

The endpoint returns `ok`, `stored`, and a `rule_signal`. Feedback never edits
active rules or becomes gold data automatically.

## POST /v1/chat/completions

Accepts an OpenAI-style Chat Completions request. Supported fields include
`messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `tools`, and
`tool_choice` values `auto` or `none`.

- The request `model` is a logical compatibility field. Server configuration
  chooses the classifier and answer models.
- Non-streaming responses follow the Chat Completions shape.
- Streaming responses use SSE and terminate with `data: [DONE]`.
- `response_format` and nonstandard `tool_choice` policies return HTTP 400.
- Tool requests use `tool_model` when the routed answer model is not listed in
  `tool_supported_models`.
- Token usage is estimated from character length and is not tokenizer-exact.

The response header `X-AI-Route` is included for streaming requests.

## POST /api/chat

Accepts an Ollama-style chat request. `options.temperature`, `options.top_p`,
and `options.num_predict` are forwarded. Streaming uses newline-delimited JSON
and includes `X-AI-Route`.

## POST /ask

Deprecated compatibility endpoint. Responses include `Deprecation`, `Sunset`,
and successor-version headers. New clients should use `/v1/chat/completions`.

## Health and metrics

- `GET /health/live` confirms that the process is running.
- `GET /health/ready` verifies required models and reports calibration, rules,
  shadow rules, and installed model digests.
- `GET /health` is an alias of readiness.
- `GET /metrics` returns Prometheus exposition format.
- `GET /v1/models` exposes the logical public model ID.

## Errors

| Status | Meaning |
|---:|---|
| 400 | Invalid messages or unsupported compatibility option |
| 401 | Missing or invalid API key |
| 409 | Stale/conflicting human review update |
| 501 | Responses API is not implemented |
| 502 | Upstream model request failed |
| 503 | Readiness, review data, or required model unavailable |
| 504 | Upstream model timed out |

If an upstream stream terminates without a `done` event, the adapter emits an
explicit terminal error rather than presenting an incomplete response as
successful.
