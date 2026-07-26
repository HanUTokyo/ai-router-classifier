# AnythingLLM Integration

This setup keeps AnythingLLM as the outer agent workspace and keeps this
project as the OpenAI-compatible routing gateway.

## Start the Router Gateway

```bash
cd /opt/ai/Router
export ROUTER_MODEL="${ROUTER_MODEL:-qwen2.5:0.2b}"
export UPSTREAM_MEMORY_MODE=anythingllm
export TOOL_CALL_MODEL="${TOOL_CALL_MODEL:-gemma4:e4b}"
bash start.sh start
```

Use `UPSTREAM_MEMORY_MODE=anythingllm` when AnythingLLM personalization is the
source of memory. Use `local` for non-AnythingLLM clients, and `hybrid` when you
want local memory writes for router bias without injecting local memory into the
answering model.

## Start AnythingLLM

```bash
cd /opt/ai/Router
docker compose -f docker-compose.anythingllm.yml up -d
```

Open AnythingLLM at `http://localhost:3002`.

## Configure AnythingLLM Provider

In AnythingLLM, choose the Generic OpenAI/OpenAI-compatible provider:

- Base URL: `http://host.docker.internal:8000/v1`
- Model: `local-agent`
- API key: any non-empty placeholder if the UI requires one

Debug models exposed by the gateway:

- `local-agent-chat`
- `local-agent-reason`
- `local-agent-code`

The debug models force the route while still using the same gateway and logging
path.

## Agent Tool Calls

AnythingLLM may send OpenAI `tools` and `tool_choice` fields during agent chats.
The gateway accepts these fields, logs `tools_present=true`, and forwards
OpenAI-style tool schemas to Ollama. Set `TOOL_CALL_MODEL` to a model that can
return tool calls. If a routed model also supports tools, list it in
`TOOL_CALL_SUPPORTED_MODELS` as a comma-separated value.

Example:

```bash
export TOOL_CALL_MODEL="llama3.1:8b"
export TOOL_CALL_SUPPORTED_MODELS="llama3.1:8b"
```

## Feedback Samples

The classifier feedback endpoint is available on the router service:

```bash
curl http://localhost:8001/route/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我写一个 Python API",
    "expected_route": "code",
    "actual_route": "chat",
    "source": "anythingllm_manual",
    "comment": "User corrected the route"
  }'
```

Samples are written to `Router/router/feedback_data.jsonl` by default. The
router analyzer merges this file into `training_data.jsonl` through its
`--feedback-path` option, which defaults to `feedback_data.jsonl` when running
from `Router/router`.
