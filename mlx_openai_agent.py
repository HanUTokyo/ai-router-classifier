from fastapi import FastAPI, Header, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import json
import os
import time
import uuid

import requests
from fastapi.responses import JSONResponse, StreamingResponse

# ✅ 导入 Prometheus 指标
from metrics import (
    record_llm_call,
    record_llm_tokens,
    record_llm_error,
    get_metrics,
    llm_active_requests,
    llm_errors_total,
    llm_latency_seconds,
    llm_calls_total,
)


app = FastAPI(title="mlx-openai-agent")

UPSTREAM_URL = os.getenv(
    "MLX_UPSTREAM_URL",
    os.getenv("MLX_OLLAMA_URL", "http://192.168.1.66:8080/v1/chat/completions"),
)
UPSTREAM_MODEL = os.getenv("MLX_OLLAMA_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
PUBLIC_MODEL_ID = os.getenv("MLX_PUBLIC_MODEL_ID", "mlx-community/Qwen3.5-9B-MLX-4bit")
REQUEST_TIMEOUT = int(os.getenv("MLX_REQUEST_TIMEOUT", "6000"))
MAX_INPUT_TOKENS = int(os.getenv("MLX_MAX_INPUT_TOKENS", "4096"))


class OpenAIMessage(BaseModel):
    role: str
    content: Any


class OpenAIChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def token_count(messages: List[Dict[str, str]]) -> int:
    text = "".join(m.get("content", "") for m in messages)
    if not text:
        return 0
    return max(1, len(text) // 4)


def trim(messages: List[Dict[str, str]], max_tokens: int = MAX_INPUT_TOKENS) -> List[Dict[str, str]]:
    if token_count(messages) <= max_tokens:
        return messages

    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]
    if not non_system_messages:
        return system_messages

    # Protocol guarantee: keep the last user message as the final query.
    last_user_idx = None
    for i in range(len(non_system_messages) - 1, -1, -1):
        if non_system_messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return system_messages

    candidate_dialogue = non_system_messages[: last_user_idx + 1]
    anchor_user = dict(candidate_dialogue[-1])

    kept_dialogue: List[Dict[str, str]] = [anchor_user]
    for msg in reversed(candidate_dialogue[:-1]):
        next_kept = [msg] + kept_dialogue
        candidate = system_messages + next_kept
        if token_count(candidate) <= max_tokens:
            kept_dialogue = next_kept

    trimmed = system_messages + kept_dialogue

    # If still too large, truncate the anchor user content but keep role=user.
    if token_count(trimmed) > max_tokens:
        last_msg = dict(trimmed[-1])
        if last_msg.get("role") == "user":
            content = last_msg.get("content", "")
            if len(content) > 2000:
                last_msg["content"] = content[-2000:]
                trimmed[-1] = last_msg

    while token_count(trimmed) > max_tokens and len(trimmed) > 1:
        drop_index = None
        for i, msg in enumerate(trimmed):
            if msg.get("role") != "system" and i != len(trimmed) - 1:
                drop_index = i
                break
        if drop_index is None:
            break
        trimmed.pop(drop_index)

    # If system messages alone keep us above cap, drop oldest system messages.
    while token_count(trimmed) > max_tokens:
        system_index = None
        for i, msg in enumerate(trimmed):
            if msg.get("role") == "system":
                system_index = i
                break
        if system_index is None:
            break
        trimmed.pop(system_index)

    # Final fallback: trim the last user content by characters until within cap.
    if token_count(trimmed) > max_tokens and trimmed and trimmed[-1].get("role") == "user":
        last_msg = dict(trimmed[-1])
        content = last_msg.get("content", "")
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            last_msg["content"] = content[-max_chars:]
            trimmed[-1] = last_msg

    return trimmed


def _prepare_messages_for_upstream(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if not role:
            continue
        cleaned.append({"role": role, "content": content})

    user_indices = [i for i, m in enumerate(cleaned) if m.get("role") == "user" and m.get("content", "").strip()]
    if not user_indices:
        raise ValueError("no_user_message")

    # Keep history only up to the latest user message so upstream always sees final role=user.
    last_user_index = user_indices[-1]
    prepared = cleaned[: last_user_index + 1]
    return prepared


def _usage_from_messages(messages: List[Dict[str, str]], assistant_content: str) -> Dict[str, int]:
    prompt_tokens = token_count(messages)
    completion_tokens = max(1, len(assistant_content) // 4) if assistant_content else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _openai_error(status_code: int, message: str, error_type: str, code: Optional[str] = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "mlx-openai-agent"}


@app.get("/metrics")
def metrics():
    """📊 Prometheus 指标端点
    
    可访问：http://localhost:8000/metrics
    """
    return get_metrics()



@app.get("/v1/models")
def openai_models(_authorization: Optional[str] = Header(default=None, alias="Authorization")):
    _ = _authorization
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": PUBLIC_MODEL_ID,
                "object": "model",
                "created": now,
                "owned_by": "mlx-openai-agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(
    request: Request,
    req: OpenAIChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    _ = request
    _ = authorization

    start_time = time.time()
    model_name = req.model or UPSTREAM_MODEL
    
    # ✅ 指标：记录调用
    llm_calls_total.labels(model=model_name, endpoint="chat").inc()
    llm_active_requests.labels(model=model_name, endpoint="chat").inc()

    try:
        if not req.messages:
            llm_errors_total.labels(
                model=model_name,
                endpoint="chat",
                error_type="invalid_request"
            ).inc()
            return _openai_error(400, "'messages' must contain at least one message", "invalid_request_error", "empty_messages")

        normalized_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": _normalize_content(m.content)} for m in req.messages
        ]

        try:
            normalized_messages = _prepare_messages_for_upstream(normalized_messages)
        except ValueError as exc:
            llm_errors_total.labels(
                model=model_name,
                endpoint="chat",
                error_type="invalid_messages"
            ).inc()
            if str(exc) == "no_user_message":
                return _openai_error(
                    400,
                    "No user query found in messages. At least one role='user' message is required.",
                    "invalid_request_error",
                    "no_user_message",
                )
            return _openai_error(400, f"Invalid messages: {exc}", "invalid_request_error", "invalid_messages")

        if token_count(normalized_messages) > MAX_INPUT_TOKENS:
            normalized_messages = trim(normalized_messages, MAX_INPUT_TOKENS)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        payload = {
            "model": model_name,
            "messages": normalized_messages,
            "stream": bool(req.stream),
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if not req.stream:
            try:
                res = requests.post(UPSTREAM_URL, json=payload, timeout=REQUEST_TIMEOUT)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                llm_errors_total.labels(
                    model=model_name,
                    endpoint="chat",
                    error_type="timeout"
                ).inc()
                llm_latency_seconds.labels(
                    model=model_name,
                    endpoint="chat",
                    status="timeout"
                ).observe(elapsed)
                return _openai_error(500, f"Upstream inference timeout", "server_error", "upstream_timeout")
            except Exception as exc:
                elapsed = time.time() - start_time
                error_type = type(exc).__name__
                llm_errors_total.labels(
                    model=model_name,
                    endpoint="chat",
                    error_type=error_type
                ).inc()
                llm_latency_seconds.labels(
                    model=model_name,
                    endpoint="chat",
                    status="error"
                ).observe(elapsed)
                return _openai_error(500, f"Upstream inference failed: {exc}", "server_error", "upstream_inference_failed")

            assistant_content = _normalize_content(
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if not assistant_content:
                assistant_content = _normalize_content(data.get("message", {}).get("content", ""))
            usage = _usage_from_messages(normalized_messages, assistant_content)

            # ✅ 指标：记录延迟和 token
            elapsed = time.time() - start_time
            llm_latency_seconds.labels(
                model=model_name,
                endpoint="chat",
                status="success"
            ).observe(elapsed)
            record_llm_tokens(
                model=model_name,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0)
            )

            return {
                "id": data.get("id", completion_id),
                "object": data.get("object", "chat.completion"),
                "created": data.get("created", created),
                "model": PUBLIC_MODEL_ID,
                "choices": data.get(
                    "choices",
                    [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": assistant_content},
                            "finish_reason": "stop",
                        }
                    ],
                ),
                "usage": data.get("usage", usage),
            }

        try:
            upstream_res = requests.post(UPSTREAM_URL, json=payload, timeout=None, stream=False)
            upstream_res.raise_for_status()
        except Exception as exc:
            elapsed = time.time() - start_time
            error_type = type(exc).__name__
            llm_errors_total.labels(
                model=model_name,
                endpoint="chat",
                error_type=error_type
            ).inc()
            llm_latency_seconds.labels(
                model=model_name,
                endpoint="chat",
                status="error"
            ).observe(elapsed)
            return _openai_error(500, f"Upstream inference failed: {exc}", "server_error", "upstream_inference_failed")

        def event_stream():
            try:
                for line in upstream_res.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # Upstream is already OpenAI-compatible SSE; pass through as-is.
                    yield f"{line}\n\n"
            finally:
                elapsed = time.time() - start_time
                llm_latency_seconds.labels(
                    model=model_name,
                    endpoint="chat",
                    status="success"
                ).observe(elapsed)
                upstream_res.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    finally:
        llm_active_requests.labels(model=model_name, endpoint="chat").dec()

