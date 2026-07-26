from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional
import json
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from gateway_service import (
    ask_with_routing,
    ask_with_routing_messages,
    ask_with_routing_messages_stream,
    get_memory_mode,
)

# ✅ 导入 Prometheus 指标
from metrics import (
    record_llm_tokens,
    record_llm_error,
    get_metrics,
    llm_calls_total,
    llm_latency_seconds,
    llm_errors_total,
    llm_active_requests,
)

try:
    from router.logger import log_event
except ImportError:
    from logger import log_event


app = FastAPI(title="local-agent")
OPENAI_PUBLIC_MODEL_ID = "local-agent"
OLLAMA_PUBLIC_MODEL_ID = "gemma4:e4b"
DEBUG_MODEL_ROUTES = {
    "local-agent-chat": "chat",
    "local-agent-reason": "reason",
    "local-agent-code": "code",
}


@app.middleware("http")
async def log_openai_compatible_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)

    if request.url.path.startswith("/v1/"):
        _log_gateway_event(
            "gateway_http_request",
            request.headers.get("x-request-id") or str(uuid.uuid4()),
            path=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            latency=round(time.time() - start, 4),
            accept=request.headers.get("accept"),
            content_type=request.headers.get("content-type"),
            response_content_type=response.headers.get("content-type"),
            user_agent=request.headers.get("user-agent"),
        )

    return response


class AskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OllamaMessage]
    stream: Optional[bool] = False


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAIChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[Dict[str, Any]] = None
    metadata: Optional[Any] = None
    response_format: Optional[Any] = None


class OpenAIResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: Any = ""
    stream: Optional[bool] = False
    temperature: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "local-agent"}


@app.get("/metrics")
def metrics():
    """📊 Prometheus 指标端点

    可访问：http://localhost:8000/metrics
    """
    return get_metrics()



def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


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


def _clip(value: Any, limit: int = 200) -> Any:
    if value is None:
        return None
    text = str(value)
    return text[:limit] if len(text) > limit else text


def _log_gateway_event(event_type: str, trace_id: str, **fields: Any) -> None:
    log_event({"type": event_type, **fields}, trace_id)


def _openai_error(
    status_code: int,
    message: str,
    error_type: str,
    param: Optional[str] = None,
    code: Optional[str] = None,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _build_openai_usage(request_messages: List[Dict[str, str]], assistant_content: str) -> Dict[str, int]:
    prompt_text = "".join(str(msg.get("content") or "") for msg in request_messages)
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(assistant_content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _extract_stream_usage(
    upstream_data: Dict[str, Any],
    request_messages: List[Dict[str, str]],
    assistant_content: str,
) -> Dict[str, int]:
    prompt_tokens = upstream_data.get("prompt_eval_count")
    completion_tokens = upstream_data.get("eval_count")

    if prompt_tokens is None or completion_tokens is None:
        return _build_openai_usage(request_messages, assistant_content)

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }


def _build_openai_completion_response(
    completion_id: str,
    model: str,
    assistant_content: str,
    request_messages: List[Dict[str, str]],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    created = int(time.time())
    usage = _build_openai_usage(request_messages, assistant_content)
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content if assistant_content else (None if tool_calls else ""),
    }
    if tool_calls:
        message["tool_calls"] = _normalize_tool_calls(tool_calls)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "message": message,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": assistant_content,
                    }
                ],
            }
        ],
        "usage": usage,
    }


def _model_route_override(requested_model: Optional[str]) -> Optional[str]:
    if not requested_model:
        return None
    return DEBUG_MODEL_ROUTES.get(requested_model)


def _normalize_openai_message(message: OpenAIMessage) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "role": message.role,
        "content": _normalize_content(message.content),
    }
    if message.name:
        normalized["name"] = message.name
    if message.tool_call_id:
        normalized["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        normalized["tool_calls"] = message.tool_calls
    return normalized


def _request_client_context(request: Request, req: OpenAIChatCompletionsRequest) -> Dict[str, Any]:
    user_agent = request.headers.get("user-agent", "")
    lowered_ua = user_agent.lower()
    metadata = req.metadata if isinstance(req.metadata, dict) else {}

    workspace_id = (
        request.headers.get("x-anythingllm-workspace-id")
        or request.headers.get("x-workspace-id")
        or metadata.get("workspace_id")
        or metadata.get("workspaceId")
    )
    user_id = (
        request.headers.get("x-anythingllm-user-id")
        or request.headers.get("x-user-id")
        or req.user
        or metadata.get("user_id")
        or metadata.get("userId")
    )
    is_anythingllm = (
        "anythingllm" in lowered_ua
        or "anything-llm" in lowered_ua
        or bool(request.headers.get("x-anythingllm-workspace-id"))
        or metadata.get("client") in {"anythingllm", "anything-llm"}
    )

    return {
        "client": "anythingllm" if is_anythingllm else "openai_compatible",
        "workspace_id": workspace_id,
        "user_id": user_id,
    }


def _normalize_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tool_calls:
        return None

    normalized = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            function = {}

        normalized.append({
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": call.get("type") or "function",
            "function": {
                "name": function.get("name") or call.get("name") or f"tool_{index}",
                "arguments": function.get("arguments") or call.get("arguments") or "{}",
            },
        })

    return normalized or None


def _tool_calls_delta(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = _normalize_tool_calls(tool_calls) or []
    return [
        {
            "index": index,
            "id": call.get("id"),
            "type": call.get("type"),
            "function": call.get("function"),
        }
        for index, call in enumerate(normalized)
    ]


@app.post("/ask")
def ask(req: AskRequest):
    trace_id = str(uuid.uuid4())
    text = req.message.strip()
    if not text:
        return {
            "route": "chat",
            "model": "chat",
            "confidence": 0.3,
            "source": "invalid_input",
            "answer": "",
        }
    return ask_with_routing(text, trace_id)


@app.post("/api/chat")
def ollama_chat(req: OllamaChatRequest):
    trace_id = str(uuid.uuid4())
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if not messages:
        content = ""
    else:
        result = ask_with_routing_messages(messages, trace_id)
        content = result["answer"]

    return {
        "model": req.model or OLLAMA_PUBLIC_MODEL_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done_reason": "stop",
        "done": True,
    }


@app.get("/api/tags")
def ollama_tags():
    return {
        "models": [
            {
                "name": OLLAMA_PUBLIC_MODEL_ID,
                "model": OLLAMA_PUBLIC_MODEL_ID,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": 1000000000,
                "digest": "gemma4:e4b",
                "details": {
                    "format": "gguf",
                    "family": "gemma4",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }


@app.get("/v1/models")
def openai_models(_authorization: Optional[str] = Header(default=None, alias="Authorization")):
    _ = _authorization
    now = int(time.time())
    models = [
        {
            "id": OPENAI_PUBLIC_MODEL_ID,
            "object": "model",
            "created": now,
            "owned_by": "local-agent",
        }
    ]
    for model_id in DEBUG_MODEL_ROUTES:
        models.append({
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "local-agent-debug",
        })
    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(
    request: Request,
    req: OpenAIChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    accept: Optional[str] = Header(default=None, alias="Accept"),
):
    start_time = time.time()
    model_name = req.model or OPENAI_PUBLIC_MODEL_ID

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
            return _openai_error(
                status_code=400,
                message="'messages' must contain at least one message",
                error_type="invalid_request_error",
                param="messages",
                code="empty_messages",
            )

        trace_id = str(uuid.uuid4())
        messages: List[Dict[str, Any]] = [
            _normalize_openai_message(m)
            for m in req.messages
        ]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        force_route_label = _model_route_override(req.model)
        tools_present = bool(req.tools)
        client_context = _request_client_context(request, req)
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg.get("role") == "user"),
            messages[-1]["content"] if messages else "",
        )

        _log_gateway_event(
            "gateway_openai_chat_request",
            trace_id,
            path=str(request.url.path),
            stream=bool(req.stream),
            requested_model=req.model,
            public_model=OPENAI_PUBLIC_MODEL_ID,
            accept=accept,
            content_type=request.headers.get("content-type"),
            user_agent=request.headers.get("user-agent"),
            authorization_present=bool(authorization),
            message_count=len(messages),
            last_user_message=_clip(last_user_message),
            client=client_context["client"],
            workspace_id=client_context["workspace_id"],
            user_id=client_context["user_id"],
            tools_present=tools_present,
            tool_choice=_clip(req.tool_choice),
            memory_mode=get_memory_mode(),
            forced_route=force_route_label,
        )

        if not req.stream:
            try:
                result = ask_with_routing_messages(
                    messages,
                    trace_id,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    force_route_label=force_route_label,
                )
                answer = result.get("answer", "")
                tool_calls = result.get("tool_calls")

                # ✅ 指标：记录成功和 token
                elapsed = time.time() - start_time
                llm_latency_seconds.labels(
                    model=model_name,
                    endpoint="chat",
                    status="success"
                ).observe(elapsed)

                usage = _build_openai_usage(messages, answer)
                record_llm_tokens(
                    model=model_name,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0)
                )

            except Exception as exc:
                elapsed = time.time() - start_time
                error_type = type(exc).__name__
                _log_gateway_event(
                    "gateway_openai_chat_error",
                    trace_id,
                    streaming=False,
                    client=client_context["client"],
                    workspace_id=client_context["workspace_id"],
                    user_id=client_context["user_id"],
                    tools_present=tools_present,
                    memory_mode=get_memory_mode(),
                    final_status="error",
                    error_type=error_type,
                    error=_clip(exc, 500),
                )
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
                return _openai_error(
                    status_code=500,
                    message=f"Gateway inference failed: {exc}",
                    error_type="server_error",
                    code="gateway_inference_failed",
                )

            _log_gateway_event(
                "gateway_openai_chat_response",
                trace_id,
                streaming=False,
                route=result.get("route"),
                selected_model=result.get("model"),
                confidence=result.get("confidence"),
                source=result.get("source"),
                client=client_context["client"],
                workspace_id=client_context["workspace_id"],
                user_id=client_context["user_id"],
                tools_present=tools_present,
                tool_calls_present=bool(tool_calls),
                memory_mode=result.get("memory_mode") or get_memory_mode(),
                final_status="tool_calls" if tool_calls else "stop",
                answer_preview=_clip(answer),
                answer_len=len(answer),
            )

            return _build_openai_completion_response(
                completion_id=completion_id,
                model=OPENAI_PUBLIC_MODEL_ID,
                assistant_content=answer,
                request_messages=messages,
                tool_calls=tool_calls,
            )

        try:
            stream_result = ask_with_routing_messages_stream(
                messages,
                trace_id,
                tools=req.tools,
                tool_choice=req.tool_choice,
                force_route_label=force_route_label,
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            error_type = type(exc).__name__
            _log_gateway_event(
                "gateway_openai_chat_error",
                trace_id,
                streaming=True,
                client=client_context["client"],
                workspace_id=client_context["workspace_id"],
                user_id=client_context["user_id"],
                tools_present=tools_present,
                memory_mode=get_memory_mode(),
                final_status="error",
                error_type=error_type,
                error=_clip(exc, 500),
            )
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
            return _openai_error(
                status_code=500,
                message=f"Gateway inference failed: {exc}",
                error_type="server_error",
                code="gateway_inference_failed",
            )

        _log_gateway_event(
            "gateway_openai_chat_stream_start",
            trace_id,
            route=stream_result.get("route"),
            selected_model=stream_result.get("model"),
            confidence=stream_result.get("confidence"),
            source=stream_result.get("source"),
            client=client_context["client"],
            workspace_id=client_context["workspace_id"],
            user_id=client_context["user_id"],
            tools_present=tools_present,
            memory_mode=stream_result.get("memory_mode") or get_memory_mode(),
        )

        created = int(time.time())

        def event_stream():
            emitted_chunks = 0
            emitted_chars = 0
            done_seen = False
            tool_calls_seen = False
            collected_text_parts: List[str] = []
            final_usage: Optional[Dict[str, int]] = None

            try:
                for _, line in stream_result["stream"]:
                    data = json.loads(line)
                    message = data.get("message", {}) or {}
                    piece = message.get("content", "")
                    tool_calls = message.get("tool_calls")

                    if piece:
                        emitted_chunks += 1
                        emitted_chars += len(piece)
                        collected_text_parts.append(piece)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": OPENAI_PUBLIC_MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        if emitted_chunks == 1:
                            _log_gateway_event(
                                "gateway_openai_chat_stream_first_content_chunk",
                                trace_id,
                                upstream_line=_clip(line, 500),
                                chunk=chunk,
                            )
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if tool_calls:
                        tool_calls_seen = True
                        emitted_chunks += 1
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": OPENAI_PUBLIC_MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": _tool_calls_delta(tool_calls),
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        _log_gateway_event(
                            "gateway_openai_chat_stream_tool_calls",
                            trace_id,
                            upstream_line=_clip(line, 500),
                            tool_call_count=len(tool_calls),
                        )
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if data.get("done"):
                        done_seen = True
                        final_usage = _extract_stream_usage(
                            data,
                            request_messages=messages,
                            assistant_content="".join(collected_text_parts),
                        )
                        break
            finally:
                # ✅ 指标：记录流式响应的延迟和 token
                elapsed = time.time() - start_time
                llm_latency_seconds.labels(
                    model=model_name,
                    endpoint="chat",
                    status="success"
                ).observe(elapsed)

                if final_usage:
                    record_llm_tokens(
                        model=model_name,
                        input_tokens=final_usage.get("prompt_tokens", 0),
                        output_tokens=final_usage.get("completion_tokens", 0)
                    )

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": OPENAI_PUBLIC_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls_seen else "stop",
                    }
                ],
                "usage": final_usage or _build_openai_usage(messages, "".join(collected_text_parts)),
            }
            _log_gateway_event(
                "gateway_openai_chat_stream_end",
                trace_id,
                emitted_chunks=emitted_chunks,
                emitted_chars=emitted_chars,
                done_seen=done_seen,
                tool_calls_seen=tool_calls_seen,
                usage=final_chunk["usage"],
                client=client_context["client"],
                workspace_id=client_context["workspace_id"],
                user_id=client_context["user_id"],
                tools_present=tools_present,
                memory_mode=stream_result.get("memory_mode") or get_memory_mode(),
                route=stream_result.get("route"),
                selected_model=stream_result.get("model"),
                confidence=stream_result.get("confidence"),
                final_status="tool_calls" if tool_calls_seen else "stop",
                final_chunk=final_chunk,
            )
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

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



@app.post("/v1/responses")
def openai_responses(
    req: OpenAIResponsesRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    _ = authorization

    input_content = req.input
    if isinstance(input_content, list):
        normalized_input = _normalize_content(input_content)
    elif isinstance(input_content, dict):
        normalized_input = _normalize_content([input_content])
    else:
        normalized_input = _normalize_content(input_content)

    messages = [{"role": "user", "content": normalized_input}]
    trace_id = str(uuid.uuid4())

    try:
        result = ask_with_routing_messages(messages, trace_id)
        answer = result.get("answer") or "..."
    except Exception as exc:
        return _openai_error(
            status_code=500,
            message=f"Gateway inference failed: {exc}",
            error_type="server_error",
            code="gateway_inference_failed",
        )

    return {
        "id": f"resp-{uuid.uuid4().hex}",
        "object": "response",
        "created": int(time.time()),
        "model": OPENAI_PUBLIC_MODEL_ID,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": answer,
                    }
                ],
            }
        ],
    }
