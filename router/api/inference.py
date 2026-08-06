from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..gateway import GatewayResult, ThinGateway
from ..observability import (
    active_requests,
    log_event,
    route_decisions_total,
    route_latency_seconds,
    streams_total,
    upstream_errors_total,
    upstream_latency_seconds,
)
from ..ollama import OllamaError, OllamaTimeout
from ..settings import Settings
from ..storage import content_hash
from ..types import Message, RouteRequest
from .runtime import RuntimeServices


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def require_nonempty_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: Any | None = None


class OllamaChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None


def build_inference_router(services: RuntimeServices) -> APIRouter:
    router = APIRouter()
    selected = services.settings
    classifier = services.classifier
    gateway = services.gateway

    @router.post("/route")
    async def route(req: RouteRequest):
        decision = await classifier.classify(req.message, req.context)
        _record_decision(decision)
        _log_decision(decision, req.message)
        return decision.model_dump(mode="json")

    @router.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": selected.gateway.public_model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @router.post("/v1/chat/completions")
    async def openai_chat(req: OpenAIChatRequest):
        if req.response_format is not None:
            return _error_response(
                400,
                "response_format is not supported by the selected Ollama adapter",
                "invalid_request_error",
                "unsupported_response_format",
            )
        if req.tool_choice not in (None, "auto", "none"):
            return _error_response(
                400,
                "Only tool_choice='auto' or 'none' is supported",
                "invalid_request_error",
                "unsupported_tool_choice",
            )
        if req.tool_choice == "none":
            req = req.model_copy(update={"tools": None})
        if req.stream:
            return await _openai_stream_response(gateway, selected, req)
        active_requests.labels(endpoint="openai_chat").inc()
        started = time.perf_counter()
        try:
            result = await gateway.complete(
                messages=req.messages,
                tools=req.tools,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )
            _record_decision(result.decision)
            upstream_latency_seconds.labels(
                model=result.selected_model,
                status="success",
                stream="false",
            ).observe(time.perf_counter() - started)
            return _openai_completion(selected, result, req.messages)
        except (OllamaError, ValueError) as exc:
            _record_upstream_error(
                _error_model(exc, selected.gateway.chat_model),
                exc,
                started,
                False,
            )
            return _upstream_error_response(exc)
        finally:
            active_requests.labels(endpoint="openai_chat").dec()

    @router.post("/api/chat")
    async def ollama_chat(req: OllamaChatRequest):
        temperature = _option_float(req.options, "temperature")
        top_p = _option_float(req.options, "top_p")
        max_tokens = _option_int(req.options, "num_predict")
        if req.stream:
            return await _ollama_stream_response(
                gateway,
                req,
                temperature,
                top_p,
                max_tokens,
            )
        active_requests.labels(endpoint="ollama_chat").inc()
        started = time.perf_counter()
        try:
            result = await gateway.complete(
                messages=req.messages,
                tools=req.tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            _record_decision(result.decision)
            upstream_latency_seconds.labels(
                model=result.selected_model,
                status="success",
                stream="false",
            ).observe(time.perf_counter() - started)
            message = result.upstream.get("message", {})
            return {
                "model": result.selected_model,
                "created_at": result.upstream.get("created_at"),
                "message": message,
                "done_reason": result.upstream.get("done_reason", "stop"),
                "done": True,
                "route": result.decision.route.value,
            }
        except (OllamaError, ValueError) as exc:
            _record_upstream_error(
                _error_model(exc, selected.gateway.chat_model),
                exc,
                started,
                False,
            )
            return _upstream_error_response(exc)
        finally:
            active_requests.labels(endpoint="ollama_chat").dec()

    @router.post("/ask")
    async def ask(req: AskRequest):
        active_requests.labels(endpoint="ask").inc()
        started = time.perf_counter()
        try:
            result = await gateway.complete(
                messages=[Message(role="user", content=req.message)]
            )
            _record_decision(result.decision)
            upstream_latency_seconds.labels(
                model=result.selected_model,
                status="success",
                stream="false",
            ).observe(time.perf_counter() - started)
            response = JSONResponse(
                content={
                    **result.decision.model_dump(mode="json"),
                    "model": result.selected_model,
                    "answer": result.upstream.get("message", {}).get("content", ""),
                }
            )
            return _with_deprecation_headers(response)
        except (OllamaError, ValueError) as exc:
            _record_upstream_error(
                _error_model(exc, selected.gateway.chat_model),
                exc,
                started,
                False,
            )
            return _with_deprecation_headers(_upstream_error_response(exc))
        finally:
            active_requests.labels(endpoint="ask").dec()

    @router.post("/v1/responses", include_in_schema=False)
    async def responses_not_implemented():
        return _error_response(
            501,
            "The Responses API is not implemented. Use /v1/chat/completions.",
            "not_implemented_error",
            "responses_api_not_supported",
        )

    return router


async def _openai_stream_response(
    gateway: ThinGateway,
    settings: Settings,
    req: OpenAIChatRequest,
):
    active_requests.labels(endpoint="openai_chat").inc()
    started = time.perf_counter()
    try:
        stream = await gateway.stream(
            messages=req.messages,
            tools=req.tools,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
        _record_decision(stream.decision)
    except (OllamaError, ValueError) as exc:
        active_requests.labels(endpoint="openai_chat").dec()
        _record_upstream_error(
            _error_model(exc, settings.gateway.chat_model),
            exc,
            started,
            True,
        )
        return _upstream_error_response(exc)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def events() -> AsyncIterator[str]:
        status = "success"
        text_parts: list[str] = []
        tool_calls_seen = False
        saw_done = False
        try:
            async for line in stream.upstream.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise OllamaError(str(data["error"]))
                message = data.get("message") or {}
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls")
                if content:
                    text_parts.append(str(content))
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.gateway.public_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                if tool_calls:
                    tool_calls_seen = True
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.gateway.public_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": _normalize_tool_calls(tool_calls)
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                if data.get("done"):
                    saw_done = True
                    break
            if not saw_done:
                raise OllamaError("Upstream stream ended before a done event")
            usage = _usage(req.messages, "".join(text_parts))
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.gateway.public_model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": (
                                "tool_calls" if tool_calls_seen else "stop"
                            ),
                        }
                    ],
                    "usage": usage,
                }
            )
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = "error"
            upstream_errors_total.labels(
                model=stream.selected_model,
                error_type=type(exc).__name__,
            ).inc()
            yield (
                "event: error\n"
                f"data: {json.dumps({'error': {'message': 'Upstream stream failed', 'type': type(exc).__name__}})}\n\n"
            )
            yield "data: [DONE]\n\n"
        finally:
            try:
                await stream.upstream.aclose()
            finally:
                streams_total.labels(protocol="openai", status=status).inc()
                upstream_latency_seconds.labels(
                    model=stream.selected_model,
                    status=status,
                    stream="true",
                ).observe(time.perf_counter() - started)
                active_requests.labels(endpoint="openai_chat").dec()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-AI-Route": stream.decision.route.value,
        },
    )


async def _ollama_stream_response(
    gateway: ThinGateway,
    req: OllamaChatRequest,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
):
    active_requests.labels(endpoint="ollama_chat").inc()
    started = time.perf_counter()
    try:
        stream = await gateway.stream(
            messages=req.messages,
            tools=req.tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        _record_decision(stream.decision)
    except (OllamaError, ValueError) as exc:
        active_requests.labels(endpoint="ollama_chat").dec()
        _record_upstream_error(
            _error_model(exc, "unknown"),
            exc,
            started,
            True,
        )
        return _upstream_error_response(exc)

    async def lines() -> AsyncIterator[str]:
        status = "success"
        saw_done = False
        try:
            async for line in stream.upstream.aiter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(str(data["error"]))
                    saw_done = saw_done or bool(data.get("done"))
                    yield f"{line}\n"
                    if data.get("done"):
                        break
            if not saw_done:
                raise OllamaError("Upstream stream ended before a done event")
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = "error"
            upstream_errors_total.labels(
                model=stream.selected_model,
                error_type=type(exc).__name__,
            ).inc()
            yield json.dumps(
                {
                    "error": "Upstream stream failed",
                    "error_type": type(exc).__name__,
                    "done": True,
                }
            ) + "\n"
        finally:
            try:
                await stream.upstream.aclose()
            finally:
                streams_total.labels(protocol="ollama", status=status).inc()
                upstream_latency_seconds.labels(
                    model=stream.selected_model,
                    status=status,
                    stream="true",
                ).observe(time.perf_counter() - started)
                active_requests.labels(endpoint="ollama_chat").dec()

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"X-AI-Route": stream.decision.route.value},
    )


def _openai_completion(
    settings: Settings,
    result: GatewayResult,
    messages: list[Message],
) -> dict[str, Any]:
    raw_message = result.upstream.get("message") or {}
    content = str(raw_message.get("content") or "")
    tool_calls = _normalize_tool_calls(raw_message.get("tool_calls") or [])
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else (None if tool_calls else ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.gateway.public_model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": _usage(messages, content),
    }


def _normalize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, call in enumerate(calls):
        function = call.get("function") or {}
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": function.get("name") or f"tool_{index}",
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _usage(messages: list[Message], completion: str) -> dict[str, int]:
    prompt = "".join(_message_text(message) for message in messages)
    prompt_tokens = max(1, len(prompt) // 4) if prompt else 0
    completion_tokens = max(1, len(completion) // 4) if completion else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _record_decision(decision) -> None:
    route_decisions_total.labels(
        route=decision.route.value,
        source=decision.source.value,
        degraded=str(decision.degraded).lower(),
    ).inc()
    route_latency_seconds.labels(source=decision.source.value).observe(
        decision.latency_ms / 1000
    )


def _log_decision(decision, message: str) -> None:
    log_event(
        "route_decision",
        message_hash=content_hash(message.strip()),
        message_length=len(message),
        route=decision.route.value,
        source=decision.source.value,
        degraded=decision.degraded,
        classifier_error_type=decision.classifier_error_type,
        classifier_model=decision.classifier_model,
        classifier_version=decision.classifier_version,
        latency_ms=decision.latency_ms,
        rule_ids=[hit.rule_id for hit in decision.rule_hits],
    )


def _record_upstream_error(
    model: str,
    exc: Exception,
    started: float,
    stream: bool,
) -> None:
    upstream_errors_total.labels(
        model=model,
        error_type=type(exc).__name__,
    ).inc()
    upstream_latency_seconds.labels(
        model=model,
        status="error",
        stream=str(stream).lower(),
    ).observe(time.perf_counter() - started)


def _error_model(exc: Exception, fallback: str) -> str:
    selected = getattr(exc, "selected_model", None)
    return str(selected) if selected else fallback


def _with_deprecation_headers(response: JSONResponse) -> JSONResponse:
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Thu, 01 Oct 2026 00:00:00 GMT"
    response.headers["Link"] = (
        '</v1/chat/completions>; rel="successor-version"'
    )
    return response


def _upstream_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, OllamaTimeout):
        return _error_response(
            504,
            "Upstream model timed out",
            "upstream_timeout",
            "upstream_timeout",
        )
    if isinstance(exc, ValueError):
        return _error_response(
            400,
            str(exc),
            "invalid_request_error",
            "invalid_messages",
        )
    return _error_response(
        502,
        "Upstream model request failed",
        "upstream_error",
        "upstream_failed",
    )


def _error_response(
    status: int,
    message: str,
    error_type: str,
    code: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _option_float(options: dict[str, Any] | None, key: str) -> float | None:
    if not options or options.get(key) is None:
        return None
    return float(options[key])


def _option_int(options: dict[str, Any] | None, key: str) -> int | None:
    if not options or options.get(key) is None:
        return None
    return int(options[key])
