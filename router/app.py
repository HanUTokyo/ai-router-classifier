from __future__ import annotations

import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .classifier import RouterClassifier
from .gateway import GatewayResult, GatewayStream, ThinGateway
from .observability import (
    active_requests,
    configure_logging,
    http_requests_total,
    log_event,
    metrics_payload,
    route_decisions_total,
    route_latency_seconds,
    streams_total,
    upstream_errors_total,
    upstream_latency_seconds,
)
from .ollama import OllamaClient, OllamaError, OllamaTimeout
from .settings import Settings
from .storage import SecureJsonlStore, content_hash
from .types import FeedbackRequest, Message, RouteRequest


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


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


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    selected = settings or Settings.load()
    configure_logging(selected.logging.level)
    ollama = OllamaClient(selected.ollama, transport=transport)
    classifier = RouterClassifier(selected, ollama)
    gateway = ThinGateway(selected, classifier, ollama)
    feedback_store = SecureJsonlStore(selected.storage.feedback_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if selected.classifier.strict_model_check:
            await ollama.ensure_models(selected.required_models)
        yield
        await ollama.close()

    app = FastAPI(title="AI Router Classifier", version="1.0.0", lifespan=lifespan)
    app.state.settings = selected
    app.state.ollama = ollama
    app.state.classifier = classifier
    app.state.gateway = gateway

    @app.middleware("http")
    async def security_and_request_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        if selected.server.api_key and request.url.path not in {
            "/health",
            "/health/live",
        }:
            supplied = _extract_api_key(request)
            if supplied is None or not hmac.compare_digest(
                supplied,
                selected.server.api_key,
            ):
                return _error_response(
                    401,
                    "Invalid or missing API key",
                    "authentication_error",
                    "invalid_api_key",
                )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            http_requests_total.labels(
                endpoint=_endpoint_label(request.url.path),
                status="500",
            ).inc()
            raise
        http_requests_total.labels(
            endpoint=_endpoint_label(request.url.path),
            status=str(response.status_code),
        ).inc()
        log_event(
            "http_request",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health/live")
    async def health_live():
        return {"status": "ok", "service": "ai-router"}

    @app.get("/health/ready")
    async def health_ready():
        try:
            installed = await ollama.ensure_models(selected.required_models)
        except OllamaError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        return {
            "status": "ready",
            "classifier_model": selected.classifier.model,
            "calibration": {
                "version": classifier.calibrator.version,
                "validated": classifier.calibrator.validated,
                "hard_rules_enabled": classifier.calibrator.hard_rules_enabled,
                "weak_fallback_enabled": (
                    classifier.calibrator.weak_fallback_enabled
                ),
            },
            "models": {
                model: installed.get(model, "") for model in sorted(selected.required_models)
            },
        }

    @app.get("/health")
    async def health():
        return await health_ready()

    @app.get("/metrics")
    async def metrics():
        body, content_type = metrics_payload()
        return Response(content=body, headers={"Content-Type": content_type})

    @app.post("/route")
    async def route(req: RouteRequest):
        decision = await classifier.classify(req.message, req.context)
        _record_decision(decision)
        _log_decision(decision, req.message)
        return decision.model_dump(mode="json")

    @app.post("/route/feedback")
    async def route_feedback(req: FeedbackRequest, request: Request):
        feedback_store.append(
            {
                "request_id": request.state.request_id,
                "message": req.message.strip(),
                "message_hash": content_hash(req.message.strip()),
                "expected_route": req.expected_route.value,
                "actual_route": (
                    req.actual_route.value if req.actual_route is not None else None
                ),
                "tags": req.tags,
                "comment": req.comment,
                "verified": True,
                "source": "manual_feedback",
            }
        )
        return {"ok": True, "stored": True}

    @app.get("/v1/models")
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

    @app.post("/v1/chat/completions")
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
            _record_upstream_error(selected.gateway.chat_model, exc, started, False)
            return _upstream_error_response(exc)
        finally:
            active_requests.labels(endpoint="openai_chat").dec()

    @app.post("/api/chat")
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
            _record_upstream_error(selected.gateway.chat_model, exc, started, False)
            return _upstream_error_response(exc)
        finally:
            active_requests.labels(endpoint="ollama_chat").dec()

    @app.post("/ask")
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
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = "2026-10-01"
            response.headers["Link"] = '</v1/chat/completions>; rel="successor-version"'
            return response
        except (OllamaError, ValueError) as exc:
            _record_upstream_error(selected.gateway.chat_model, exc, started, False)
            return _upstream_error_response(exc)
        finally:
            active_requests.labels(endpoint="ask").dec()

    @app.post("/v1/responses", include_in_schema=False)
    async def responses_not_implemented():
        return _error_response(
            501,
            "The Responses API is not implemented. Use /v1/chat/completions.",
            "not_implemented_error",
            "responses_api_not_supported",
        )

    return app


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
        _record_upstream_error(settings.gateway.chat_model, exc, started, True)
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
            await stream.upstream.aclose()
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
        _record_upstream_error("unknown", exc, started, True)
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
            if not saw_done:
                raise OllamaError("Upstream stream ended before a done event")
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
            await stream.upstream.aclose()
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


def _extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key")


def _endpoint_label(path: str) -> str:
    return {
        "/route": "route",
        "/route/feedback": "route_feedback",
        "/v1/chat/completions": "openai_chat",
        "/api/chat": "ollama_chat",
        "/ask": "ask",
        "/metrics": "metrics",
        "/health": "health",
        "/health/live": "health_live",
        "/health/ready": "health_ready",
    }.get(path, "other")


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
