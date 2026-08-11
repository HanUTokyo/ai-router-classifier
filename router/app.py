from __future__ import annotations

import hmac
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.control import build_control_router
from .api.health import build_health_router
from .api.inference import build_inference_router
from .api.runtime import RuntimeServices
from .classifier import RouterClassifier
from .gateway import ThinGateway
from .observability import configure_logging, http_requests_total, log_event
from .ollama import OllamaClient, OllamaError, OllamaUnavailable
from .review_ui import ReviewDatasetStore
from .settings import Settings
from .storage import SecureJsonlStore


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
    services = RuntimeServices(
        settings=selected,
        ollama=ollama,
        classifier=classifier,
        gateway=gateway,
        feedback_store=SecureJsonlStore(selected.storage.feedback_path),
        review_store=ReviewDatasetStore(selected.storage.review_dataset_path),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if selected.classifier.strict_model_check:
            installed = await ollama.ensure_models(selected.required_models)
            if not classifier.bind_model_digest(
                installed.get(selected.classifier.model)
            ):
                raise OllamaUnavailable(
                    "Installed classifier model digest does not match the "
                    "accepted calibration receipt"
                )
        else:
            try:
                installed = await ollama.installed_models()
            except OllamaError:
                installed = {}
            classifier.bind_model_digest(installed.get(selected.classifier.model))
        yield
        await ollama.close()

    app = FastAPI(title="AI Router Classifier", version="1.0.0", lifespan=lifespan)
    app.state.services = services
    app.state.settings = selected
    app.state.ollama = ollama
    app.state.classifier = classifier
    app.state.gateway = gateway
    app.state.review_store = services.review_store

    @app.middleware("http")
    async def security_and_request_log(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
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
                response = _error_response(
                    401,
                    "Invalid or missing API key",
                    "authentication_error",
                    "invalid_api_key",
                )
                http_requests_total.labels(
                    endpoint=_endpoint_label(request.url.path),
                    status="401",
                ).inc()
                log_event(
                    "http_request",
                    request_id=request_id,
                    path=request.url.path,
                    method=request.method,
                    status=401,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                )
                response.headers["X-Request-ID"] = request_id
                return response
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

    app.include_router(build_health_router(services))
    app.include_router(build_control_router(services))
    app.include_router(build_inference_router(services))
    return app


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
    if path == "/review" or path.startswith("/review/api/"):
        return "human_review"
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
