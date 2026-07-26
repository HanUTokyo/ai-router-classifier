from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from pathlib import Path
from datetime import datetime
import uuid
import time
import json
import os

try:
    from .router_service import build_route_decision, route_text
    from .logger import log_event
except ImportError:
    from router_service import build_route_decision, route_text
    from logger import log_event

# ✅ 导入 Prometheus 指标
try:
    from metrics import (
        llm_calls_total,
        llm_latency_seconds,
        llm_errors_total,
        llm_active_requests,
        get_metrics,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from metrics import (
        llm_calls_total,
        llm_latency_seconds,
        llm_errors_total,
        llm_active_requests,
        get_metrics,
    )

app = FastAPI(title="Router Classifier")


class RouteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class RouteMessage(BaseModel):
    role: str = "user"
    content: Any = ""


class RouteDecisionRequest(BaseModel):
    message: Optional[str] = Field(default=None, max_length=10000)
    messages: Optional[List[RouteMessage]] = None


class RouteFeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    expected_route: str = Field(..., pattern="^(code|reason|chat)$")
    actual_route: Optional[str] = Field(default=None, pattern="^(code|reason|chat)$")
    source: Optional[str] = Field(default="manual")
    comment: Optional[str] = Field(default=None, max_length=2000)


FEEDBACK_PATH = Path(
    os.getenv(
        "ROUTER_FEEDBACK_PATH",
        Path(__file__).resolve().parent / "feedback_data.jsonl",
    )
)


def _append_feedback_sample(sample: dict) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ✅ 1️⃣ 先 log（内层）
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    path = request.url.path
    method = request.method

    body = None

    def safe_body(b: str):
        return b[:200] if b and len(b) > 200 else b

    if method == "POST":
        body_bytes = await request.body()
        body = body_bytes.decode("utf-8", errors="ignore")
        body = safe_body(body)

        async def receive():
            return {
                "type": "http.request",
                "body": body_bytes,
                "more_body": False,
            }

        request._receive = receive

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        latency = time.time() - start

        log_event({
            "type": "http_error",
            "path": path,
            "method": method,
            "body": body,
            "error": str(e),
            "latency": round(latency, 4)
        }, getattr(request.state, "trace_id", None))

        raise

    latency = time.time() - start

    log_event({
        "type": "http_request",
        "path": path,
        "method": method,
        "body": body,
        "status_code": status_code,
        "latency": round(latency, 4)
    }, getattr(request.state, "trace_id", None))

    return response


# ✅ 2️⃣ 再 trace（最外层）
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    request.state.trace_id = str(uuid.uuid4())
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "service": "router-classifier"}


@app.get("/metrics")
def metrics():
    """📊 Prometheus 指标端点

    可访问：http://localhost:8080/metrics
    """
    return get_metrics()

@app.post("/route")
async def route(req: RouteRequest, request: Request):
    start_time = time.time()
    trace_id = request.state.trace_id

    # ✅ 指标：记录调用
    llm_calls_total.labels(model="classifier", endpoint="route").inc()
    llm_active_requests.labels(model="classifier", endpoint="route").inc()

    try:
        text = req.message.strip()
        if not text:
            result = {
                "route": "chat",
                "confidence": 0.3,
                "source": "invalid_input"
            }
            elapsed = time.time() - start_time
            llm_latency_seconds.labels(
                model="classifier",
                endpoint="route",
                status="success"
            ).observe(elapsed)
            return result

        result = route_text(text, trace_id)

        # ✅ 指标：记录成功
        elapsed = time.time() - start_time
        llm_latency_seconds.labels(
            model="classifier",
            endpoint="route",
            status="success"
        ).observe(elapsed)

        return result
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        llm_errors_total.labels(
            model="classifier",
            endpoint="route",
            error_type=error_type
        ).inc()
        llm_latency_seconds.labels(
            model="classifier",
            endpoint="route",
            status="error"
        ).observe(elapsed)
        raise
    finally:
        llm_active_requests.labels(model="classifier", endpoint="route").dec()


@app.post("/route/decision")
async def route_decision(req: RouteDecisionRequest, request: Request):
    start_time = time.time()
    trace_id = request.state.trace_id

    llm_calls_total.labels(model="classifier", endpoint="route_decision").inc()
    llm_active_requests.labels(model="classifier", endpoint="route_decision").inc()

    try:
        messages = None
        if req.messages is not None:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
        message = req.message.strip() if isinstance(req.message, str) else req.message

        if not message and not messages:
            result = {
                "route": "chat",
                "model": "gemma4:e4b",
                "confidence": 0.3,
                "source": "invalid_input",
                "needs_rag": False,
                "prompt_policy": "memory_aware_chat",
                "context_used": {
                    "routing_text": "",
                    "recent_messages": [],
                    "memory": {"available": False, "keys": [], "code_bias": False},
                    "rag_reason": "not_required",
                },
            }
        else:
            result = build_route_decision(message=message, messages=messages, trace_id=trace_id)

        llm_latency_seconds.labels(
            model="classifier",
            endpoint="route_decision",
            status="success",
        ).observe(time.time() - start_time)
        return result
    except Exception as e:
        error_type = type(e).__name__
        llm_errors_total.labels(
            model="classifier",
            endpoint="route_decision",
            error_type=error_type,
        ).inc()
        llm_latency_seconds.labels(
            model="classifier",
            endpoint="route_decision",
            status="error",
        ).observe(time.time() - start_time)
        raise
    finally:
        llm_active_requests.labels(model="classifier", endpoint="route_decision").dec()


@app.post("/route/feedback")
async def route_feedback(req: RouteFeedbackRequest, request: Request):
    trace_id = request.state.trace_id
    sample = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "trace_id": trace_id,
        "message": req.message.strip(),
        "expected_route": req.expected_route,
        "actual_route": req.actual_route,
        "source": req.source or "manual",
        "comment": req.comment,
        "instruction": f"分类以下输入：{req.message.strip()}",
        "output": req.expected_route,
    }
    _append_feedback_sample(sample)
    log_event({
        "type": "router_feedback",
        "message": sample["message"],
        "expected_route": sample["expected_route"],
        "actual_route": sample["actual_route"],
        "source": sample["source"],
        "stored_path": str(FEEDBACK_PATH),
    }, trace_id)
    return {"ok": True, "stored": True}
