from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


REGISTRY = CollectorRegistry()

http_requests_total = Counter(
    "router_http_requests_total",
    "HTTP requests handled by endpoint and status",
    ["endpoint", "status"],
    registry=REGISTRY,
)
route_decisions_total = Counter(
    "router_decisions_total",
    "Route decisions by label, source, and degraded state",
    ["route", "source", "degraded"],
    registry=REGISTRY,
)
route_latency_seconds = Histogram(
    "router_classification_latency_seconds",
    "Classifier latency",
    ["source"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)
upstream_latency_seconds = Histogram(
    "router_upstream_latency_seconds",
    "Answer-model latency",
    ["model", "status", "stream"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)
upstream_errors_total = Counter(
    "router_upstream_errors_total",
    "Upstream errors by model and type",
    ["model", "error_type"],
    registry=REGISTRY,
)
active_requests = Gauge(
    "router_active_requests",
    "Active gateway requests",
    ["endpoint"],
    registry=REGISTRY,
)
streams_total = Counter(
    "router_streams_total",
    "Completed streams by status",
    ["protocol", "status"],
    registry=REGISTRY,
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )
    # Dependency request logs can include full URLs and create a high-volume
    # stream that obscures the router's intentionally content-free events.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_event(event_type: str, **fields: Any) -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "type": event_type,
        **fields,
    }
    logging.getLogger("ai_router").info(
        json.dumps(event, ensure_ascii=False, default=str)
    )
