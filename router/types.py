from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RouteLabel(str, Enum):
    CODE = "code"
    REASON = "reason"
    CHAT = "chat"


class RouteSource(str, Enum):
    HARD_RULE = "hard_rule"
    SMALL_MODEL = "small_model"
    SMALL_MODEL_RULE_AGREE = "small_model_rule_agree"
    WEAK_RULE_FALLBACK = "weak_rule_fallback"
    DEFAULT_FALLBACK = "default_fallback"
    FORCED = "forced"
    LEGACY_BASELINE = "legacy_baseline"


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1, max_length=32)
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class RouteRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    context: list[Message] = Field(default_factory=list, max_length=6)


class RuleHit(BaseModel):
    rule_id: str
    label: RouteLabel
    kind: str
    weight: float


class RouteDecision(BaseModel):
    route: RouteLabel
    confidence: float | None
    source: RouteSource
    rule_hits: list[RuleHit] = Field(default_factory=list)
    classifier_model: str | None = None
    classifier_error_type: str | None = None
    classifier_version: str
    degraded: bool = False
    latency_ms: float


class FeedbackRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    expected_route: RouteLabel
    actual_route: RouteLabel | None = None
    tags: list[str] = Field(default_factory=list, max_length=20)
    comment: str | None = Field(default=None, max_length=2_000)


class SmallModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: RouteLabel
