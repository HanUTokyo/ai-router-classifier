from __future__ import annotations

import json

import pytest

from router.classifier import RouterClassifier
from router.legacy_baseline import LegacyBaselineClassifier
from router.ollama import OllamaTimeout, OllamaUnavailable
from router.types import Message, RouteLabel, RouteSource


class FakeOllama:
    def __init__(self, route: RouteLabel = RouteLabel.CHAT, error: Exception | None = None):
        self.route = route
        self.error = error
        self.calls = []

    async def classify(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.route


@pytest.mark.asyncio
async def test_calibrated_hard_rule_short_circuits_small_model(settings):
    settings.classifier.calibration_path.write_text(
        json.dumps(
            {
                "version": "test",
                "validated": True,
                "classifier_model": settings.classifier.model,
                "classifier_version": settings.classifier.version,
                "prompt_version": settings.classifier.prompt_version,
                "rule_version": "1.2.0",
                "weak_fallback_threshold": (
                    settings.classifier.weak_fallback_threshold
                ),
                "weak_fallback_margin": settings.classifier.weak_fallback_margin,
                "hard_rules_enabled": True,
                "weak_fallback_enabled": True,
                "hard_rule_precision": 1.0,
                "precision": {"hard_rule": {"code": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    ollama = FakeOllama(route=RouteLabel.CHAT)
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("请帮我修复这段 Python 代码")

    assert result.route == RouteLabel.CODE
    assert result.source == RouteSource.HARD_RULE
    assert result.classifier_model is None
    assert ollama.calls == []


@pytest.mark.asyncio
async def test_uncalibrated_hard_rule_is_only_model_evidence(settings):
    ollama = FakeOllama(route=RouteLabel.CHAT)
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("请帮我修复这段 Python 代码")

    assert result.route == RouteLabel.CHAT
    assert result.source == RouteSource.SMALL_MODEL
    assert len(ollama.calls) == 1


@pytest.mark.asyncio
async def test_calibration_is_invalidated_when_model_metadata_drifts(settings):
    settings.classifier.calibration_path.write_text(
        json.dumps(
            {
                "version": "test",
                "validated": True,
                "classifier_model": "different-model",
                "classifier_version": settings.classifier.version,
                "prompt_version": settings.classifier.prompt_version,
                "rule_version": "1.2.0",
                "weak_fallback_threshold": (
                    settings.classifier.weak_fallback_threshold
                ),
                "weak_fallback_margin": settings.classifier.weak_fallback_margin,
                "hard_rules_enabled": True,
                "weak_fallback_enabled": True,
                "precision": {"hard_rule": {"code": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    ollama = FakeOllama(route=RouteLabel.CHAT)
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("请帮我修复这段 Python 代码")

    assert result.source == RouteSource.SMALL_MODEL
    assert result.confidence is None
    assert len(ollama.calls) == 1


@pytest.mark.asyncio
async def test_conflict_is_decided_by_small_model(settings):
    ollama = FakeOllama(route=RouteLabel.CODE)
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("为什么这段 Python 代码会报错")

    assert result.route == RouteLabel.CODE
    assert result.source in {
        RouteSource.SMALL_MODEL,
        RouteSource.SMALL_MODEL_RULE_AGREE,
    }
    assert len(ollama.calls) == 1


@pytest.mark.asyncio
async def test_model_failure_uses_calibrated_weak_threshold(settings):
    settings.classifier.calibration_path.write_text(
        json.dumps(
            {
                "version": "test",
                "validated": True,
                "classifier_model": settings.classifier.model,
                "classifier_version": settings.classifier.version,
                "prompt_version": settings.classifier.prompt_version,
                "rule_version": "1.2.0",
                "hard_rules_enabled": True,
                "weak_fallback_enabled": True,
                "weak_fallback_threshold": (
                    settings.classifier.weak_fallback_threshold
                ),
                "weak_fallback_margin": settings.classifier.weak_fallback_margin,
                "precision": {},
            }
        ),
        encoding="utf-8",
    )
    ollama = FakeOllama(error=OllamaUnavailable("offline"))
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("部署 Python 数据库单元测试")

    assert result.route == RouteLabel.CODE
    assert result.source == RouteSource.WEAK_RULE_FALLBACK
    assert result.degraded is True


@pytest.mark.asyncio
async def test_uncalibrated_weak_rules_cannot_take_over(settings):
    ollama = FakeOllama(error=OllamaUnavailable("offline"))
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("部署 Python 数据库单元测试")

    assert result.route == RouteLabel.CHAT
    assert result.source == RouteSource.DEFAULT_FALLBACK
    assert result.classifier_error_type == "OllamaUnavailable"


@pytest.mark.asyncio
async def test_model_failure_defaults_to_explicit_degraded_chat(settings):
    ollama = FakeOllama(error=OllamaUnavailable("offline"))
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("今天吃什么")

    assert result.route == RouteLabel.CHAT
    assert result.source == RouteSource.DEFAULT_FALLBACK
    assert result.degraded is True
    assert result.confidence is None
    assert result.classifier_error_type == "OllamaUnavailable"


@pytest.mark.asyncio
async def test_model_timeout_is_an_explicit_degraded_result(settings):
    ollama = FakeOllama(error=OllamaTimeout("slow"))
    classifier = RouterClassifier(settings, ollama)

    result = await classifier.classify("今天吃什么")

    assert result.route == RouteLabel.CHAT
    assert result.degraded is True
    assert result.classifier_error_type == "OllamaTimeout"


@pytest.mark.asyncio
async def test_context_only_added_for_referential_query(settings):
    ollama = FakeOllama(route=RouteLabel.CODE)
    classifier = RouterClassifier(settings, ollama)
    context = [
        Message(role="user", content="帮我写一个 Python 函数"),
        Message(role="assistant", content="def add(a, b): return a + b"),
    ]

    await classifier.classify("这个怎么改", context)

    payload = json.loads(ollama.calls[0]["user_prompt"])
    assert len(payload["recent_context"]) == 2
    assert payload["recent_context"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_legacy_baseline_preserves_known_substring_false_positive(settings):
    ollama = FakeOllama(route=RouteLabel.REASON)
    legacy = LegacyBaselineClassifier(settings, ollama)

    result = await legacy.classify("Please analyze capitalism")

    assert result.route == RouteLabel.CODE
    assert result.source == RouteSource.LEGACY_BASELINE
