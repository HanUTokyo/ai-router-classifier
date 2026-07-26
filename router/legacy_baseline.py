from __future__ import annotations

import time
from pathlib import Path

import yaml

from .classifier import SYSTEM_PROMPT
from .ollama import OllamaClient, OllamaError
from .settings import Settings
from .types import (
    Message,
    RouteDecision,
    RouteLabel,
    RouteSource,
    RuleHit,
)


class LegacyBaselineClassifier:
    """Reproduces the v1 substring/fixed-confidence policy for comparison."""

    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama
        path = Path(__file__).resolve().parent.parent / "legacy/router_v1/rules.yaml"
        self.rules = yaml.safe_load(path.read_text(encoding="utf-8"))

    async def classify(
        self,
        message: str,
        context: list[Message] | None = None,
        *,
        model_override: str | None = None,
        **_: object,
    ) -> RouteDecision:
        del context
        started = time.perf_counter()
        rule_label, rule_confidence, hits = self._rule(message)
        degraded = False
        model = model_override or self.settings.classifier.model
        try:
            model_label = await self.ollama.classify(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=message,
                max_attempts=self.settings.classifier.max_model_attempts,
            )
            model_confidence = 0.9
        except OllamaError:
            model_label = RouteLabel.CHAT
            model_confidence = 0.2
            degraded = True
        final, confidence = self._fuse(
            rule_label,
            rule_confidence,
            model_label,
            model_confidence,
        )
        return RouteDecision(
            route=final,
            confidence=confidence,
            source=RouteSource.LEGACY_BASELINE,
            rule_hits=hits,
            classifier_model=model,
            classifier_version="legacy-v1-policy",
            degraded=degraded,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _rule(
        self, text: str
    ) -> tuple[RouteLabel | None, float, list[RuleHit]]:
        lowered = text.lower()
        scores: dict[RouteLabel, float] = {}
        hits: dict[RouteLabel, list[str]] = {}
        for raw_label in ("code", "reason"):
            label = RouteLabel(raw_label)
            config = self.rules[raw_label]
            matched = [
                keyword
                for keyword in config.get("keywords", [])
                if keyword.lower() in lowered
            ]
            if matched:
                scores[label] = len(matched) * float(config.get("weight", 1.0))
                hits[label] = matched
        if scores:
            label = max(scores, key=scores.get)
            base = float(self.rules[label.value].get("confidence", 0.5))
            confidence = min(base + min(0.1 * scores[label], 0.2), 1.0)
            return (
                label,
                confidence,
                [
                    RuleHit(
                        rule_id=f"legacy:{keyword}",
                        label=label,
                        kind="hard",
                        weight=float(self.rules[label.value].get("weight", 1.0)),
                    )
                    for keyword in hits[label]
                ],
            )
        if len(lowered) < int(self.rules["chat"]["short_text_len"]):
            return (
                RouteLabel.CHAT,
                float(self.rules["chat"]["confidence"]),
                [],
            )
        return None, 0.0, []

    @staticmethod
    def _fuse(
        rule_label: RouteLabel | None,
        rule_confidence: float,
        model_label: RouteLabel,
        model_confidence: float,
    ) -> tuple[RouteLabel, float]:
        if rule_confidence >= 0.9 and rule_label is not None:
            return rule_label, rule_confidence
        if rule_label == model_label:
            return model_label, min(max(rule_confidence, model_confidence) + 0.05, 1.0)
        if rule_label is not None and rule_confidence > model_confidence:
            return rule_label, rule_confidence
        return model_label, model_confidence
