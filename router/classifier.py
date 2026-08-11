from __future__ import annotations

import time

from .calibration import ConfidenceCalibrator
from .normalization import normalize_classification_text
from .ollama import OllamaClient, OllamaError
from .prompting import (
    PROMPT_DIGEST,
    SYSTEM_PROMPT,
    ClassificationPromptBuilder,
)
from .rule_lifecycle import load_shadow_rule_engine
from .rules import RuleEngine, RuleEvidence
from .settings import Settings
from .storage import SecureJsonlStore, content_hash
from .types import Message, RouteDecision, RouteLabel, RouteSource


class RouterClassifier:
    def __init__(
        self,
        settings: Settings,
        ollama: OllamaClient,
        *,
        rule_engine: RuleEngine | None = None,
        activate_provisional_hard_rules: bool = False,
    ):
        self.settings = settings
        self.ollama = ollama
        self.rules = rule_engine or RuleEngine(settings.classifier.rules_path)
        self.review_store = SecureJsonlStore(settings.storage.review_queue_path)
        self.shadow_rules, self.shadow_rules_error = load_shadow_rule_engine(
            active_rules_path=settings.classifier.rules_path,
            report_path=settings.classifier.shadow_report_path,
        )
        self.prompt_builder = ClassificationPromptBuilder(self.rules)
        self.prompt_digest = self.prompt_builder.assets.digest
        self.calibrator = ConfidenceCalibrator(
            settings.classifier.calibration_path,
            classifier_model=settings.classifier.model,
            classifier_version=settings.classifier.version,
            prompt_version=settings.classifier.prompt_version,
            prompt_digest=self.prompt_digest,
            rule_version=self.rules.version,
            rule_digest=self.rules.digest,
            weak_fallback_threshold=settings.classifier.weak_fallback_threshold,
            weak_fallback_margin=settings.classifier.weak_fallback_margin,
        )
        self.activate_provisional_hard_rules = activate_provisional_hard_rules

        # Kept as compatibility attributes for callers that inspect selected examples.
        self.few_shots = self.prompt_builder.few_shots
        self.contextual_few_shots = self.prompt_builder.contextual_few_shots
        self.followup_action_few_shots = (
            self.prompt_builder.followup_action_few_shots
        )

    def bind_model_digest(self, digest: str | None) -> bool:
        """Bind the installed classifier artifact before enabling calibration."""

        return self.calibrator.bind_model_digest(digest)

    async def classify(
        self,
        message: str,
        context: list[Message] | None = None,
        *,
        force_route: RouteLabel | None = None,
        model_override: str | None = None,
    ) -> RouteDecision:
        started = time.perf_counter()
        text = message.strip()
        classification_text, route_instruction_removed = (
            normalize_classification_text(text)
        )
        evidence = self.rules.evaluate(classification_text)
        self._record_shadow_difference(text, evidence)

        if force_route is not None:
            return self._decision(
                started=started,
                route=force_route,
                source=RouteSource.FORCED,
                evidence=evidence,
                classifier_model=None,
                degraded=False,
            )

        hard_label = evidence.unique_hard_label
        hard_rule_is_eligible = (
            self.calibrator.hard_rules_enabled
            or self.activate_provisional_hard_rules
        )
        if hard_label is not None and hard_rule_is_eligible:
            return self._decision(
                started=started,
                route=hard_label,
                source=RouteSource.HARD_RULE,
                evidence=evidence,
                classifier_model=None,
                degraded=not self.calibrator.validated,
            )

        model = model_override or self.settings.classifier.model
        selected_context = context or []
        try:
            route = await self.ollama.classify(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self.prompt_builder.model_prompt(
                    classification_text,
                    selected_context,
                    evidence,
                    route_instruction_removed=route_instruction_removed,
                ),
                few_shots=self.prompt_builder.select_few_shots(
                    classification_text,
                    selected_context,
                    evidence,
                ),
                max_attempts=self.settings.classifier.max_model_attempts,
            )
            top_weak = self._top_weak_label(evidence)
            source = (
                RouteSource.SMALL_MODEL_RULE_AGREE
                if top_weak == route
                else RouteSource.SMALL_MODEL
            )
            if top_weak is not None and top_weak != route:
                self._record_disagreement(text, evidence, route, model)
            return self._decision(
                started=started,
                route=route,
                source=source,
                evidence=evidence,
                classifier_model=model,
                degraded=not self.calibrator.validated,
            )
        except OllamaError as exc:
            fallback = (
                self._weak_fallback(evidence)
                if self.calibrator.weak_fallback_enabled
                else None
            )
            if fallback is not None:
                return self._decision(
                    started=started,
                    route=fallback,
                    source=RouteSource.WEAK_RULE_FALLBACK,
                    evidence=evidence,
                    classifier_model=model,
                    degraded=True,
                    classifier_error_type=type(exc).__name__,
                )
            self.review_store.append(
                {
                    "kind": "classifier_failure",
                    "message_hash": content_hash(text),
                    "message": (
                        text if self.settings.storage.capture_review_text else None
                    ),
                    "error_type": type(exc).__name__,
                    "rule_version": self.rules.version,
                    "model": model,
                }
            )
            return self._decision(
                started=started,
                route=RouteLabel.CHAT,
                source=RouteSource.DEFAULT_FALLBACK,
                evidence=evidence,
                classifier_model=model,
                degraded=True,
                classifier_error_type=type(exc).__name__,
            )

    def _decision(
        self,
        *,
        started: float,
        route: RouteLabel,
        source: RouteSource,
        evidence: RuleEvidence,
        classifier_model: str | None,
        degraded: bool,
        classifier_error_type: str | None = None,
    ) -> RouteDecision:
        return RouteDecision(
            route=route,
            confidence=self.calibrator.get(source, route),
            source=source,
            rule_hits=evidence.hard_hits + evidence.weak_hits,
            classifier_model=classifier_model,
            classifier_error_type=classifier_error_type,
            classifier_version=(
                f"{self.settings.classifier.version}+rules-{self.rules.version}"
                f"-{self.rules.digest[:12]}"
                f"+prompt-{self.settings.classifier.prompt_version}"
                f"-{self.prompt_digest[:12]}"
                f"+cal-{self.calibrator.version}"
            ),
            degraded=degraded,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _weak_fallback(self, evidence: RuleEvidence) -> RouteLabel | None:
        ranked = sorted(
            evidence.weak_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            return None
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            top_score >= self.settings.classifier.weak_fallback_threshold
            and top_score - second_score
            >= self.settings.classifier.weak_fallback_margin
        ):
            return top_label
        return None

    @staticmethod
    def _top_weak_label(evidence: RuleEvidence) -> RouteLabel | None:
        if not evidence.weak_scores:
            return None
        ranked = sorted(
            evidence.weak_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def _record_disagreement(
        self,
        text: str,
        evidence: RuleEvidence,
        model_route: RouteLabel,
        model: str,
    ) -> None:
        self.review_store.append(
            {
                "kind": "model_rule_disagreement",
                "message_hash": content_hash(text),
                "message": (
                    text if self.settings.storage.capture_review_text else None
                ),
                "model_route": model_route.value,
                "weak_scores": {
                    label.value: score
                    for label, score in evidence.weak_scores.items()
                },
                "rule_hits": [
                    hit.model_dump(mode="json") for hit in evidence.weak_hits
                ],
                "rule_version": self.rules.version,
                "model": model,
            }
        )

    def _record_shadow_difference(
        self,
        text: str,
        active: RuleEvidence,
    ) -> None:
        if self.shadow_rules is None:
            return
        shadow = self.shadow_rules.evaluate(text)
        active_hits = sorted(
            (hit.rule_id, hit.label.value) for hit in active.hard_hits
        )
        shadow_hits = sorted(
            (hit.rule_id, hit.label.value) for hit in shadow.hard_hits
        )
        if (
            active.unique_hard_label == shadow.unique_hard_label
            and active_hits == shadow_hits
        ):
            return
        self.review_store.append(
            {
                "kind": "shadow_rule_difference",
                "message_hash": content_hash(text),
                "message": (
                    text if self.settings.storage.capture_review_text else None
                ),
                "active_rule_version": active.version,
                "active_rule_digest": self.rules.digest,
                "shadow_rule_version": shadow.version,
                "shadow_rule_digest": self.shadow_rules.digest,
                "active_hard_route": (
                    active.unique_hard_label.value
                    if active.unique_hard_label is not None
                    else None
                ),
                "shadow_hard_route": (
                    shadow.unique_hard_label.value
                    if shadow.unique_hard_label is not None
                    else None
                ),
                "active_hard_hits": [
                    hit.model_dump(mode="json") for hit in active.hard_hits
                ],
                "shadow_hard_hits": [
                    hit.model_dump(mode="json") for hit in shadow.hard_hits
                ],
            }
        )


__all__ = [
    "ConfidenceCalibrator",
    "PROMPT_DIGEST",
    "RouterClassifier",
    "SYSTEM_PROMPT",
    "normalize_classification_text",
]
