from __future__ import annotations

import json
from pathlib import Path

from .types import RouteLabel, RouteSource


IDENTITY_SCHEMA = "classifier-artifact-v1"


class ConfidenceCalibrator:
    def __init__(
        self,
        path: Path,
        *,
        classifier_model: str,
        classifier_version: str,
        prompt_version: str,
        prompt_digest: str,
        rule_version: str,
        rule_digest: str,
        weak_fallback_threshold: float,
        weak_fallback_margin: float,
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {"validated": False, "precision": {}}
        self.version = str(payload.get("version", "unvalidated"))
        self.expected_model_digest = str(payload.get("model_digest", ""))
        self.installed_model_digest: str | None = None
        expected = {
            "identity_schema": IDENTITY_SCHEMA,
            "classifier_model": classifier_model,
            "classifier_version": classifier_version,
            "prompt_version": prompt_version,
            "prompt_digest": prompt_digest,
            "rule_version": rule_version,
            "rule_digest": rule_digest,
            "weak_fallback_threshold": weak_fallback_threshold,
            "weak_fallback_margin": weak_fallback_margin,
        }
        self.metadata_matches = all(
            payload.get(key) == value for key, value in expected.items()
        )
        self.receipt_validated = bool(payload.get("validated", False))
        self._hard_rules_requested = bool(payload.get("hard_rules_enabled", False))
        self._weak_fallback_requested = bool(
            payload.get("weak_fallback_enabled", False)
        )
        self.precision = payload.get("precision", {})

    def bind_model_digest(self, digest: str | None) -> bool:
        self.installed_model_digest = digest or None
        return self.model_digest_matches

    @property
    def model_digest_matches(self) -> bool:
        return bool(
            self.expected_model_digest
            and self.installed_model_digest == self.expected_model_digest
        )

    @property
    def validated(self) -> bool:
        return bool(
            self.receipt_validated
            and self.metadata_matches
            and self.model_digest_matches
        )

    @property
    def identity_valid(self) -> bool:
        return self.validated

    @property
    def hard_rules_enabled(self) -> bool:
        return self.validated and self._hard_rules_requested

    @property
    def weak_fallback_enabled(self) -> bool:
        return self.validated and self._weak_fallback_requested

    def get(self, source: RouteSource, label: RouteLabel) -> float | None:
        if not self.validated:
            return None
        raw = self.precision.get(source.value, {}).get(label.value)
        if raw is None:
            return None
        return round(float(raw), 4)
