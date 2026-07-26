from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import RouteLabel, RuleHit


class RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: RouteLabel
    kind: str = Field(pattern="^(hard|weak)$")
    pattern_type: str = Field(pattern="^(exact|phrase|token|regex)$")
    pattern: str = Field(min_length=1)
    exclusions: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip()


class RuleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    rules: list[RuleSpec]

    @model_validator(mode="after")
    def require_unique_rule_ids(self) -> "RuleFile":
        ids = [rule.id for rule in self.rules]
        duplicates = sorted(
            rule_id for rule_id in set(ids) if ids.count(rule_id) > 1
        )
        if duplicates:
            raise ValueError(
                "Duplicate rule IDs: " + ", ".join(duplicates)
            )
        return self


class RuleEvidence(BaseModel):
    version: str
    hard_hits: list[RuleHit] = Field(default_factory=list)
    weak_hits: list[RuleHit] = Field(default_factory=list)
    weak_scores: dict[RouteLabel, float] = Field(default_factory=dict)

    @property
    def hard_labels(self) -> set[RouteLabel]:
        return {hit.label for hit in self.hard_hits}

    @property
    def unique_hard_label(self) -> RouteLabel | None:
        labels = self.hard_labels
        return next(iter(labels)) if len(labels) == 1 else None


class CompiledRule:
    def __init__(self, spec: RuleSpec):
        self.spec = spec
        self._regex = self._compile_pattern(spec)
        self._exclusions = [
            re.compile(pattern, re.IGNORECASE) for pattern in spec.exclusions
        ]

    @staticmethod
    def _compile_pattern(spec: RuleSpec) -> re.Pattern[str]:
        if spec.pattern_type == "exact":
            pattern = rf"^\s*{re.escape(spec.pattern)}\s*$"
        elif spec.pattern_type == "phrase":
            pattern = re.escape(spec.pattern)
        elif spec.pattern_type == "token":
            tokens = [re.escape(token.strip()) for token in spec.pattern.split("|")]
            tokens = [token for token in tokens if token]
            if not tokens:
                raise ValueError(f"Rule {spec.id!r} has no tokens")
            pattern = rf"(?<![\w])(?:{'|'.join(tokens)})(?![\w])"
        else:
            pattern = spec.pattern
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid pattern in rule {spec.id!r}: {exc}") from exc

    def matches(self, text: str) -> bool:
        if any(exclusion.search(text) for exclusion in self._exclusions):
            return False
        return self._regex.search(text) is not None


class RuleEngine:
    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        self.rule_file = RuleFile.model_validate(loaded)
        self.rules = [
            CompiledRule(spec) for spec in self.rule_file.rules if spec.enabled
        ]

    @property
    def version(self) -> str:
        return self.rule_file.version

    def evaluate(self, text: str) -> RuleEvidence:
        hard_hits: list[RuleHit] = []
        weak_hits: list[RuleHit] = []
        weak_scores: defaultdict[RouteLabel, float] = defaultdict(float)

        for rule in self.rules:
            if not rule.matches(text):
                continue
            hit = RuleHit(
                rule_id=rule.spec.id,
                label=rule.spec.label,
                kind=rule.spec.kind,
                weight=rule.spec.weight,
            )
            if rule.spec.kind == "hard":
                hard_hits.append(hit)
            else:
                weak_hits.append(hit)
                weak_scores[rule.spec.label] += rule.spec.weight

        return RuleEvidence(
            version=self.version,
            hard_hits=hard_hits,
            weak_hits=weak_hits,
            weak_scores={
                label: round(score, 4) for label, score in weak_scores.items()
            },
        )
