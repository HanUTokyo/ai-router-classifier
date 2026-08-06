from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .normalization import normalize_classification_text
from .rules import RuleEngine, RuleEvidence
from .types import Message, RouteLabel


CLASSIFIER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["code", "reason", "chat"],
        }
    },
    "required": ["route"],
    "additionalProperties": False,
}
CLASSIFIER_GENERATION_OPTIONS: dict[str, int] = {
    "temperature": 0,
    "num_predict": 48,
}
CLASSIFIER_RETRY_PROMPT = (
    "Your previous output was invalid. Return only the required "
    'JSON object, for example {"route":"chat"}.'
)

REFERENTIAL_HINTS = (
    "这个",
    "那个",
    "它",
    "继续",
    "接下来",
    "上面",
    "刚才",
    "之前",
    "这段",
    "这里",
    "怎么改",
    "怎么做",
    "this one",
    "that",
    "continue",
    "above",
    "previous",
    "this",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FewShotSpec(_StrictModel):
    task: str = Field(min_length=1)
    route: RouteLabel

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("few-shot task must not be blank")
        return value


class ContextMessageSpec(_StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context content must not be blank")
        return value


class ContextualFewShotSpec(FewShotSpec):
    context: tuple[ContextMessageSpec, ...] = Field(min_length=1)


class FewShotFile(_StrictModel):
    ordinary: tuple[FewShotSpec, ...] = Field(min_length=1)
    followup_action: tuple[FewShotSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def tasks_are_unique(self) -> "FewShotFile":
        _require_unique_tasks((*self.ordinary, *self.followup_action))
        return self


class ContextualFewShotFile(_StrictModel):
    contextual: tuple[ContextualFewShotSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def tasks_are_unique(self) -> "ContextualFewShotFile":
        _require_unique_tasks(self.contextual)
        return self


class PromptAssets(_StrictModel):
    system_prompt: str = Field(min_length=1)
    ordinary: tuple[FewShotSpec, ...]
    contextual: tuple[ContextualFewShotSpec, ...]
    followup_action: tuple[FewShotSpec, ...]
    digest: str


def _require_unique_tasks(items: tuple[FewShotSpec, ...]) -> None:
    tasks = [item.task for item in items]
    duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
    if duplicates:
        raise ValueError(f"duplicate few-shot tasks: {duplicates}")


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_prompt_assets(asset_root: Path | None = None) -> PromptAssets:
    if asset_root is None:
        root = files("router").joinpath("prompts")
        system_prompt = root.joinpath("system.txt").read_text(encoding="utf-8")
        few_shots_raw = root.joinpath("few_shots.yaml").read_text(encoding="utf-8")
        contextual_raw = root.joinpath("contextual_few_shots.yaml").read_text(
            encoding="utf-8"
        )
    else:
        system_prompt = (asset_root / "system.txt").read_text(encoding="utf-8")
        few_shots_raw = (asset_root / "few_shots.yaml").read_text(encoding="utf-8")
        contextual_raw = (asset_root / "contextual_few_shots.yaml").read_text(
            encoding="utf-8"
        )

    system_prompt = system_prompt.rstrip("\n")
    if not system_prompt.strip():
        raise ValueError("system prompt must not be blank")
    few_shots = FewShotFile.model_validate(yaml.safe_load(few_shots_raw) or {})
    contextual = ContextualFewShotFile.model_validate(
        yaml.safe_load(contextual_raw) or {}
    )
    _require_unique_tasks(
        (*few_shots.ordinary, *few_shots.followup_action, *contextual.contextual)
    )

    canonical = {
        "identity_schema": "classifier-artifact-v1",
        "system_prompt": system_prompt,
        "ordinary": [item.model_dump(mode="json") for item in few_shots.ordinary],
        "contextual": [
            item.model_dump(mode="json") for item in contextual.contextual
        ],
        "followup_action": [
            item.model_dump(mode="json") for item in few_shots.followup_action
        ],
        "output_schema": CLASSIFIER_OUTPUT_SCHEMA,
        "retry_prompt": CLASSIFIER_RETRY_PROMPT,
        "generation_options": CLASSIFIER_GENERATION_OPTIONS,
    }
    digest = _canonical_digest(canonical)
    return PromptAssets(
        system_prompt=system_prompt,
        ordinary=few_shots.ordinary,
        contextual=contextual.contextual,
        followup_action=few_shots.followup_action,
        digest=digest,
    )


PROMPT_ASSETS = load_prompt_assets()
SYSTEM_PROMPT = PROMPT_ASSETS.system_prompt
PROMPT_DIGEST = PROMPT_ASSETS.digest
LEGACY_PROMPT_DIGEST = _canonical_digest(
    {
        "identity_schema": "classifier-artifact-v1",
        "system_prompt": SYSTEM_PROMPT,
        "ordinary": [],
        "contextual": [],
        "followup_action": [],
        "output_schema": CLASSIFIER_OUTPUT_SCHEMA,
        "retry_prompt": CLASSIFIER_RETRY_PROMPT,
        "generation_options": CLASSIFIER_GENERATION_OPTIONS,
    }
)


class ClassificationPromptBuilder:
    def __init__(
        self,
        rules: RuleEngine,
        *,
        assets: PromptAssets = PROMPT_ASSETS,
    ):
        self.rules = rules
        self.assets = assets
        self.few_shots = self._ordinary_few_shots()
        self.contextual_few_shots = self.few_shots + self._contextual_few_shots()
        self.followup_action_few_shots = (
            self.few_shots + self._followup_action_few_shots()
        )

    def model_prompt(
        self,
        text: str,
        context: list[Message],
        evidence: RuleEvidence,
        *,
        route_instruction_removed: bool = False,
    ) -> str:
        context_payload: list[dict[str, str]] = []
        context_user_texts: list[str] = []
        if self.uses_recent_context(text, context):
            for item in context[-4:]:
                content = _normalize_content(item.content)
                context_payload.append(
                    {"role": item.role, "content": content[:2_000]}
                )
                if item.role == "user" and content.strip():
                    context_user_texts.append(content[:2_000])
        weak = {
            label.value: score for label, score in evidence.weak_scores.items()
        }
        context_evidence = self.rules.evaluate(" ".join(context_user_texts))
        payload: dict[str, Any] = {
            "task": text,
            "recent_context": context_payload,
            "referential_task": bool(context_payload),
            "weak_rule_evidence": weak,
            "hard_rule_conflict": len(evidence.hard_labels) > 1,
            "route_instruction_removed": route_instruction_removed,
        }
        if context_payload:
            payload["resolved_task"] = (
                {
                    "prior_user_request": context_user_texts[-1],
                    "current_followup": text,
                }
                if context_user_texts
                else None
            )
            payload["context_rule_evidence"] = {
                "hard_labels": sorted(
                    label.value for label in context_evidence.hard_labels
                ),
                "weak_scores": {
                    label.value: score
                    for label, score in context_evidence.weak_scores.items()
                },
            }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def uses_recent_context(text: str, context: list[Message]) -> bool:
        lowered = text.lower()
        return bool(context) and any(hint in lowered for hint in REFERENTIAL_HINTS)

    def select_few_shots(
        self,
        text: str,
        context: list[Message],
        evidence: RuleEvidence,
    ) -> list[tuple[str, RouteLabel]]:
        if self.uses_recent_context(text, context):
            return self.contextual_few_shots
        if any(
            hit.rule_id in {"weak-code-followup-action", "weak-code-action-en"}
            for hit in evidence.weak_hits
        ):
            return self.followup_action_few_shots
        return self.few_shots

    def _ordinary_few_shots(self) -> list[tuple[str, RouteLabel]]:
        return [
            (
                self.model_prompt(
                    normalized,
                    [],
                    self.rules.evaluate(normalized),
                    route_instruction_removed=removed,
                ),
                item.route,
            )
            for item in self.assets.ordinary
            for normalized, removed in [normalize_classification_text(item.task)]
        ]

    def _contextual_few_shots(self) -> list[tuple[str, RouteLabel]]:
        return [
            (
                self.model_prompt(
                    item.task,
                    [
                        Message(role=message.role, content=message.content)
                        for message in item.context
                    ],
                    self.rules.evaluate(item.task),
                ),
                item.route,
            )
            for item in self.assets.contextual
        ]

    def _followup_action_few_shots(self) -> list[tuple[str, RouteLabel]]:
        return [
            (
                self.model_prompt(
                    item.task,
                    [],
                    self.rules.evaluate(item.task),
                ),
                item.route,
            )
            for item in self.assets.followup_action
        ]


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {
                "text",
                "input_text",
            }:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)
