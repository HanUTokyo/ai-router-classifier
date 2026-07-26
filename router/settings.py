from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerSettings(FrozenModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = ""


class OllamaSettings(FrozenModel):
    base_url: str = "http://127.0.0.1:11434"
    connect_timeout_seconds: float = Field(default=3.0, gt=0)
    read_timeout_seconds: float = Field(default=120.0, gt=0)
    stream_timeout_seconds: float = Field(default=300.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")


class ClassifierSettings(FrozenModel):
    version: str = "1.0.0"
    model: str = "qwen2.5:0.5b"
    candidates: list[str] = Field(default_factory=list)
    prompt_version: str = "classifier-v1"
    rules_path: Path = Path("config/rules.yaml")
    calibration_path: Path = Path("config/calibration.json")
    weak_fallback_threshold: float = Field(default=2.0, ge=0)
    weak_fallback_margin: float = Field(default=0.75, ge=0)
    max_model_attempts: int = Field(default=2, ge=1, le=2)
    strict_model_check: bool = True


class GatewaySettings(FrozenModel):
    public_model_id: str = "local-router"
    think: bool = False
    chat_model: str
    reason_model: str
    code_model: str
    tool_model: str
    tool_supported_models: list[str] = Field(default_factory=list)


class StorageSettings(FrozenModel):
    feedback_path: Path = Path("data/feedback.jsonl")
    review_queue_path: Path = Path("data/review_queue.jsonl")
    capture_review_text: bool = True
    memory_enabled: bool = False
    memory_path: Path = Path("data/memory.md")


class LoggingSettings(FrozenModel):
    level: str = "INFO"
    include_content: bool = False


class Settings(FrozenModel):
    version: str = "1.0"
    server: ServerSettings
    ollama: OllamaSettings
    classifier: ClassifierSettings
    gateway: GatewaySettings
    storage: StorageSettings
    logging: LoggingSettings
    config_path: Path

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        selected = Path(
            config_path
            or os.getenv("AI_ROUTER_CONFIG", PROJECT_ROOT / "config/router.yaml")
        ).expanduser()
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
        with selected.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        _apply_environment_overrides(raw)
        raw["config_path"] = selected
        settings = cls.model_validate(raw)
        settings = settings.model_copy(
            update={
                "classifier": settings.classifier.model_copy(
                    update={
                        "rules_path": _absolute(settings.classifier.rules_path),
                        "calibration_path": _absolute(settings.classifier.calibration_path),
                    }
                ),
                "storage": settings.storage.model_copy(
                    update={
                        "feedback_path": _absolute(settings.storage.feedback_path),
                        "review_queue_path": _absolute(settings.storage.review_queue_path),
                        "memory_path": _absolute(settings.storage.memory_path),
                    }
                ),
            }
        )
        if not _is_loopback(settings.server.host) and not settings.server.api_key:
            raise ValueError(
                "AI_ROUTER_API_KEY is required when binding to a non-loopback host"
            )
        return settings

    def route_model(self, label: str) -> str:
        return {
            "code": self.gateway.code_model,
            "reason": self.gateway.reason_model,
            "chat": self.gateway.chat_model,
        }.get(label, self.gateway.chat_model)

    @property
    def required_models(self) -> set[str]:
        return {
            self.classifier.model,
            self.gateway.chat_model,
            self.gateway.reason_model,
            self.gateway.code_model,
            self.gateway.tool_model,
        }


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_environment_overrides(raw: dict[str, Any]) -> None:
    mappings: dict[str, tuple[str, str, Any]] = {
        "AI_ROUTER_BIND_HOST": ("server", "host", str),
        "AI_ROUTER_PORT": ("server", "port", int),
        "AI_ROUTER_API_KEY": ("server", "api_key", str),
        "AI_ROUTER_OLLAMA_BASE_URL": ("ollama", "base_url", str),
        "AI_ROUTER_CLASSIFIER_MODEL": ("classifier", "model", str),
        "AI_ROUTER_STRICT_MODEL_CHECK": (
            "classifier",
            "strict_model_check",
            _parse_bool,
        ),
        "AI_ROUTER_CHAT_MODEL": ("gateway", "chat_model", str),
        "AI_ROUTER_REASON_MODEL": ("gateway", "reason_model", str),
        "AI_ROUTER_CODE_MODEL": ("gateway", "code_model", str),
        "AI_ROUTER_TOOL_MODEL": ("gateway", "tool_model", str),
        "AI_ROUTER_MEMORY_ENABLED": ("storage", "memory_enabled", _parse_bool),
    }
    for env_name, (section, key, converter) in mappings.items():
        if env_name not in os.environ:
            continue
        raw.setdefault(section, {})[key] = converter(os.environ[env_name])
