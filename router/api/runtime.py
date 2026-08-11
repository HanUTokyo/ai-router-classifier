from __future__ import annotations

from dataclasses import dataclass

from ..classifier import RouterClassifier
from ..gateway import ThinGateway
from ..ollama import OllamaClient
from ..review_ui import ReviewDatasetStore
from ..settings import Settings
from ..storage import SecureJsonlStore


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Explicit application dependencies shared by HTTP router factories."""

    settings: Settings
    ollama: OllamaClient
    classifier: RouterClassifier
    gateway: ThinGateway
    feedback_store: SecureJsonlStore
    review_store: ReviewDatasetStore
