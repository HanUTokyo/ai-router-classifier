from __future__ import annotations

from pathlib import Path

import pytest

from router.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.load()
    return base.model_copy(
        update={
            "classifier": base.classifier.model_copy(
                update={
                    "strict_model_check": False,
                    "calibration_path": tmp_path / "calibration.json",
                }
            ),
            "storage": base.storage.model_copy(
                update={
                    "feedback_path": tmp_path / "feedback.jsonl",
                    "review_queue_path": tmp_path / "review.jsonl",
                    "memory_path": tmp_path / "memory.md",
                    "memory_enabled": False,
                }
            ),
        }
    )
