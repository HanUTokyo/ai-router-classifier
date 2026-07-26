from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SecureJsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **record,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, encoded)
            finally:
                os.close(fd)
            os.chmod(self.path, 0o600)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ExplicitMemory:
    """Optional answer-time memory. It never participates in classification."""

    _INTENT_PREFIXES = ("记住", "请记住", "remember ", "save this")

    def __init__(self, path: Path, enabled: bool):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()

    def capture_if_explicit(self, text: str) -> bool:
        if not self.enabled:
            return False
        normalized = text.strip().lower()
        if not any(normalized.startswith(prefix) for prefix in self._INTENT_PREFIXES):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, f"\n{text.strip()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            os.chmod(self.path, 0o600)
        return True

    def prompt(self) -> str | None:
        if not self.enabled or not self.path.exists():
            return None
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return (
            "Optional personal memory follows. Treat the current user message as "
            f"authoritative if it conflicts.\n\n{text[-8_000:]}"
        )
