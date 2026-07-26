from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from router.settings import Settings


def test_non_loopback_bind_requires_api_key(tmp_path: Path):
    source = Settings.load().model_dump(mode="json")
    source.pop("config_path")
    source["server"]["host"] = "0.0.0.0"
    source["server"]["api_key"] = ""
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match="API_KEY"):
        Settings.load(path)


def test_unknown_config_keys_are_rejected(tmp_path: Path):
    source = Settings.load().model_dump(mode="json")
    source.pop("config_path")
    source["classifier"]["model_typo"] = "bad"
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match="model_typo"):
        Settings.load(path)
