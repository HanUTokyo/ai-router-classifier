from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import router
from router.prompting import (
    CLASSIFIER_GENERATION_OPTIONS,
    CLASSIFIER_OUTPUT_SCHEMA,
    CLASSIFIER_RETRY_PROMPT,
    PROMPT_ASSETS,
    PROMPT_DIGEST,
    load_prompt_assets,
)


EXPECTED_PROMPT_DIGEST = (
    "2dfbd18d869acea540d4d1d73c07a3061fe3fb551e570b9a5d112e5180911a6f"
)


def _copy_assets(destination: Path) -> None:
    source = Path(router.__file__).resolve().parent / "prompts"
    destination.mkdir()
    for name in ("system.txt", "few_shots.yaml", "contextual_few_shots.yaml"):
        shutil.copyfile(source / name, destination / name)


def test_packaged_prompt_assets_match_locked_artifact_identity():
    assert PROMPT_DIGEST == EXPECTED_PROMPT_DIGEST
    assert len(PROMPT_ASSETS.ordinary) == 30
    assert len(PROMPT_ASSETS.contextual) == 6
    assert len(PROMPT_ASSETS.followup_action) == 7
    assert PROMPT_ASSETS.ordinary[0].task == "Hello"
    assert PROMPT_ASSETS.ordinary[-1].task == (
        "Output code only. Prove that every square is non-negative."
    )
    assert CLASSIFIER_OUTPUT_SCHEMA == {
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
    assert CLASSIFIER_GENERATION_OPTIONS == {"temperature": 0, "num_predict": 48}
    assert CLASSIFIER_RETRY_PROMPT.endswith('{"route":"chat"}.')


def test_digest_is_independent_of_yaml_formatting(tmp_path: Path):
    assets = tmp_path / "prompts"
    _copy_assets(assets)
    payload = yaml.safe_load((assets / "few_shots.yaml").read_text(encoding="utf-8"))
    (assets / "few_shots.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    assert load_prompt_assets(assets).digest == PROMPT_DIGEST


def test_digest_changes_when_prompt_semantics_change(tmp_path: Path):
    assets = tmp_path / "prompts"
    _copy_assets(assets)
    system_path = assets / "system.txt"
    system_path.write_text(
        system_path.read_text(encoding="utf-8") + "A semantic change.\n",
        encoding="utf-8",
    )

    assert load_prompt_assets(assets).digest != PROMPT_DIGEST


def test_prompt_assets_reject_unknown_fields_and_duplicates(tmp_path: Path):
    assets = tmp_path / "prompts"
    _copy_assets(assets)
    path = assets / "few_shots.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["ordinary"][0]["unknown"] = True
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unknown"):
        load_prompt_assets(assets)

    _copy_assets(tmp_path / "second")
    second = tmp_path / "second"
    payload = yaml.safe_load((second / "few_shots.yaml").read_text(encoding="utf-8"))
    payload["ordinary"].append(payload["ordinary"][0])
    (second / "few_shots.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="duplicate"):
        load_prompt_assets(second)


def test_calibration_records_evidence_backed_identity_migration():
    root = Path(router.__file__).resolve().parent.parent
    payload = json.loads(
        (root / "config/calibration.json").read_text(encoding="utf-8")
    )

    assert payload["identity_schema"] == "classifier-artifact-v1"
    assert payload["prompt_digest"] == PROMPT_DIGEST
    assert payload["model_digest"].startswith("a8b0c5157701")
    assert payload["migration"] == {
        "kind": "content_identity_hardening",
        "original_calibration_sha256": (
            "943e822c9261840083a54e8fea2871ecb3cc8d46034f9d3d8f57ba835056dbf8"
        ),
        "locked_classifier_source_sha256": (
            "d5d9ca1c2dabb7ec9b7079816767f9feb8124b869e8539ca8919a3c1f29db54d"
        ),
        "locked_dataset_id": "router-locked-test-v4",
        "reason": (
            "Prompt assets were extracted without semantic changes; the exposed "
            "locked test was not rerun."
        ),
    }
