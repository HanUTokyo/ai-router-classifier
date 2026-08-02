import hashlib
import json
from pathlib import Path

import pytest

from router.evaluation import (
    EvaluationCase,
    Prediction,
    build_calibration,
    load_cases,
    score_predictions,
    validate_locked_dataset_manifest,
)
from router.settings import PROJECT_ROOT
from router.types import RouteLabel


def test_release_dev_dataset_is_verified_and_manifest_bound():
    path = Path(PROJECT_ROOT / "data/router_dev_v4.jsonl")
    manifest_path = path.with_suffix(".manifest.json")
    cases = load_cases(path, split="dev")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(cases) == 329
    assert all(case.verified for case in cases)
    assert {
        route: sum(case.expected_route == route for case in cases)
        for route in RouteLabel
    } == {
        RouteLabel.CODE: 111,
        RouteLabel.REASON: 108,
        RouteLabel.CHAT: 110,
    }
    assert manifest["dataset_id"] == "router-dev-v4"
    assert manifest["purpose"] == "development"
    assert manifest["status"] == "reviewed_frozen"
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_regression_dataset_is_isolated_and_verified():
    path = Path(PROJECT_ROOT / "data/router_regression_v2.jsonl")
    manifest = json.loads(
        path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    cases = load_cases(path, split="regression")

    assert len(cases) == 270
    assert all(case.verified for case in cases)
    assert {
        route: sum(case.expected_route == route for case in cases)
        for route in RouteLabel
    } == {
        RouteLabel.CODE: 90,
        RouteLabel.REASON: 90,
        RouteLabel.CHAT: 90,
    }
    assert manifest["dataset_id"] == "router-regression-v2"
    assert manifest["purpose"] == "regression"
    assert manifest["status"] == "exposed"
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_locked_test_is_exposed_and_cannot_recalibrate():
    path = Path(PROJECT_ROOT / "data/router_locked_test_v4.jsonl")
    manifest = json.loads(
        path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    cases = load_cases(path, split="test")

    assert len(cases) == 90
    assert all(case.verified for case in cases)
    assert {
        route: sum(case.expected_route == route for case in cases)
        for route in RouteLabel
    } == {
        RouteLabel.CODE: 30,
        RouteLabel.REASON: 30,
        RouteLabel.CHAT: 30,
    }
    assert sum(bool(case.context) for case in cases) >= 9
    assert manifest["dataset_id"] == "router-locked-test-v4"
    assert manifest["status"] == "exposed_passed"
    assert manifest["evaluation_receipt"]["quality_gate_passed"] is True
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="purpose=locked_test and status=frozen"):
        validate_locked_dataset_manifest(path)


def test_unverified_dataset_is_rejected(tmp_path):
    path = tmp_path / "unverified.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "pending-1",
                "text": "hello",
                "expected_route": "chat",
                "language": "en",
                "split": "dev",
                "verified": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unverified"):
        load_cases(path, split="dev")
    assert len(load_cases(path, split="dev", allow_unverified=True)) == 1


def test_dataset_requires_real_booleans_and_unique_cases(tmp_path):
    path = tmp_path / "invalid.jsonl"
    base = {
        "id": "case-1",
        "text": "hello",
        "expected_route": "chat",
        "language": "en",
        "split": "dev",
        "verified": "false",
    }
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified must be a JSON boolean"):
        load_cases(path, allow_unverified=True)

    duplicate = {**base, "verified": False}
    path.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path, allow_unverified=True)


def test_only_frozen_locked_manifest_can_authorize_calibration(tmp_path):
    path = tmp_path / "locked.jsonl"
    payload = (
        json.dumps(
            {
                "id": "locked-1",
                "text": "hello",
                "expected_route": "chat",
                "language": "en",
                "split": "test",
                "verified": True,
            }
        )
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "router-locked-test-v2",
                "purpose": "regression",
                "status": "exposed",
                "data_file": path.name,
                "case_count": 1,
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="purpose=locked_test"):
        validate_locked_dataset_manifest(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"purpose": "locked_test", "status": "frozen"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_locked_dataset_manifest(path)["dataset_id"] == (
        "router-locked-test-v2"
    )

    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_locked_dataset_manifest(path)


def test_metrics_are_computed_without_self_labels():
    cases = [
        EvaluationCase(
            id="1",
            text="write code",
            expected_route=RouteLabel.CODE,
            language="en",
            split="test",
            tags=("unit",),
            verified=True,
        ),
        EvaluationCase(
            id="2",
            text="hello",
            expected_route=RouteLabel.CHAT,
            language="en",
            split="test",
            tags=("unit",),
            verified=True,
        ),
    ]
    predictions = [
        Prediction(cases[0], RouteLabel.CODE, "hard_rule", False, 1.0),
        Prediction(cases[1], RouteLabel.CHAT, "hard_rule", False, 1.0),
    ]

    result = score_predictions(predictions)

    assert result["confusion_matrix"]["code"]["code"] == 1
    assert result["confusion_matrix"]["chat"]["chat"] == 1
    assert result["hard_rule"]["precision"] == 1.0
    assert result["path_latency_ms"]["hard_rule"]["p95"] == 1.0
    assert result["classifier_errors"]["count"] == 0
    assert result["fallback"] == {
        "count": 0,
        "rate": 0.0,
        "weak_rule_count": 0,
        "default_count": 0,
    }

    calibration = build_calibration(predictions)
    assert calibration["hard_rules_enabled"] is True
    assert calibration["hard_rule_precision"] == 1.0


def test_fallback_rate_is_separate_from_degraded_rate():
    cases = [
        EvaluationCase(
            id=str(index),
            text=f"case {index}",
            expected_route=RouteLabel.CHAT,
            language="en",
            split="dev",
            tags=("unit",),
            verified=True,
        )
        for index in range(4)
    ]
    predictions = [
        Prediction(cases[0], RouteLabel.CHAT, "small_model", True, 1.0),
        Prediction(cases[1], RouteLabel.CHAT, "small_model", False, 1.0),
        Prediction(cases[2], RouteLabel.CHAT, "weak_rule_fallback", True, 1.0),
        Prediction(cases[3], RouteLabel.CHAT, "default_fallback", True, 1.0),
    ]

    metrics = score_predictions(predictions)

    assert metrics["degraded_rate"] == 0.75
    assert metrics["fallback"] == {
        "count": 2,
        "rate": 0.5,
        "weak_rule_count": 1,
        "default_count": 1,
    }


def test_critical_slice_gate_blocks_weak_targeted_accuracy():
    cases = [
        EvaluationCase(
            id=str(index),
            text=f"case {index}",
            expected_route=RouteLabel.CHAT,
            language="en",
            split="dev",
            tags=("prompt_injection",),
            verified=True,
        )
        for index in range(5)
    ]
    predictions = [
        Prediction(
            case,
            RouteLabel.CHAT if index < 3 else RouteLabel.CODE,
            "small_model",
            False,
            1.0,
        )
        for index, case in enumerate(cases)
    ]

    metrics = score_predictions(predictions)

    assert metrics["critical_slice_accuracy"]["prompt_injection"] == 0.6
    assert (
        metrics["quality_gate"]["critical_slice_accuracy_at_least_0_80"]
        is False
    )
    assert metrics["quality_gate"]["passed"] is False
