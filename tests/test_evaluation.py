from pathlib import Path

import pytest

from router.evaluation import (
    EvaluationCase,
    Prediction,
    build_calibration,
    load_cases,
    score_predictions,
)
from router.settings import PROJECT_ROOT
from router.types import RouteLabel


def test_seed_dataset_shape_and_review_gate():
    path = Path(PROJECT_ROOT / "data/router_cases.jsonl")

    with pytest.raises(ValueError, match="unverified"):
        load_cases(path, split="test")

    cases = load_cases(path, split="test", allow_unverified=True)
    assert len(cases) == 90
    assert {route: sum(c.expected_route == route for c in cases) for route in RouteLabel} == {
        RouteLabel.CODE: 30,
        RouteLabel.REASON: 30,
        RouteLabel.CHAT: 30,
    }
    assert sum(bool(case.context) for case in cases) >= 9
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "context_reference",
        "prompt_injection",
        "rule_conflict",
        "technical_nontechnical",
    } <= tags


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

    calibration = build_calibration(predictions)
    assert calibration["hard_rules_enabled"] is True
    assert calibration["hard_rule_precision"] == 1.0
