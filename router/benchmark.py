from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from .classifier import RouterClassifier
from .evaluation import (
    EvaluationCase,
    Prediction,
    predict_cases,
    score_predictions,
)


async def benchmark_candidates(
    classifier: RouterClassifier,
    cases: list[EvaluationCase],
    candidates: list[str],
    installed: dict[str, str],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for model in candidates:
        if model not in installed:
            results.append(
                {
                    "model": model,
                    "digest": None,
                    "status": "missing",
                }
            )
            continue
        predictions = await predict_cases(
            classifier,
            cases,
            model_override=model,
        )
        metrics = score_predictions(predictions)
        folds = _five_folds(predictions)
        fold_metrics = [score_predictions(fold) for fold in folds]
        fold_macro_f1 = [float(item["macro_f1"]) for item in fold_metrics]
        results.append(
            {
                "model": model,
                "digest": installed.get(model),
                "status": "evaluated",
                "metrics": metrics,
                "cross_validation": {
                    "folds": 5,
                    "macro_f1_mean": round(mean(fold_macro_f1), 4),
                    "macro_f1_stddev": round(pstdev(fold_macro_f1), 4),
                    "fold_metrics": fold_metrics,
                },
            }
        )

    evaluated = [item for item in results if item["status"] == "evaluated"]
    evaluated.sort(
        key=lambda item: (
            -float(item["cross_validation"]["macro_f1_mean"]),
            float(item["metrics"]["latency_ms"]["p95"]),
        )
    )
    winner = evaluated[0] if evaluated else None
    if len(evaluated) > 1:
        best_f1 = float(evaluated[0]["cross_validation"]["macro_f1_mean"])
        tied = [
            item
            for item in evaluated
            if best_f1
            - float(item["cross_validation"]["macro_f1_mean"])
            < 0.005
        ]
        tied.sort(key=lambda item: float(item["metrics"]["latency_ms"]["p95"]))
        winner = tied[0]
    passing = [
        item for item in evaluated if item["metrics"]["quality_gate"]["passed"]
    ]
    eligible = _select_ranked(passing)
    return {
        "prompt_version": classifier.settings.classifier.prompt_version,
        "rule_version": classifier.rules.version,
        "candidate_results": results,
        "provisional_winner": (
            {
                "model": winner["model"],
                "digest": winner["digest"],
                "quality_gate_passed": winner["metrics"]["quality_gate"]["passed"],
            }
            if winner
            else None
        ),
        "quality_eligible_candidate": (
            {"model": eligible["model"], "digest": eligible["digest"]}
            if eligible
            else None
        ),
        "next_action": (
            "Validate the frozen candidate on the verified locked test split."
            if eligible
            else (
                "No candidate passed. Add at least 30 human-reviewed cases "
                "to the weakest class and conflict slices, then rerun."
            )
        ),
    }


def _five_folds(predictions: list[Prediction]) -> list[list[Prediction]]:
    folds: list[list[Prediction]] = [[] for _ in range(5)]
    grouped: defaultdict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.case.expected_route.value].append(prediction)
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda item: item.case.id)
        for index, prediction in enumerate(ordered):
            folds[index % 5].append(prediction)
    return folds


def _select_ranked(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["cross_validation"]["macro_f1_mean"]),
            float(item["metrics"]["latency_ms"]["p95"]),
        ),
    )
    best_f1 = float(ordered[0]["cross_validation"]["macro_f1_mean"])
    tied = [
        item
        for item in ordered
        if best_f1 - float(item["cross_validation"]["macro_f1_mean"]) < 0.005
    ]
    return min(
        tied,
        key=lambda item: float(item["metrics"]["latency_ms"]["p95"]),
    )
