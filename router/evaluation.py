from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .classifier import RouterClassifier
from .types import Message, RouteLabel


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    text: str
    expected_route: RouteLabel
    language: str
    split: str
    tags: tuple[str, ...]
    verified: bool
    context: tuple[Message, ...] = ()


@dataclass(frozen=True)
class Prediction:
    case: EvaluationCase
    predicted_route: RouteLabel
    source: str
    degraded: bool
    latency_ms: float
    classifier_error_type: str | None = None


def load_cases(
    path: Path,
    *,
    split: str | None = None,
    allow_unverified: bool = False,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
                verified = item.get("verified", False)
                if not isinstance(verified, bool):
                    raise TypeError("verified must be a JSON boolean")
                case_id = str(item["id"])
                if case_id in seen_ids:
                    raise ValueError(f"duplicate case id {case_id!r}")
                seen_ids.add(case_id)
                content_fingerprint = json.dumps(
                    {
                        "text": str(item["text"]).strip(),
                        "context": item.get("context", []),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if content_fingerprint in seen_content:
                    raise ValueError("duplicate text/context case")
                seen_content.add(content_fingerprint)
                review_result = item.get("review_result")
                if review_result not in {None, "correct", "incorrect"}:
                    raise ValueError("invalid review_result")
                if review_result == "correct" and not verified:
                    raise ValueError("correct review must be verified")
                if review_result == "incorrect" and verified:
                    raise ValueError("incorrect review cannot be verified")
                case = EvaluationCase(
                    id=case_id,
                    text=str(item["text"]),
                    expected_route=RouteLabel(item["expected_route"]),
                    language=str(item["language"]),
                    split=str(item["split"]),
                    tags=tuple(str(tag) for tag in item.get("tags", [])),
                    verified=verified,
                    context=tuple(
                        Message.model_validate(message)
                        for message in item.get("context", [])
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid dataset row at {path}:{line_number}: {exc}"
                ) from exc
            if split is None or case.split == split:
                cases.append(case)
    if not cases:
        raise ValueError(f"No cases found for split={split!r}")
    unverified = [case.id for case in cases if not case.verified]
    if unverified and not allow_unverified:
        preview = ", ".join(unverified[:5])
        raise ValueError(
            f"Dataset contains {len(unverified)} unverified cases ({preview}). "
            "Human review is required; use --allow-unverified only for diagnostics."
        )
    return cases


async def predict_cases(
    classifier: RouterClassifier,
    cases: Iterable[EvaluationCase],
    *,
    model_override: str | None = None,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for case in cases:
        decision = await classifier.classify(
            case.text,
            list(case.context),
            model_override=model_override,
        )
        predictions.append(
            Prediction(
                case=case,
                predicted_route=decision.route,
                source=decision.source.value,
                degraded=decision.degraded,
                latency_ms=decision.latency_ms,
                classifier_error_type=decision.classifier_error_type,
            )
        )
    return predictions


def score_predictions(predictions: list[Prediction]) -> dict[str, Any]:
    labels = list(RouteLabel)
    confusion: dict[str, dict[str, int]] = {
        expected.value: {predicted.value: 0 for predicted in labels}
        for expected in labels
    }
    source_counts: Counter[str] = Counter()
    tag_counts: defaultdict[str, list[bool]] = defaultdict(list)
    latencies: list[float] = []
    hard_latencies: list[float] = []
    small_model_latencies: list[float] = []
    classifier_errors: Counter[str] = Counter()
    model_attempts = 0
    hard_total = 0
    hard_correct = 0
    degraded = 0
    misclassified_cases: list[dict[str, Any]] = []
    conflict_samples: list[dict[str, Any]] = []

    for prediction in predictions:
        expected = prediction.case.expected_route
        actual = prediction.predicted_route
        correct = expected == actual
        case_result = {
            "id": prediction.case.id,
            "expected": expected.value,
            "predicted": actual.value,
            "source": prediction.source,
            "tags": list(prediction.case.tags),
        }
        if not correct:
            misclassified_cases.append(case_result)
        if "rule_conflict" in prediction.case.tags:
            conflict_samples.append({**case_result, "correct": correct})
        confusion[expected.value][actual.value] += 1
        source_counts[prediction.source] += 1
        latencies.append(prediction.latency_ms)
        degraded += int(prediction.degraded)
        if prediction.source == "hard_rule":
            hard_total += 1
            hard_correct += int(correct)
            hard_latencies.append(prediction.latency_ms)
        else:
            model_attempts += 1
            small_model_latencies.append(prediction.latency_ms)
        if prediction.classifier_error_type:
            classifier_errors[prediction.classifier_error_type] += 1
        for tag in prediction.case.tags:
            tag_counts[tag].append(correct)

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        name = label.value
        tp = confusion[name][name]
        fp = sum(confusion[other.value][name] for other in labels if other != label)
        fn = sum(confusion[name][other.value] for other in labels if other != label)
        support = sum(confusion[name].values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    latencies_sorted = sorted(latencies)
    macro_f1 = sum(f1_values) / len(f1_values)
    hard_precision = _safe_div(hard_correct, hard_total)
    hard_path_latency = _latency_summary(hard_latencies)
    small_model_path_latency = _latency_summary(small_model_latencies)
    classifier_error_count = sum(classifier_errors.values())
    classifier_error_rate = _safe_div(classifier_error_count, model_attempts)
    format_error_rate = _safe_div(
        classifier_errors["ClassifierOutputError"],
        model_attempts,
    )
    tag_accuracy = {
        tag: round(_safe_div(sum(values), len(values)), 4)
        for tag, values in sorted(tag_counts.items())
    }
    critical_slices = (
        "prompt_injection",
        "context_reference",
        "rule_conflict",
        "technical_nontechnical",
        "implementation_after_analysis",
    )
    critical_slice_accuracy = {
        tag: tag_accuracy[tag]
        for tag in critical_slices
        if tag in tag_accuracy
    }
    quality_gate = {
        "macro_f1_at_least_0_92": macro_f1 >= 0.92,
        "each_class_recall_at_least_0_88": all(
            float(metrics["recall"]) >= 0.88 for metrics in per_class.values()
        ),
        "hard_rule_precision_at_least_0_98": (
            hard_total > 0 and hard_precision >= 0.98
        ),
        "hard_rule_p95_below_10ms": (
            bool(hard_latencies) and hard_path_latency["p95"] < 10
        ),
        "small_model_p95_at_most_2000ms": (
            bool(small_model_latencies)
            and small_model_path_latency["p95"] <= 2000
        ),
        "format_error_rate_below_0_005": format_error_rate < 0.005,
        "critical_slice_accuracy_at_least_0_80": all(
            accuracy >= 0.80
            for accuracy in critical_slice_accuracy.values()
        ),
    }
    quality_gate["passed"] = all(quality_gate.values())
    weak_fallback_count = source_counts["weak_rule_fallback"]
    default_fallback_count = source_counts["default_fallback"]
    fallback_count = weak_fallback_count + default_fallback_count

    return {
        "cases": len(predictions),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "hard_rule": {
            "coverage": round(_safe_div(hard_total, len(predictions)), 4),
            "matches": hard_total,
            "precision": round(hard_precision, 4),
        },
        "degraded_rate": round(_safe_div(degraded, len(predictions)), 4),
        "fallback": {
            "count": fallback_count,
            "rate": round(_safe_div(fallback_count, len(predictions)), 4),
            "weak_rule_count": weak_fallback_count,
            "default_count": default_fallback_count,
        },
        "source_counts": dict(sorted(source_counts.items())),
        "latency_ms": {
            "p50": round(median(latencies_sorted), 3),
            "p95": round(_percentile(latencies_sorted, 0.95), 3),
            "max": round(max(latencies_sorted), 3),
        },
        "path_latency_ms": {
            "hard_rule": hard_path_latency,
            "small_model": small_model_path_latency,
        },
        "classifier_errors": {
            "attempted": model_attempts,
            "count": classifier_error_count,
            "rate": round(classifier_error_rate, 4),
            "format_error_rate": round(format_error_rate, 4),
            "by_type": dict(sorted(classifier_errors.items())),
        },
        "tag_accuracy": tag_accuracy,
        "critical_slice_accuracy": critical_slice_accuracy,
        "misclassified_cases": misclassified_cases[:100],
        "conflict_samples": conflict_samples[:100],
        "quality_gate": quality_gate,
    }


def build_calibration(
    predictions: list[Prediction],
    *,
    version: str = "gold-test-v1",
) -> dict[str, Any]:
    totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
    correct: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for prediction in predictions:
        source = prediction.source
        predicted = prediction.predicted_route.value
        totals[source][predicted] += 1
        if prediction.predicted_route == prediction.case.expected_route:
            correct[source][predicted] += 1
    precision: dict[str, dict[str, float]] = {}
    for source, label_totals in totals.items():
        precision[source] = {}
        for label, total in label_totals.items():
            precision[source][label] = round(correct[source][label] / total, 4)
    hard_total = sum(totals["hard_rule"].values())
    hard_correct = sum(correct["hard_rule"].values())
    hard_precision = _safe_div(hard_correct, hard_total)
    return {
        "version": version,
        "validated": True,
        "hard_rules_enabled": hard_total > 0 and hard_precision >= 0.98,
        "hard_rule_precision": round(hard_precision, 4),
        "precision": precision,
    }


def validate_locked_dataset_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Locked-test manifest is missing or invalid: {manifest_path}"
        ) from exc
    required = {
        "dataset_id",
        "purpose",
        "status",
        "data_file",
        "case_count",
        "sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Locked-test manifest is missing fields: {missing}")
    if manifest["purpose"] != "locked_test" or manifest["status"] != "frozen":
        raise ValueError(
            "Calibration requires purpose=locked_test and status=frozen"
        )
    if manifest["data_file"] != path.name:
        raise ValueError("Locked-test manifest points to a different data file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if manifest["sha256"] != digest:
        raise ValueError("Locked-test dataset hash does not match its manifest")
    case_count = sum(bool(line.strip()) for line in raw.splitlines())
    if manifest["case_count"] != case_count:
        raise ValueError("Locked-test case count does not match its manifest")
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return values[index]


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": round(median(ordered), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(max(ordered), 3),
    }
