from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rules import RuleEngine, RuleFile, RuleSpec

if TYPE_CHECKING:
    from .evaluation import EvaluationCase


RuleChangeAction = Literal["add", "update", "disable", "demote", "promote"]


class RuleGateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_precision: float = Field(default=0.98, ge=0.98, le=1)
    target_rule_precision: float = Field(default=0.98, ge=0.98, le=1)
    target_rule_min_matches: int = Field(default=20, ge=20)


class RuleChange(BaseModel):
    """A reviewable proposal. It never changes the active rules by itself."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    change_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9._-]+$")
    base_version: str = Field(
        min_length=1,
        pattern=r"^\d+\.\d+\.\d+(?:[-+][a-zA-Z0-9.-]+)?$",
    )
    target_version: str = Field(
        min_length=1,
        pattern=r"^\d+\.\d+\.\d+(?:[-+][a-zA-Z0-9.-]+)?$",
    )
    status: Literal["candidate"] = "candidate"
    action: RuleChangeAction
    rule_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_refs: list[str] = Field(default_factory=list)
    proposed_rule: RuleSpec | None = None
    thresholds: RuleGateThresholds = Field(default_factory=RuleGateThresholds)

    @model_validator(mode="after")
    def validate_change_shape(self) -> "RuleChange":
        if not self.reason.strip() or not self.created_by.strip():
            raise ValueError("reason and created_by must not be blank")
        if any(not reference.strip() for reference in self.evidence_refs):
            raise ValueError("evidence_refs must not contain blank values")
        if self.base_version == self.target_version:
            raise ValueError("target_version must differ from base_version")
        if _version_core(self.target_version) <= _version_core(self.base_version):
            raise ValueError("target_version must be greater than base_version")
        needs_rule = self.action in {"add", "update"}
        if needs_rule and self.proposed_rule is None:
            raise ValueError(f"{self.action} requires proposed_rule")
        if not needs_rule and self.proposed_rule is not None:
            raise ValueError(f"{self.action} must not include proposed_rule")
        if self.proposed_rule is not None and self.proposed_rule.id != self.rule_id:
            raise ValueError("proposed_rule.id must match rule_id")
        if not self.evidence_refs:
            raise ValueError("Every rule change requires at least one evidence_ref")
        return self


def load_rule_change(path: Path) -> RuleChange:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read rule change: {path}") from exc
    return RuleChange.model_validate(payload)


def write_rule_change(path: Path, change: RuleChange) -> None:
    _write_yaml_atomic(path, change.model_dump(mode="json"))


def load_shadow_rule_engine(
    *,
    active_rules_path: Path,
    report_path: Path | None,
) -> tuple[RuleEngine | None, str | None]:
    if report_path is None or not report_path.exists():
        return None, None
    try:
        report = _load_json(report_path)
        if report.get("status") != "shadow_evaluated":
            raise ValueError("Shadow report has an invalid status")
        if (
            report.get("receipts", {}).get("active_rules_sha256")
            != _sha256_file(active_rules_path)
        ):
            raise ValueError("Shadow report targets a different active rule file")
        candidate = RuleFile.model_validate(report.get("candidate_rules"))
        expected = report["receipts"]["candidate_rules_sha256"]
        if _sha256_payload(candidate.model_dump(mode="json")) != expected:
            raise ValueError("Shadow candidate hash does not match")
        return RuleEngine.from_rule_file(candidate), None
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)


def build_candidate_rule_file(
    active: RuleFile,
    change: RuleChange,
) -> RuleFile:
    if active.version != change.base_version:
        raise ValueError(
            f"Active rule version is {active.version}, "
            f"but change expects {change.base_version}"
        )
    rules = [rule.model_copy(deep=True) for rule in active.rules]
    index = {rule.id: position for position, rule in enumerate(rules)}

    if change.action == "add":
        if change.rule_id in index:
            raise ValueError(f"Rule already exists: {change.rule_id}")
        assert change.proposed_rule is not None
        rules.append(change.proposed_rule)
    else:
        if change.rule_id not in index:
            raise ValueError(f"Rule does not exist: {change.rule_id}")
        position = index[change.rule_id]
        current = rules[position]
        if change.action == "update":
            assert change.proposed_rule is not None
            if (
                change.proposed_rule.kind != current.kind
                or change.proposed_rule.enabled != current.enabled
            ):
                raise ValueError(
                    "update cannot change kind/enabled; use promote, demote, "
                    "or disable"
                )
            rules[position] = change.proposed_rule
        elif change.action == "disable":
            rules[position] = current.model_copy(update={"enabled": False})
        elif change.action == "demote":
            rules[position] = current.model_copy(update={"kind": "weak"})
        elif change.action == "promote":
            rules[position] = current.model_copy(update={"kind": "hard"})

    return RuleFile(version=change.target_version, rules=rules)


def evaluate_rule_change(
    *,
    active_rules_path: Path,
    change_path: Path,
    datasets: list[tuple[Path, str]],
) -> dict[str, Any]:
    """Evaluate a candidate in shadow mode without calling the classifier model."""

    from .evaluation import load_cases

    change = load_rule_change(change_path)
    active_engine = RuleEngine(active_rules_path)
    candidate_file = build_candidate_rule_file(active_engine.rule_file, change)
    candidate_engine = RuleEngine.from_rule_file(candidate_file)

    cases: list[EvaluationCase] = []
    dataset_receipts: list[dict[str, Any]] = []
    for dataset_path, split in datasets:
        selected = load_cases(dataset_path, split=split)
        cases.extend(selected)
        dataset_receipts.append(
            {
                "path": str(dataset_path.resolve()),
                "split": split,
                "cases": len(selected),
                "sha256": _sha256_file(dataset_path),
            }
        )
    if not cases:
        raise ValueError("At least one verified evaluation case is required")

    baseline = _rule_metrics(active_engine, cases)
    candidate = _rule_metrics(candidate_engine, cases)
    comparisons = _compare_engines(active_engine, candidate_engine, cases)
    target = candidate["per_rule"].get(
        change.rule_id,
        {"matches": 0, "correct": 0, "precision": 0.0},
    )
    target_is_hard = any(
        rule.id == change.rule_id and rule.kind == "hard" and rule.enabled
        for rule in candidate_file.rules
    )
    needs_target_gate = change.action in {"add", "update", "promote"} and target_is_hard
    gate = {
        "candidate_changes_rule_evidence": (
            comparisons["affected_rule_evidence_cases"] > 0
        ),
        "hard_precision_at_least_threshold": (
            candidate["unique_hard_decisions"] == 0
            or candidate["precision"] >= change.thresholds.hard_precision
        ),
        "no_new_hard_errors": not comparisons["new_hard_errors"],
        "no_new_hard_conflicts": not comparisons["new_hard_conflicts"],
        "no_new_regression_hard_errors": not comparisons[
            "new_regression_hard_errors"
        ],
        "target_rule_precision_at_least_threshold": (
            not needs_target_gate
            or target["precision"] >= change.thresholds.target_rule_precision
        ),
        "target_rule_has_minimum_support": (
            not needs_target_gate
            or target["matches"] >= change.thresholds.target_rule_min_matches
        ),
        "human_evidence_present": bool(change.evidence_refs),
    }
    gate["passed"] = all(gate.values())

    candidate_payload = candidate_file.model_dump(mode="json")
    return {
        "schema_version": "1",
        "status": "shadow_evaluated",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "change": change.model_dump(mode="json"),
        "receipts": {
            "active_rules_path": str(active_rules_path.resolve()),
            "active_rules_sha256": _sha256_file(active_rules_path),
            "change_path": str(change_path.resolve()),
            "change_sha256": _sha256_file(change_path),
            "candidate_rules_sha256": _sha256_payload(candidate_payload),
            "datasets": dataset_receipts,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparisons,
        "gate": gate,
        "candidate_rules": candidate_payload,
    }


def promote_rule_change(
    *,
    active_rules_path: Path,
    change_path: Path,
    report_path: Path,
    history_dir: Path,
    approved_by: str,
) -> dict[str, Any]:
    approved_by = approved_by.strip()
    if not approved_by:
        raise ValueError("approved_by must not be blank")
    change = load_rule_change(change_path)
    report = _load_json(report_path)
    report_sha256 = _sha256_file(report_path)
    _validate_promotion_receipts(
        active_rules_path=active_rules_path,
        change_path=change_path,
        report=report,
    )
    if report.get("change", {}).get("change_id") != change.change_id:
        raise ValueError("Report belongs to a different rule change")
    if not report.get("gate", {}).get("passed"):
        raise ValueError("Rule change did not pass the shadow evaluation gate")
    candidate = RuleFile.model_validate(report.get("candidate_rules"))
    candidate_payload = candidate.model_dump(mode="json")
    expected_candidate_hash = report["receipts"]["candidate_rules_sha256"]
    if _sha256_payload(candidate_payload) != expected_candidate_hash:
        raise ValueError("Candidate rules in report do not match their receipt")

    active = RuleEngine(active_rules_path).rule_file
    if active.version != change.base_version:
        raise ValueError("Active rule version changed after shadow evaluation")
    history_dir.mkdir(parents=True, exist_ok=True)
    _reject_reused_release_version(history_dir, candidate.version)
    snapshot_path = _unique_snapshot_path(
        history_dir,
        active.version,
        _sha256_file(active_rules_path),
    )
    shutil.copyfile(active_rules_path, snapshot_path)
    os.chmod(snapshot_path, 0o600)
    archive_suffix = uuid.uuid4().hex[:12]
    change_archive = history_dir / (
        f"{change.change_id}.candidate-{archive_suffix}.yaml"
    )
    report_archive = history_dir / (
        f"{change.change_id}.shadow-{archive_suffix}.json"
    )
    shutil.copyfile(change_path, change_archive)
    shutil.copyfile(report_path, report_archive)
    os.chmod(change_archive, 0o600)
    os.chmod(report_archive, 0o600)
    _write_yaml_atomic(active_rules_path, candidate_payload)

    receipt = {
        "schema_version": "1",
        "event": "promoted",
        "change_id": change.change_id,
        "from_version": active.version,
        "to_version": candidate.version,
        "approved_by": approved_by,
        "approved_at": datetime.now(UTC).isoformat(),
        "active_rules_sha256": _sha256_file(active_rules_path),
        "rule_digest": RuleEngine(active_rules_path).digest,
        "previous_snapshot": str(snapshot_path.resolve()),
        "candidate_archive": str(change_archive.resolve()),
        "shadow_report_archive": str(report_archive.resolve()),
        "change_sha256": _sha256_file(change_path),
        "report_sha256": report_sha256,
        "dataset_receipts": report["receipts"]["datasets"],
    }
    receipt_path = history_dir / (
        f"{change.change_id}.promotion-{uuid.uuid4().hex[:12]}.json"
    )
    _write_json_atomic(receipt_path, receipt)
    shadow_report_cleared = False
    if report_path.resolve() != report_archive.resolve():
        try:
            report_path.unlink()
            shadow_report_cleared = True
        except OSError:
            pass
    return {
        **receipt,
        "receipt_path": str(receipt_path.resolve()),
        "shadow_report_cleared": shadow_report_cleared,
    }


def rollback_rules(
    *,
    active_rules_path: Path,
    snapshot_path: Path,
    history_dir: Path,
    approved_by: str,
    reason: str,
) -> dict[str, Any]:
    approved_by = approved_by.strip()
    reason = reason.strip()
    if not approved_by or not reason:
        raise ValueError("approved_by and reason must not be blank")
    current_engine = RuleEngine(active_rules_path)
    target_engine = RuleEngine(snapshot_path)
    current = current_engine.rule_file
    target = target_engine.rule_file
    if (
        current.version == target.version
        and current_engine.digest == target_engine.digest
    ):
        raise ValueError("Rollback target is already active")
    history_dir.mkdir(parents=True, exist_ok=True)
    current_snapshot = _unique_snapshot_path(
        history_dir,
        current.version,
        _sha256_file(active_rules_path),
    )
    shutil.copyfile(active_rules_path, current_snapshot)
    os.chmod(current_snapshot, 0o600)
    target_bytes = snapshot_path.read_bytes()
    _write_bytes_atomic(active_rules_path, target_bytes)

    event_id = (
        datetime.now(UTC).strftime("rollback-%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:12]
    )
    receipt = {
        "schema_version": "1",
        "event": "rolled_back",
        "from_version": current.version,
        "to_version": target.version,
        "approved_by": approved_by,
        "reason": reason,
        "approved_at": datetime.now(UTC).isoformat(),
        "active_rules_sha256": _sha256_file(active_rules_path),
        "rule_digest": RuleEngine(active_rules_path).digest,
        "rollback_source": str(snapshot_path.resolve()),
        "replaced_snapshot": str(current_snapshot.resolve()),
    }
    receipt_path = history_dir / f"{event_id}.json"
    _write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path.resolve())}


def _rule_metrics(
    engine: RuleEngine,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    correct = 0
    unique = 0
    conflicts = 0
    per_rule: dict[str, Counter[str]] = {}
    for case in cases:
        evidence = engine.evaluate(case.text)
        if len(evidence.hard_labels) > 1:
            conflicts += 1
        hard_label = evidence.unique_hard_label
        if hard_label is not None:
            unique += 1
            correct += int(hard_label == case.expected_route)
        for hit in evidence.hard_hits:
            stats = per_rule.setdefault(hit.rule_id, Counter())
            stats["matches"] += 1
            stats["correct"] += int(hit.label == case.expected_route)
    return {
        "version": engine.version,
        "cases": len(cases),
        "unique_hard_decisions": unique,
        "correct": correct,
        "precision": round(correct / unique, 4) if unique else 0.0,
        "coverage": round(unique / len(cases), 4),
        "conflicts": conflicts,
        "per_rule": {
            rule_id: {
                "matches": stats["matches"],
                "correct": stats["correct"],
                "precision": round(
                    stats["correct"] / stats["matches"],
                    4,
                ),
            }
            for rule_id, stats in sorted(per_rule.items())
        },
    }


def _compare_engines(
    baseline: RuleEngine,
    candidate: RuleEngine,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    affected: list[str] = []
    evidence_affected: list[str] = []
    new_errors: list[str] = []
    fixed_errors: list[str] = []
    new_regression_errors: list[str] = []
    new_conflicts: list[str] = []
    resolved_conflicts: list[str] = []
    for case in cases:
        before = baseline.evaluate(case.text)
        after = candidate.evaluate(case.text)
        before_label = before.unique_hard_label
        after_label = after.unique_hard_label
        before_hits = tuple(
            sorted((hit.rule_id, hit.label.value, hit.kind) for hit in before.hard_hits)
        )
        after_hits = tuple(
            sorted((hit.rule_id, hit.label.value, hit.kind) for hit in after.hard_hits)
        )
        before_all_hits = tuple(
            sorted(
                (hit.rule_id, hit.label.value, hit.kind, hit.weight)
                for hit in before.hard_hits + before.weak_hits
            )
        )
        after_all_hits = tuple(
            sorted(
                (hit.rule_id, hit.label.value, hit.kind, hit.weight)
                for hit in after.hard_hits + after.weak_hits
            )
        )
        if before_label != after_label or before_hits != after_hits:
            affected.append(case.id)
        if (
            before_all_hits != after_all_hits
            or before.weak_scores != after.weak_scores
        ):
            evidence_affected.append(case.id)
        before_wrong = before_label is not None and before_label != case.expected_route
        after_wrong = after_label is not None and after_label != case.expected_route
        before_conflict = len(before.hard_labels) > 1
        after_conflict = len(after.hard_labels) > 1
        if after_wrong and not before_wrong:
            new_errors.append(case.id)
            if case.split == "regression":
                new_regression_errors.append(case.id)
        if before_wrong and not after_wrong:
            fixed_errors.append(case.id)
        if after_conflict and not before_conflict:
            new_conflicts.append(case.id)
        if before_conflict and not after_conflict:
            resolved_conflicts.append(case.id)
    return {
        "affected_cases": len(affected),
        "affected_case_ids": affected,
        "affected_rule_evidence_cases": len(evidence_affected),
        "affected_rule_evidence_case_ids": evidence_affected,
        "new_hard_errors": new_errors,
        "fixed_hard_errors": fixed_errors,
        "new_hard_conflicts": new_conflicts,
        "resolved_hard_conflicts": resolved_conflicts,
        "new_regression_hard_errors": new_regression_errors,
    }


def _validate_promotion_receipts(
    *,
    active_rules_path: Path,
    change_path: Path,
    report: dict[str, Any],
) -> None:
    receipts = report.get("receipts", {})
    if receipts.get("active_rules_sha256") != _sha256_file(active_rules_path):
        raise ValueError("Active rules changed after shadow evaluation")
    if receipts.get("change_sha256") != _sha256_file(change_path):
        raise ValueError("Rule change changed after shadow evaluation")
    for dataset in receipts.get("datasets", []):
        path = Path(dataset["path"])
        if not path.exists() or dataset.get("sha256") != _sha256_file(path):
            raise ValueError(f"Evaluation dataset changed: {path}")


def _reject_reused_release_version(history_dir: Path, version: str) -> None:
    for path in history_dir.glob("*.promotion-*.json"):
        try:
            receipt = _load_json(path)
        except ValueError:
            continue
        if receipt.get("to_version") == version:
            raise ValueError(
                f"Rule release version was already used: {version}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Rule report must be a JSON object")
    return payload


def _version_core(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    major, minor, patch = core.split(".")
    return int(major), int(minor), int(patch)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_snapshot_path(
    history_dir: Path,
    version: str,
    digest: str,
) -> Path:
    return history_dir / f"rules-v{version}-{digest[:12]}.yaml"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()
