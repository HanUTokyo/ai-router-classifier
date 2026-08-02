from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from router.rule_lifecycle import (
    RuleChange,
    RuleGateThresholds,
    build_candidate_rule_file,
    evaluate_rule_change,
    promote_rule_change,
    rollback_rules,
    write_rule_change,
)
from router.rules import RuleEngine, RuleSpec
from router.types import RouteLabel


def _write_base_rules(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0.0",
                "rules": [
                    {
                        "id": "chat-hello",
                        "label": "chat",
                        "kind": "hard",
                        "pattern_type": "exact",
                        "pattern": "hello",
                        "enabled": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_cases(path: Path, *, count: int = 20) -> None:
    rows = [
        {
            "id": f"code-{index}",
            "text": f"write code sample {index}",
            "expected_route": "code",
            "language": "en",
            "split": "dev",
            "tags": ["rule_candidate"],
            "verified": True,
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _good_change() -> RuleChange:
    return RuleChange(
        change_id="add-code-write-v2",
        base_version="1.0.0",
        target_version="2.0.0",
        action="add",
        rule_id="code-write",
        reason="Twenty reviewed examples use this explicit phrase.",
        created_by="reviewer",
        evidence_refs=["code-0"],
        proposed_rule=RuleSpec(
            id="code-write",
            label="code",
            kind="hard",
            pattern_type="phrase",
            pattern="write code",
        ),
    )


def test_negative_rule_change_requires_human_evidence():
    with pytest.raises(ValueError, match="evidence_ref"):
        RuleChange(
            change_id="disable-chat",
            base_version="1.0.0",
            target_version="1.1.0",
            action="disable",
            rule_id="chat-hello",
            reason="Reported false positive",
            created_by="reviewer",
        )


def test_rule_gate_thresholds_cannot_be_weakened():
    with pytest.raises(ValueError):
        RuleGateThresholds(hard_precision=0.9)
    with pytest.raises(ValueError):
        RuleGateThresholds(target_rule_min_matches=5)


def test_update_cannot_bypass_promote_or_demote_action(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    _write_base_rules(rules_path)
    active = RuleEngine(rules_path).rule_file
    change = RuleChange(
        change_id="hidden-demotion",
        base_version="1.0.0",
        target_version="1.1.0",
        action="update",
        rule_id="chat-hello",
        reason="Attempted kind change",
        created_by="reviewer",
        evidence_refs=["feedback-1"],
        proposed_rule=active.rules[0].model_copy(update={"kind": "weak"}),
    )

    with pytest.raises(ValueError, match="cannot change kind"):
        build_candidate_rule_file(active, change)


def test_shadow_report_passes_precise_supported_hard_rule(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    change_path = tmp_path / "change.yaml"
    dataset_path = tmp_path / "dev.jsonl"
    _write_base_rules(rules_path)
    _write_cases(dataset_path)
    write_rule_change(change_path, _good_change())

    report = evaluate_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        datasets=[(dataset_path, "dev")],
    )

    assert report["status"] == "shadow_evaluated"
    assert report["candidate"]["per_rule"]["code-write"] == {
        "matches": 20,
        "correct": 20,
        "precision": 1.0,
    }
    assert report["comparison"]["affected_cases"] == 20
    assert report["comparison"]["new_hard_errors"] == []
    assert report["gate"]["passed"] is True


def test_shadow_report_rejects_false_positive_hard_rule(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    change_path = tmp_path / "change.yaml"
    dataset_path = tmp_path / "dev.jsonl"
    _write_base_rules(rules_path)
    _write_cases(dataset_path)
    bad = _good_change().model_copy(
        update={
            "proposed_rule": _good_change().proposed_rule.model_copy(
                update={"label": RouteLabel.CHAT}
            )
        }
    )
    write_rule_change(change_path, bad)

    report = evaluate_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        datasets=[(dataset_path, "dev")],
    )

    assert report["candidate"]["precision"] == 0.0
    assert len(report["comparison"]["new_hard_errors"]) == 20
    assert report["gate"]["passed"] is False


def test_shadow_report_can_gate_weak_evidence_without_hard_takeover(
    tmp_path: Path,
):
    rules_path = tmp_path / "rules.yaml"
    change_path = tmp_path / "change.yaml"
    dataset_path = tmp_path / "dev.jsonl"
    _write_base_rules(rules_path)
    _write_cases(dataset_path)
    change = RuleChange(
        change_id="add-weak-code",
        base_version="1.0.0",
        target_version="1.1.0",
        action="add",
        rule_id="weak-code",
        reason="Reviewed evidence for a weak feature",
        created_by="reviewer",
        evidence_refs=["code-0"],
        proposed_rule=RuleSpec(
            id="weak-code",
            label="code",
            kind="weak",
            pattern_type="phrase",
            pattern="write code",
        ),
    )
    write_rule_change(change_path, change)

    report = evaluate_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        datasets=[(dataset_path, "dev")],
    )

    assert report["comparison"]["affected_cases"] == 0
    assert report["comparison"]["affected_rule_evidence_cases"] == 20
    assert report["gate"]["candidate_changes_rule_evidence"] is True
    assert report["gate"]["passed"] is True


def test_promotion_requires_unchanged_receipts_and_supports_rollback(
    tmp_path: Path,
):
    rules_path = tmp_path / "rules.yaml"
    change_path = tmp_path / "change.yaml"
    report_path = tmp_path / "report.json"
    dataset_path = tmp_path / "dev.jsonl"
    history_dir = tmp_path / "history"
    _write_base_rules(rules_path)
    _write_cases(dataset_path)
    write_rule_change(change_path, _good_change())
    report = evaluate_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        datasets=[(dataset_path, "dev")],
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    receipt = promote_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        report_path=report_path,
        history_dir=history_dir,
        approved_by="Kai",
    )

    assert RuleEngine(rules_path).version == "2.0.0"
    snapshot = Path(receipt["previous_snapshot"])
    archived_report = Path(receipt["shadow_report_archive"])
    assert RuleEngine(snapshot).version == "1.0.0"
    assert Path(receipt["receipt_path"]).exists()
    assert not report_path.exists()

    rollback = rollback_rules(
        active_rules_path=rules_path,
        snapshot_path=snapshot,
        history_dir=history_dir,
        approved_by="Kai",
        reason="Canary regression",
    )

    assert RuleEngine(rules_path).version == "1.0.0"
    assert rollback["from_version"] == "2.0.0"
    assert rollback["to_version"] == "1.0.0"

    with pytest.raises(ValueError, match="version was already used"):
        promote_rule_change(
            active_rules_path=rules_path,
            change_path=change_path,
            report_path=archived_report,
            history_dir=history_dir,
            approved_by="Kai",
        )


def test_promotion_rejects_changed_dataset(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    change_path = tmp_path / "change.yaml"
    report_path = tmp_path / "report.json"
    dataset_path = tmp_path / "dev.jsonl"
    _write_base_rules(rules_path)
    _write_cases(dataset_path)
    write_rule_change(change_path, _good_change())
    report = evaluate_rule_change(
        active_rules_path=rules_path,
        change_path=change_path,
        datasets=[(dataset_path, "dev")],
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dataset_path.write_text(
        dataset_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset changed"):
        promote_rule_change(
            active_rules_path=rules_path,
            change_path=change_path,
            report_path=report_path,
            history_dir=tmp_path / "history",
            approved_by="Kai",
        )
