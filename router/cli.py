from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Router classifier")
    parser.add_argument(
        "--config",
        default=os.getenv("AI_ROUTER_CONFIG", "config/router.yaml"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Run the classifier and thin gateway")
    sub.add_parser("review", help="Run the local human-review interface")
    sub.add_parser("doctor", help="Check configuration and installed models")

    evaluate = sub.add_parser("evaluate", help="Evaluate one classifier model")
    evaluate.add_argument("--dataset", default="data/router_dev_v4.jsonl")
    evaluate.add_argument(
        "--split",
        choices=["dev", "test", "regression"],
        default="dev",
    )
    evaluate.add_argument("--model")
    evaluate.add_argument(
        "--legacy",
        action="store_true",
        help="Evaluate the original substring/fixed-confidence policy",
    )
    evaluate.add_argument("--allow-unverified", action="store_true")
    evaluate.add_argument("--output")
    evaluate.add_argument("--write-calibration", action="store_true")

    benchmark = sub.add_parser("benchmark", help="Benchmark configured candidates")
    benchmark.add_argument("--dataset", default="data/router_dev_v4.jsonl")
    benchmark.add_argument("--split", choices=["dev"], default="dev")
    benchmark.add_argument("--allow-unverified", action="store_true")
    benchmark.add_argument("--output", default="artifacts/benchmark.json")

    rules = sub.add_parser(
        "rules",
        help="Propose, shadow-evaluate, promote, or roll back rule changes",
    )
    rule_actions = rules.add_subparsers(dest="rules_command", required=True)

    propose = rule_actions.add_parser(
        "propose",
        help="Create a non-active rule change proposal",
    )
    propose.add_argument(
        "--action",
        choices=["add", "update", "disable", "demote", "promote"],
        required=True,
    )
    propose.add_argument("--rule-id", required=True)
    propose.add_argument("--target-version", required=True)
    propose.add_argument("--reason", required=True)
    propose.add_argument("--created-by", required=True)
    propose.add_argument("--change-id")
    propose.add_argument(
        "--rule-spec",
        help="YAML file containing one RuleSpec; required for add/update",
    )
    propose.add_argument(
        "--evidence-ref",
        action="append",
        default=[],
        help="Human-reviewed sample or feedback reference; repeatable",
    )
    propose.add_argument("--output", required=True)

    shadow = rule_actions.add_parser(
        "evaluate",
        help="Run a rule proposal in shadow mode on verified development data",
    )
    shadow.add_argument("--candidate", required=True)
    shadow.add_argument("--dev-dataset", default="data/router_dev_v4.jsonl")
    shadow.add_argument(
        "--regression-dataset",
        default="data/router_regression_v2.jsonl",
    )
    shadow.add_argument(
        "--output",
        help="Defaults to classifier.shadow_report_path from router config",
    )

    promote = rule_actions.add_parser(
        "promote",
        help="Activate a proposal that passed an unchanged shadow report",
    )
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--report", required=True)
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--history-dir", default="config/rule_history")

    rollback = rule_actions.add_parser(
        "rollback",
        help="Restore a previously snapshotted rule version",
    )
    rollback.add_argument("--snapshot", required=True)
    rollback.add_argument("--approved-by", required=True)
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--history-dir", default="config/rule_history")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .settings import Settings

    settings = Settings.load(args.config)
    if args.command == "rules":
        _run_rule_command(args, settings)
        return
    if args.command in {"serve", "review"}:
        import uvicorn

        from .app import create_app

        if args.command == "review":
            settings = settings.model_copy(
                update={
                    "classifier": settings.classifier.model_copy(
                        update={"strict_model_check": False}
                    )
                }
            )
            print(
                "Human review interface: "
                f"http://{settings.server.host}:{settings.server.port}/review"
            )
        uvicorn.run(
            create_app(settings),
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.logging.level.lower(),
        )
        return
    asyncio.run(_run_async(args, settings))


def _run_rule_command(args, settings) -> None:
    import yaml

    from .evaluation import write_json_atomic
    from .rule_lifecycle import (
        RuleChange,
        evaluate_rule_change,
        promote_rule_change,
        rollback_rules,
        write_rule_change,
    )
    from .rules import RuleEngine, RuleSpec

    def selected_path(raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    active_path = settings.classifier.rules_path
    if args.rules_command == "propose":
        proposed_rule = None
        if args.rule_spec:
            rule_spec_path = selected_path(args.rule_spec)
            try:
                raw_rule = yaml.safe_load(
                    rule_spec_path.read_text(encoding="utf-8")
                )
            except (OSError, yaml.YAMLError) as exc:
                raise ValueError(
                    f"Unable to read rule spec: {rule_spec_path}"
                ) from exc
            proposed_rule = RuleSpec.model_validate(raw_rule)
        active = RuleEngine(active_path).rule_file
        change_id = args.change_id or (
            f"{args.rule_id}-{args.action}-v{args.target_version}"
        )
        change = RuleChange(
            change_id=change_id,
            base_version=active.version,
            target_version=args.target_version,
            action=args.action,
            rule_id=args.rule_id,
            reason=args.reason,
            created_by=args.created_by,
            evidence_refs=args.evidence_ref,
            proposed_rule=proposed_rule,
        )
        output = selected_path(args.output)
        if output.exists():
            raise ValueError(f"Refusing to overwrite rule proposal: {output}")
        write_rule_change(output, change)
        print(
            json.dumps(
                {
                    "status": "candidate",
                    "change_id": change.change_id,
                    "path": str(output),
                    "next": "Run `ai-router rules evaluate` before promotion.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.rules_command == "evaluate":
        output = (
            selected_path(args.output)
            if args.output
            else settings.classifier.shadow_report_path
        )
        if output is None:
            raise ValueError(
                "--output is required when classifier.shadow_report_path is disabled"
            )
        report = evaluate_rule_change(
            active_rules_path=active_path,
            change_path=selected_path(args.candidate),
            datasets=[
                (selected_path(args.dev_dataset), "dev"),
                (selected_path(args.regression_dataset), "regression"),
            ],
        )
        write_json_atomic(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    history_dir = selected_path(args.history_dir)
    if args.rules_command == "promote":
        receipt = promote_rule_change(
            active_rules_path=active_path,
            change_path=selected_path(args.candidate),
            report_path=selected_path(args.report),
            history_dir=history_dir,
            approved_by=args.approved_by,
        )
    else:
        receipt = rollback_rules(
            active_rules_path=active_path,
            snapshot_path=selected_path(args.snapshot),
            history_dir=history_dir,
            approved_by=args.approved_by,
            reason=args.reason,
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


async def _run_async(args, settings) -> None:
    from .benchmark import benchmark_candidates
    from .classifier import RouterClassifier
    from .evaluation import (
        build_calibration,
        load_cases,
        predict_cases,
        score_predictions,
        validate_locked_dataset_manifest,
        write_json_atomic,
    )
    from .ollama import OllamaClient

    ollama = OllamaClient(settings.ollama)
    classifier = RouterClassifier(
        settings,
        ollama,
        activate_provisional_hard_rules=args.command in {"evaluate", "benchmark"},
    )
    try:
        if args.command == "doctor":
            installed = await ollama.installed_models()
            missing = sorted(settings.required_models - set(installed))
            print(
                json.dumps(
                    {
                        "config": str(settings.config_path),
                        "ollama": settings.ollama.base_url,
                        "classifier_model": settings.classifier.model,
                        "calibration": {
                            "version": classifier.calibrator.version,
                            "validated": classifier.calibrator.validated,
                            "hard_rules_enabled": (
                                classifier.calibrator.hard_rules_enabled
                            ),
                            "weak_fallback_enabled": (
                                classifier.calibrator.weak_fallback_enabled
                            ),
                        },
                        "rule_version": classifier.rules.version,
                        "rule_digest": classifier.rules.digest,
                        "shadow_rules": {
                            "version": (
                                classifier.shadow_rules.version
                                if classifier.shadow_rules is not None
                                else None
                            ),
                            "loaded": classifier.shadow_rules is not None,
                            "error": classifier.shadow_rules_error,
                        },
                        "installed": installed,
                        "missing_required": missing,
                        "ready": not missing,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(1 if missing else 0)

        cases = load_cases(
            Path(args.dataset),
            split=args.split,
            allow_unverified=args.allow_unverified,
        )
        if args.command == "evaluate":
            locked_manifest = None
            if args.write_calibration:
                if args.split != "test":
                    raise ValueError(
                        "Calibration may only be written from split=test"
                    )
                locked_manifest = validate_locked_dataset_manifest(
                    Path(args.dataset)
                )
            selected_classifier = classifier
            if args.legacy:
                from .legacy_baseline import LegacyBaselineClassifier

                selected_classifier = LegacyBaselineClassifier(settings, ollama)
            predictions = await predict_cases(
                selected_classifier,
                cases,
                model_override=args.model,
            )
            metrics = score_predictions(predictions)
            model_name = args.model or settings.classifier.model
            installed = await ollama.installed_models()
            dataset_path = Path(args.dataset)
            result = {
                "model": model_name,
                "model_digest": installed.get(model_name),
                "policy": "legacy" if args.legacy else "current",
                "split": args.split,
                "dataset": dataset_path.name,
                "dataset_sha256": hashlib.sha256(
                    dataset_path.read_bytes()
                ).hexdigest(),
                "verified": all(case.verified for case in cases),
                "classifier_version": settings.classifier.version,
                "prompt_version": settings.classifier.prompt_version,
                "rule_version": classifier.rules.version,
                "rule_digest": classifier.rules.digest,
                "metrics": metrics,
            }
            if args.write_calibration:
                if args.legacy:
                    raise ValueError(
                        "Legacy policy results cannot be used for calibration"
                    )
                if not all(case.verified for case in cases):
                    raise ValueError(
                        "Calibration may only be written from a fully verified test split"
                    )
                if args.model and args.model != settings.classifier.model:
                    raise ValueError(
                        "Freeze the winning model in config/router.yaml before "
                        "writing calibration"
                    )
                if not metrics["quality_gate"]["passed"]:
                    result["calibration_written"] = None
                    result["calibration_error"] = (
                        "Test quality gate failed; calibration was not written."
                    )
                else:
                    calibration = build_calibration(
                        predictions,
                        version=str(locked_manifest["dataset_id"]),
                    )
                    if not calibration:
                        raise ValueError("Unable to build calibration")
                    calibration.update(
                        {
                            "classifier_model": (
                                args.model or settings.classifier.model
                            ),
                            "classifier_version": settings.classifier.version,
                            "prompt_version": settings.classifier.prompt_version,
                            "rule_version": classifier.rules.version,
                            "rule_digest": classifier.rules.digest,
                            "dataset_id": locked_manifest["dataset_id"],
                            "dataset_sha256": locked_manifest["sha256"],
                            "weak_fallback_enabled": True,
                            "weak_fallback_threshold": (
                                settings.classifier.weak_fallback_threshold
                            ),
                            "weak_fallback_margin": (
                                settings.classifier.weak_fallback_margin
                            ),
                        }
                    )
                    write_json_atomic(
                        settings.classifier.calibration_path,
                        calibration,
                    )
                    result["calibration_written"] = str(
                        settings.classifier.calibration_path
                    )
            if args.output:
                write_json_atomic(Path(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("calibration_error"):
                raise SystemExit(2)
            return

        installed = await ollama.installed_models()
        result = await benchmark_candidates(
            classifier,
            cases,
            settings.classifier.candidates,
            installed,
        )
        result["split"] = args.split
        verified = all(case.verified for case in cases)
        result["verified"] = verified
        if not verified:
            result["quality_eligible_candidate"] = None
            result["next_action"] = (
                "Complete human review before treating any candidate as eligible."
            )
        write_json_atomic(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await ollama.close()


if __name__ == "__main__":
    main()
