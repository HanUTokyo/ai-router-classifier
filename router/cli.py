from __future__ import annotations

import argparse
import asyncio
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
    sub.add_parser("doctor", help="Check configuration and installed models")

    evaluate = sub.add_parser("evaluate", help="Evaluate one classifier model")
    evaluate.add_argument("--dataset", default="data/router_cases.jsonl")
    evaluate.add_argument("--split", choices=["dev", "test"], default="dev")
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
    benchmark.add_argument("--dataset", default="data/router_cases.jsonl")
    benchmark.add_argument("--split", choices=["dev"], default="dev")
    benchmark.add_argument("--allow-unverified", action="store_true")
    benchmark.add_argument("--output", default="artifacts/benchmark.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .settings import Settings

    settings = Settings.load(args.config)
    if args.command == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(
            create_app(settings),
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.logging.level.lower(),
        )
        return
    asyncio.run(_run_async(args, settings))


async def _run_async(args, settings) -> None:
    from .benchmark import benchmark_candidates
    from .classifier import RouterClassifier
    from .evaluation import (
        build_calibration,
        load_cases,
        predict_cases,
        score_predictions,
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
            result = {
                "model": args.model or settings.classifier.model,
                "policy": "legacy" if args.legacy else "current",
                "split": args.split,
                "verified": all(case.verified for case in cases),
                "metrics": metrics,
            }
            if args.write_calibration:
                if args.legacy:
                    raise ValueError(
                        "Legacy policy results cannot be used for calibration"
                    )
                if args.split != "test" or not all(case.verified for case in cases):
                    raise ValueError(
                        "Calibration may only be written from a fully verified test split"
                    )
                if args.model and args.model != settings.classifier.model:
                    raise ValueError(
                        "Freeze the winning model in config/router.yaml before "
                        "writing calibration"
                    )
                if not metrics["quality_gate"]["passed"]:
                    raise ValueError(
                        "Calibration was not written because the test quality gate failed"
                    )
                calibration = build_calibration(predictions)
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
                        "weak_fallback_enabled": True,
                        "weak_fallback_threshold": (
                            settings.classifier.weak_fallback_threshold
                        ),
                        "weak_fallback_margin": (
                            settings.classifier.weak_fallback_margin
                        ),
                    }
                )
                write_json_atomic(settings.classifier.calibration_path, calibration)
                result["calibration_written"] = str(
                    settings.classifier.calibration_path
                )
            if args.output:
                write_json_atomic(Path(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
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
