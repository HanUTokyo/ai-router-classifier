import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_atomic(data, path: Path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    temp_path.replace(path)


def parse_rules_patch(path: Path):
    patch = defaultdict(list)
    current_label = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_label = line[1:-1].strip()
                continue

            if line.startswith("+ - ") and current_label:
                keyword = line[4:].strip()
                if keyword:
                    patch[current_label].append(keyword)

    return dict(patch)


def load_approval_file(path: Path):
    if not path:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"approval file not found: {path}")

    if path.suffix.lower() in {".json"}:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = load_yaml(path)

    if not isinstance(data, dict):
        raise ValueError("approval file must be a mapping: label -> [keywords]")

    normalized = {}
    for label, words in data.items():
        if words is None:
            normalized[str(label)] = []
            continue
        if not isinstance(words, list):
            raise ValueError(f"approval label '{label}' must map to a list")
        normalized[str(label)] = [str(w).strip() for w in words if str(w).strip()]

    return normalized


def parse_approve_args(approve_args):
    approvals = defaultdict(set)
    for item in approve_args:
        if ":" not in item:
            raise ValueError(
                f"invalid --approve value '{item}', expected format label:kw1,kw2"
            )

        label, raw_keywords = item.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"invalid --approve value '{item}', empty label")

        keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        for kw in keywords:
            approvals[label].add(kw)

    return approvals


def merge_approvals(file_approvals, cli_approvals):
    merged = defaultdict(set)

    for label, keywords in file_approvals.items():
        for kw in keywords:
            merged[label].add(kw)

    for label, keywords in cli_approvals.items():
        for kw in keywords:
            merged[label].add(kw)

    return merged


def validate_approvals(approvals, patch):
    errors = []

    for label, keywords in approvals.items():
        patch_keywords = set(patch.get(label, []))
        if not patch_keywords:
            errors.append(f"label '{label}' has no patch candidates")
            continue

        for kw in keywords:
            if kw not in patch_keywords:
                errors.append(f"keyword '{kw}' not found in patch label '{label}'")

    if errors:
        raise ValueError("invalid approvals:\n- " + "\n- ".join(errors))


def backup_rules_file(rules_path: Path):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = rules_path.with_name(f"{rules_path.stem}.{ts}{rules_path.suffix}")
    shutil.copy2(rules_path, backup_path)
    return backup_path


def apply_keywords(rules, patch, approvals):
    applied = defaultdict(list)

    for label, approved_set in approvals.items():
        if not approved_set:
            continue

        if label not in rules or not isinstance(rules[label], dict):
            rules[label] = {"keywords": []}

        existing = rules[label].get("keywords", [])
        if not isinstance(existing, list):
            existing = []

        patch_order = patch.get(label, [])
        for kw in patch_order:
            if kw in approved_set and kw not in existing:
                existing.append(kw)
                applied[label].append(kw)

        rules[label]["keywords"] = existing

    return rules, dict(applied)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-auto rules patch applier with human approval gate"
    )
    parser.add_argument("--rules-path", default="rules.yaml")
    parser.add_argument("--patch-path", default="rules_patch.txt")
    parser.add_argument(
        "--approval-file",
        default="",
        help="YAML/JSON file mapping label -> approved keywords",
    )
    parser.add_argument(
        "--approve",
        action="append",
        default=[],
        help="Inline approval, format label:kw1,kw2 (can repeat)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    rules_path = Path(args.rules_path)
    patch_path = Path(args.patch_path)
    approval_file = Path(args.approval_file) if args.approval_file else None

    if not rules_path.exists():
        raise FileNotFoundError(f"rules file not found: {rules_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"patch file not found: {patch_path}")

    patch = parse_rules_patch(patch_path)
    file_approvals = load_approval_file(approval_file) if approval_file else {}
    cli_approvals = parse_approve_args(args.approve)
    approvals = merge_approvals(file_approvals, cli_approvals)

    if not approvals:
        raise ValueError(
            "no approved keywords provided; use --approval-file or --approve"
        )

    validate_approvals(approvals, patch)

    rules = load_yaml(rules_path)
    updated_rules, applied = apply_keywords(rules, patch, approvals)

    applied_total = sum(len(v) for v in applied.values())
    print(f"approved_labels: {len(approvals)}")
    print(f"applied_keywords: {applied_total}")

    for label, words in applied.items():
        if words:
            print(f"- {label}: {', '.join(words)}")

    if args.dry_run:
        print("dry_run: true (rules.yaml not modified)")
        return

    backup_path = backup_rules_file(rules_path)
    save_yaml_atomic(updated_rules, rules_path)

    print(f"backup: {backup_path}")
    print(f"updated: {rules_path}")


if __name__ == "__main__":
    main()
