from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


LABEL_KEYS = {
    "c": "code",
    "r": "reason",
    "h": "chat",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Review AI Router seed labels")
    parser.add_argument("--dataset", default="data/router_dev_v4.jsonl")
    parser.add_argument("--reviewer", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.dataset)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remaining = sum(not record.get("verified", False) for record in records)
    print(f"{remaining} records require review")
    for record in records:
        if record.get("verified", False):
            continue
        print()
        print(f"{record['id']} [{record['language']}/{record['split']}]")
        if record.get("context"):
            print("context:")
            for message in record["context"]:
                print(f"  {message['role']}: {message.get('content', '')}")
            print("current:")
        print(record["text"])
        print(f"proposed: {record['expected_route']}")
        answer = input(
            "Enter=accept, c=code, r=reason, h=chat, s=skip, q=quit: "
        ).strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        if answer in LABEL_KEYS:
            record["expected_route"] = LABEL_KEYS[answer]
        elif answer:
            print("Unknown choice; record was not changed")
            continue
        record["verified"] = True
        record["review_status"] = "verified"
        record["reviewed_by"] = args.reviewer
        record["reviewed_at"] = datetime.now(UTC).isoformat()
        _write_atomic(path, records)
        remaining -= 1
        print(f"saved; {remaining} remaining")


def _write_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
