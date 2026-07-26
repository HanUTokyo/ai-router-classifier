import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import yaml

MIN_FREQ = 3
TARGET_LABELS = ["code", "reason", "chat"]
STOPWORDS = {
    "的是", "的是", "了", "和", "就", "都", "而", "及", "与", "着", "或", "一个", "一下",
    "可以", "这个", "那个", "我们", "你们", "他们", "帮我", "请问", "how", "what", "why",
}
NOISE_CHUNKS = {"帮我", "写一个", "帮我写", "一个", "问题", "事情"}


def load_logs(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                if "input" in j and "final" in j and isinstance(j["final"], dict):
                    if j["final"].get("label"):
                        data.append(j)
            except json.JSONDecodeError:
                continue
    return data


def extract_keywords(text):
    if not text:
        return []

    text_lower = text.lower()
    keywords = []

    english_tokens = re.findall(r"[a-z][a-z0-9_+#.-]{1,}", text_lower)
    keywords.extend(english_tokens)

    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for chunk in chinese_chunks:
        max_n = min(3, len(chunk))
        for n in range(2, max_n + 1):
            for i in range(len(chunk) - n + 1):
                token = chunk[i:i + n]
                if token not in NOISE_CHUNKS:
                    keywords.append(token)

    deduped = []
    seen = set()
    for word in keywords:
        if len(word) < 2:
            continue
        if word in STOPWORDS:
            continue
        if word in seen:
            continue
        seen.add(word)
        deduped.append(word)

    return deduped


def build_rule_candidates(logs, min_freq=MIN_FREQ):
    label_words = defaultdict(list)

    for item in logs:
        text = item.get("input", "")
        label = item.get("final", {}).get("label")
        if not label:
            continue

        words = extract_keywords(text)
        label_words[label].extend(words)

    result = {}
    for label in TARGET_LABELS:
        counter = Counter(label_words[label])
        result[label] = [w for w, freq in counter.items() if freq >= min_freq]

    return result


def build_training_data(logs):
    dataset = []

    for item in logs:
        text = item.get("input", "")
        final_label = item.get("final", {}).get("label")
        rule_label = item.get("rule", {}).get("label")
        model_label = item.get("model", {}).get("label")

        if not final_label:
            continue

        if rule_label != final_label or model_label != final_label:
            dataset.append({
                "instruction": f"分类以下输入：{text}",
                "output": final_label,
            })

    return dataset


def load_feedback_training_data(path):
    if not path or not os.path.exists(path):
        return []

    dataset = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = item.get("message") or item.get("input") or ""
            label = item.get("expected_route") or item.get("label") or item.get("output")
            if label not in TARGET_LABELS or not message:
                continue

            key = (message, label)
            if key in seen:
                continue
            seen.add(key)
            dataset.append({
                "instruction": item.get("instruction") or f"分类以下输入：{message}",
                "output": label,
                "source": item.get("source", "feedback"),
            })

    return dataset


def find_rule_missing(logs):
    missing = []
    seen = set()
    for item in logs:
        rule_hits = item.get("rule", {}).get("hits", [])
        final_label = item.get("final", {}).get("label")
        input_text = item.get("input", "")

        if final_label and not rule_hits:
            key = (input_text, final_label)
            if key in seen:
                continue
            seen.add(key)
            missing.append({"input": input_text, "label": final_label})

    return missing


def compute_tfidf(logs):
    label_docs = defaultdict(list)
    doc_freq = Counter()
    total_docs = 0

    for item in logs:
        label = item.get("final", {}).get("label")
        text = item.get("input", "")
        if not label:
            continue

        words = set(extract_keywords(text))
        if not words:
            continue

        label_docs[label].append(words)
        total_docs += 1
        for w in words:
            doc_freq[w] += 1

    if total_docs == 0:
        return {}, {}

    tfidf_scores = defaultdict(dict)
    for label, docs in label_docs.items():
        tf = Counter()
        for doc in docs:
            tf.update(doc)

        for word, freq in tf.items():
            idf = math.log(total_docs / (1 + doc_freq[word]))
            tfidf_scores[label][word] = round(freq * idf, 6)

    return dict(tfidf_scores), dict(doc_freq)


def load_rules_keywords(path):
    with open(path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}

    existing = {}
    for label, cfg in rules.items():
        existing[label] = set(cfg.get("keywords", [])) if isinstance(cfg, dict) else set()
    return existing


def generate_rule_patch(tfidf_scores, doc_freq, existing_rules, top_k=5, min_doc_freq=2):
    patch = {}

    for label, scores in tfidf_scores.items():
        if label == "chat":
            patch[label] = []
            continue

        current = existing_rules.get(label, set())
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        additions = []
        for word, score in sorted_words:
            if score <= 0:
                continue
            if doc_freq.get(word, 0) < min_doc_freq:
                continue
            if word in current:
                continue
            additions.append(word)
            if len(additions) >= top_k:
                break

        patch[label] = additions

    return patch


def save_yaml_like(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k}:\n")
            for item in v:
                f.write(f"  - {item}\n")

    os.replace(temp_path, path)


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    os.replace(temp_path, path)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def save_patch(patch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("# Proposed patch for rules.yaml (review before applying)\n")
        for label, words in patch.items():
            f.write(f"\n[{label}]\n")
            if not words:
                f.write("# no new keywords\n")
            for w in words:
                f.write(f"+ - {w}\n")

    os.replace(temp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Router analyzer v3")
    parser.add_argument("--log-path", default="logs/router.log", help="Path to router logs")
    parser.add_argument("--rules-path", default="rules.yaml", help="Path to existing rules yaml")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top keywords per label for patch")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQ, help="Min frequency for rule_candidates")
    parser.add_argument(
        "--feedback-path",
        default="feedback_data.jsonl",
        help="Optional router feedback jsonl to merge into training_data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logs = load_logs(args.log_path)

    candidates = build_rule_candidates(logs, min_freq=args.min_freq)
    feedback_dataset = load_feedback_training_data(args.feedback_path)
    dataset = build_training_data(logs) + feedback_dataset
    missing = find_rule_missing(logs)
    tfidf_scores, doc_freq = compute_tfidf(logs)
    existing_rules = load_rules_keywords(args.rules_path)
    patch = generate_rule_patch(
        tfidf_scores,
        doc_freq,
        existing_rules,
        top_k=args.top_k,
    )

    save_yaml_like(candidates, os.path.join(args.output_dir, "rule_candidates.yaml"))
    save_jsonl(dataset, os.path.join(args.output_dir, "training_data.jsonl"))
    save_jsonl(missing, os.path.join(args.output_dir, "rule_missing.jsonl"))
    save_json(tfidf_scores, os.path.join(args.output_dir, "tfidf_scores.json"))
    save_patch(patch, os.path.join(args.output_dir, "rules_patch.txt"))

    print("✅ Analyzer v3 done")
    print(f"logs: {len(logs)}")
    print(f"rule_candidates labels: {len(candidates)}")
    print(f"training_data: {len(dataset)}")
    print(f"feedback_training_data: {len(feedback_dataset)}")
    print(f"rule_missing: {len(missing)}")
    print(f"patch_labels: {sum(1 for _, v in patch.items() if v)}")


if __name__ == "__main__":
    main()
