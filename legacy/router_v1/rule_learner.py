import json
from collections import defaultdict, Counter
from pathlib import Path


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                if "input" in j and "final_route" in j:
                    data.append((j["input"], j["final_route"]))
            except json.JSONDecodeError:
                continue
    return data


def extract_ngrams(text, n=2):
    text = text.lower()
    return [text[i:i+n] for i in range(len(text)-n+1)]


def learn_rules(data):
    label_tokens = defaultdict(list)

    for text, label in data:
        tokens = extract_ngrams(text, n=2)
        label_tokens[label].extend(tokens)

    label_counts = {
        label: Counter(tokens)
        for label, tokens in label_tokens.items()
    }

    return label_counts


def find_candidates(label_counts, min_freq=5):
    candidates = {}

    for label, counter in label_counts.items():
        candidates[label] = [
            token for token, freq in counter.items()
            if freq >= min_freq
        ]

    return candidates


def main(log_path="logs/router.log", min_freq=5):
    path = Path(log_path)
    if not path.exists():
        print(f"log file not found: {path}")
        return

    data = load_data(path)
    label_counts = learn_rules(data)
    candidates = find_candidates(label_counts, min_freq=min_freq)

    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()