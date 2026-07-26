import yaml
from pathlib import Path

class RuleEngine:
    def __init__(self, config_path="rules.yaml"):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = Path(__file__).resolve().parent / config_file

        try:
            with config_file.open("r", encoding="utf-8") as f:
                self.rules = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Rule config not found: {config_file}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML format in {config_file}: {exc}") from exc

        if not isinstance(self.rules, dict) or not self.rules:
            raise ValueError(f"Rule config is empty or invalid: {config_file}")

    def match(self, text):
        t = text.lower()

        scores = {}
        hits = {}

        # 🔥 keyword scoring
        for label, cfg in self.rules.items():
            score = 0
            hit_keywords = []

            for kw in cfg.get("keywords", []):
                if kw in t:
                    score += cfg.get("weight", 1.0)
                    hit_keywords.append(kw)

            if score > 0:
                scores[label] = score
                hits[label] = hit_keywords

        # 🧠 有命中 → 选最高
        if scores:
            best_label = max(scores, key=scores.get)

            base_conf = self.rules[best_label].get("confidence", 0.5)
            boost = min(0.1 * scores[best_label], 0.2)
            final_conf = min(base_conf + boost, 1.0)

            return {
                "label": best_label,
                "confidence": round(final_conf, 3),
                "hits": hits.get(best_label, []),
                "score": scores[best_label]
            }

        # 📏 fallback：短文本 → chat
        if len(t) < self.rules["chat"]["short_text_len"]:
            return {
                "label": "chat",
                "confidence": self.rules["chat"]["confidence"],
                "hits": [],
                "score": 0
            }

        return {
            "label": None,
            "confidence": 0.0,
            "hits": [],
            "score": 0
        }