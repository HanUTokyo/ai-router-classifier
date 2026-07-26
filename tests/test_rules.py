from pathlib import Path

import pytest
import yaml

from router.rules import RuleEngine
from router.settings import PROJECT_ROOT
from router.types import RouteLabel


def engine() -> RuleEngine:
    return RuleEngine(Path(PROJECT_ROOT / "config/rules.yaml"))


def test_english_tokens_do_not_match_substrings():
    evidence = engine().evaluate("Please explain capitalism in one sentence")

    assert not any(hit.label == RouteLabel.CODE for hit in evidence.weak_hits)


def test_broad_words_are_not_hard_code_rules():
    for text in (
        "spring flowers are beautiful",
        "怎样实现人生目标",
        "介绍发展中国家的文化",
    ):
        assert engine().evaluate(text).unique_hard_label is None


def test_technical_words_in_non_technical_usage_are_excluded():
    for text in (
        "Python 这种蛇生活在哪里？",
        "What does API stand for?",
        "介绍 Java 岛的景点",
    ):
        evidence = engine().evaluate(text)
        assert not any(
            hit.rule_id == "weak-code-tech-en" for hit in evidence.weak_hits
        )


def test_explicit_code_request_is_a_unique_hard_rule():
    evidence = engine().evaluate("请帮我修复这段 Python 代码")

    assert evidence.unique_hard_label == RouteLabel.CODE


def test_conflicting_hard_rules_do_not_short_circuit():
    for text in (
        "为什么这段 Python 代码会报错",
        "Why is this SQL query slow, and how do I fix it?",
    ):
        evidence = engine().evaluate(text)
        assert evidence.unique_hard_label is None
        assert evidence.hard_labels == {RouteLabel.CODE, RouteLabel.REASON}


def test_exclusion_pattern_blocks_a_match(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        yaml.safe_dump(
            {
                "version": "test",
                "rules": [
                    {
                        "id": "python-code-not-snake",
                        "label": "code",
                        "kind": "weak",
                        "pattern_type": "token",
                        "pattern": "python",
                        "exclusions": ["python\\s+(snake|这种蛇)"],
                        "weight": 1.0,
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rules = RuleEngine(rules_path)

    assert rules.evaluate("debug this Python code").weak_hits
    assert not rules.evaluate("tell me about a Python snake").weak_hits


def test_duplicate_rule_ids_are_rejected(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rule = {
        "id": "duplicate",
        "label": "chat",
        "kind": "weak",
        "pattern_type": "exact",
        "pattern": "hello",
    }
    rules_path.write_text(
        yaml.safe_dump({"version": "test", "rules": [rule, rule]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate rule IDs"):
        RuleEngine(rules_path)
