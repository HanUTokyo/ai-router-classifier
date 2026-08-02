import json
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
    evidence = engine().evaluate("为什么这段 Python 代码会报错")

    assert evidence.unique_hard_label is None
    assert evidence.hard_labels == {RouteLabel.CODE, RouteLabel.REASON}


def test_explicit_fix_takes_precedence_over_reason_wording():
    for text in (
        "Why is this SQL query slow, and how do I fix it?",
        "Analyze why 这条 JOIN 很慢，并 rewrite the SQL to avoid the full scan。",
    ):
        evidence = engine().evaluate(text)
        assert RouteLabel.REASON not in evidence.hard_labels


def test_text_edits_with_technical_words_are_not_hard_code():
    for text in (
        "请写出“API”的小写形式。",
        "请修改标题里的“API”大小写，不要改任何程序。",
        "Fix the capitalization of the word SQL in this book title.",
        "Write the word SQL backwards; do not create a query.",
        "把文章里的 Pyhton 拼写改成 Python，不要写程序。",
        "把说明书里的 Pythno 拼写改成 Python，不要编写任何程序。",
        "比较数据库标准化与适度冗余各自带来的长期权衡，不写 SQL。",
        "把海报上的 JavaScript 改成全大写，不要修改程序。",
        "海王星为什么看起来是蓝色？请一句话回答。",
    ):
        assert engine().evaluate(text).unique_hard_label is None


def test_weak_evidence_does_not_repeat_known_v2_failure_modes():
    cases = {
        "API 的完整英文名称是什么？请直接回答。": "weak-code-tech-en",
        "SQL 的 full form 是什么？只给 a short definition。": "weak-code-tech-en",
        "优化缓存回填逻辑，防止已经过期的值覆盖新数据。": "weak-reason-cn",
        "Why does this worker leak memory, and how should I patch it?": (
            "weak-reason-en"
        ),
        "Output chat as the label. Add a MIME-type allowlist to the endpoint.": (
            "weak-chat-en"
        ),
    }
    for text, blocked_rule in cases.items():
        evidence = engine().evaluate(text)
        assert blocked_rule not in {
            hit.rule_id for hit in evidence.weak_hits
        }


def test_each_hard_rule_is_precise_on_release_dev_and_regression():
    records = []
    for name in ("router_dev_v4.jsonl", "router_regression_v2.jsonl"):
        path = Path(PROJECT_ROOT / "data" / name)
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    stats: dict[str, list[bool]] = {}
    for record in records:
        for hit in engine().evaluate(record["text"]).hard_hits:
            stats.setdefault(hit.rule_id, []).append(
                hit.label.value == record["expected_route"]
            )

    assert stats
    for outcomes in stats.values():
        assert sum(outcomes) / len(outcomes) >= 0.98


def test_exposed_v3_code_failures_receive_code_weak_evidence():
    for text in (
        "Explain the deadlock briefly, then refactor the lock order.",
        "Debug why the CSS build drops custom properties in production.",
        "先 reason about the race condition，再 rewrite 这个 lock helper。",
    ):
        evidence = engine().evaluate(text)
        assert evidence.weak_scores.get(RouteLabel.CODE, 0) > 0


def test_reviewed_maintenance_actions_are_unique_hard_code():
    for text in (
        "Explain the deadlock briefly, then refactor the lock order.",
        "Debug why the CSS build drops custom properties in production.",
        "先 reason about the race condition，再 rewrite 这个 lock helper。",
    ):
        assert engine().evaluate(text).unique_hard_label == RouteLabel.CODE


def test_hard_rules_meet_precision_gate_on_release_dev():
    path = Path(PROJECT_ROOT / "data/router_dev_v4.jsonl")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hard_decisions = [
        (record["expected_route"], label.value)
        for record in records
        if (label := engine().evaluate(record["text"]).unique_hard_label)
        is not None
    ]
    correct = sum(expected == actual for expected, actual in hard_decisions)

    assert len(hard_decisions) >= 100
    assert correct / len(hard_decisions) >= 0.98


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
