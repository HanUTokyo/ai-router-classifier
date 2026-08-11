from __future__ import annotations

import re


_CLAUSE_SPLIT = re.compile(r"[。.!?！？;；,，]+")
_ROUTE_LABEL = re.compile(r"(?i)(?<![\w])(?:code|reason|chat)(?![\w])")
_ROUTE_DIRECTIVE = re.compile(
    r"(?i)(output|return|respond|route|router|label|select|emit|always|"
    r"ignore|输出|返回|路由|标成|选择|选|无论|忽略)"
)
_LEADING_CONNECTOR = re.compile(
    r"(?i)^\s*(?:then|and\s+then|actual\s+task\s+is|"
    r"然后|再|接着|实际任务是|真正的任务是|实际请)\s*[:：]?\s*"
)


def normalize_classification_text(text: str) -> tuple[str, bool]:
    """Remove clauses whose only purpose is manipulating the route label."""

    original = text.strip()
    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(original)]
    clauses = [part for part in clauses if part]
    if not clauses:
        return original, False

    kept: list[str] = []
    removed = False
    for clause in clauses:
        if _ROUTE_LABEL.search(clause) and _ROUTE_DIRECTIVE.search(clause):
            removed = True
            continue
        cleaned = _LEADING_CONNECTOR.sub("", clause) if removed else clause
        if cleaned:
            kept.append(cleaned)
    if not removed:
        return original, False
    semantic = " ".join(kept).strip()
    return (semantic or "No semantic task was provided."), True
