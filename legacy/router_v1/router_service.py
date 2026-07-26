try:
    from .router_model import model_route
    from .logger import log_event
    from .rule_engine import RuleEngine
except ImportError:
    from router_model import model_route
    from logger import log_event
    from rule_engine import RuleEngine
try:
    from memory_manager import parse_memory_pairs, read_long_term_memory
except ImportError:
    parse_memory_pairs = None
    read_long_term_memory = None
import time


_RULE_ENGINE = None

MODEL_BY_ROUTE = {
    "chat": "gemma4:e4b",
    "reason": "deepseek-r1:8b",
    "code": "deepseek-coder:6.7b",
}

PROMPT_POLICY_BY_ROUTE = {
    "chat": "memory_aware_chat",
    "reason": "reasoning",
    "code": "engineering_code",
}

RAG_HINTS = (
    "rag",
    "文档",
    "资料",
    "笔记",
    "知识库",
    "检索",
    "搜索",
    "查一下",
    "根据我的",
    "根据之前",
    "之前说过",
    "我们刚才",
    "上面提到",
    "历史记录",
    "memory",
    "notes",
    "docs",
)

CODE_MEMORY_HINTS = (
    "工程师",
    "程序员",
    "开发",
    "代码",
    "编程",
    "python",
    "fastapi",
    "api",
    "router",
    "agent",
)

PRONOUN_CONTEXT_HINTS = (
    "这个",
    "那个",
    "它",
    "继续",
    "上面",
    "刚才",
    "之前",
    "这段",
    "这里",
    "怎么改",
    "怎么做",
)


def get_rule_engine():
    global _RULE_ENGINE
    if _RULE_ENGINE is None:
        _RULE_ENGINE = RuleEngine()
    return _RULE_ENGINE

def fuse_decision(rule_res, model_res):
    rule_label, rule_conf = rule_res
    model_label, model_conf = model_res

    # 情况1：rule强命中 → 直接用
    if rule_conf >= 0.9:
        return rule_label, rule_conf, "rule_strong"

    # 情况2：两者一致 → 提升置信度
    if rule_label and rule_label == model_label:
        return model_label, min(max(rule_conf, model_conf) + 0.05, 1.0), "agree"

    # 情况3：冲突 → 用更高置信度
    if rule_label and model_label:
        if rule_conf > model_conf:
            return rule_label, rule_conf, "rule_win"
        else:
            return model_label, model_conf, "model_win"

    # 情况4：只有model
    if model_label:
        return model_label, model_conf, "model_only"

    # 情况5：只有rule
    if rule_label:
        return rule_label, rule_conf, "rule_only"

    return "chat", 0.3, "fallback"

def router_v2(text, trace_id=None):
    start = time.time()

    # 1️⃣ rule
    rule_engine = get_rule_engine()
    rule_data = rule_engine.match(text)

    rule_res = (
        rule_data["label"],
        rule_data["confidence"]
    )

    # 2️⃣ model
    model_res = model_route(text, trace_id)

    # 3️⃣ fusion（核心）
    final_label, final_conf, source = fuse_decision(rule_res, model_res)

    latency = time.time() - start

    log_event({
        "type": "router_v2",
        "input": text,
        "rule": {
            "label": rule_res[0],
            "confidence": rule_res[1],
            "hits": rule_data["hits"],
            "score": rule_data["score"]
        },
        "model": {
            "label": model_res[0],
            "confidence": model_res[1]
        },
        "final": {
            "label": final_label,
            "confidence": final_conf,
            "source": source
        },
        "latency": round(latency, 4)
    }, trace_id)

    return final_label, final_conf, source

def route_text(text: str, trace_id: str):
    start = time.time()

    # 🚀 统一走 v2
    route, confidence, source = router_v2(text, trace_id)

    latency = time.time() - start

    log_event({
        "type": "router_decision",
        "route": route,
        "confidence": confidence,
        "source": source,
        "latency": round(latency, 4),
        "input_len": len(text)
    }, trace_id)

    return {
        "route": route,
        "confidence": confidence,
        "source": source
    }


def _normalize_message(msg):
    if not isinstance(msg, dict):
        return None
    role = str(msg.get("role", "")).strip()
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content = "".join(parts)
    elif content is None:
        content = ""
    else:
        content = str(content)
    if not role and not content:
        return None
    return {"role": role or "user", "content": content}


def _extract_routing_text(message=None, messages=None):
    normalized = [_normalize_message(m) for m in (messages or [])]
    normalized = [m for m in normalized if m]
    for msg in reversed(normalized):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            return msg["content"].strip(), normalized
    if message:
        return str(message).strip(), normalized
    if normalized:
        return normalized[-1].get("content", "").strip(), normalized
    return "", normalized


def _recent_context_summary(messages, limit=6):
    recent = messages[-limit:]
    summary = []
    for msg in recent:
        content = msg.get("content", "").strip().replace("\n", " ")
        if len(content) > 160:
            content = content[:160]
        summary.append({"role": msg.get("role", "user"), "content": content})
    return summary


def _memory_context_summary():
    if not parse_memory_pairs or not read_long_term_memory:
        return {"available": False, "keys": [], "code_bias": False}
    pairs = parse_memory_pairs(read_long_term_memory())
    keys = list(pairs.keys())[-20:]
    combined = " ".join([str(k) + " " + str(v) for k, v in pairs.items()]).lower()
    code_bias = any(hint.lower() in combined for hint in CODE_MEMORY_HINTS)
    return {
        "available": True,
        "keys": keys,
        "code_bias": code_bias,
    }


def _needs_rag(text, messages):
    lowered = text.lower()
    if any(hint.lower() in lowered for hint in RAG_HINTS):
        return True, "explicit_rag_hint"
    if messages and len(text) <= 12 and any(hint in text for hint in PRONOUN_CONTEXT_HINTS):
        return True, "context_dependent_short_query"
    if "最新" in text or "现在" in text and ("价格" in text or "版本" in text or "新闻" in text):
        return True, "possibly_time_sensitive"
    return False, "not_required"


def _apply_context_bias(route, confidence, source, text, messages, memory_summary):
    context_text = " ".join(m.get("content", "") for m in messages[-4:])
    combined = f"{context_text} {text}".lower()

    code_terms = ("代码", "报错", "函数", "接口", "python", "fastapi", "api", "sql", "docker", "router")
    reason_terms = ("为什么", "推理", "分析", "证明", "原因", "逻辑")
    has_code_signal = any(term in combined for term in code_terms)
    has_reason_signal = any(term in combined for term in reason_terms)

    if has_reason_signal and not has_code_signal:
        return "reason", max(confidence, 0.7), f"{source}+context_reason"
    if route == "chat" and has_code_signal:
        return "code", max(confidence, 0.65), f"{source}+context_code"
    if route == "chat" and memory_summary.get("code_bias") and any(term in text.lower() for term in ("项目", "实现", "改造", "修复")):
        return "code", max(confidence, 0.6), f"{source}+memory_code_bias"
    return route, confidence, source


def build_route_decision(message=None, messages=None, trace_id=None):
    start = time.time()
    routing_text, normalized_messages = _extract_routing_text(message, messages)
    if not routing_text:
        routing_text = ""

    base = route_text(routing_text, trace_id)
    memory_summary = _memory_context_summary()
    route, confidence, source = _apply_context_bias(
        base.get("route", "chat"),
        base.get("confidence", 0.0),
        base.get("source", "router_api"),
        routing_text,
        normalized_messages,
        memory_summary,
    )
    needs_rag, rag_reason = _needs_rag(routing_text, normalized_messages)
    prompt_policy = "rag_required" if needs_rag else PROMPT_POLICY_BY_ROUTE.get(route, "memory_aware_chat")
    decision = {
        "route": route,
        "model": MODEL_BY_ROUTE.get(route, MODEL_BY_ROUTE["chat"]),
        "confidence": confidence,
        "source": source,
        "needs_rag": needs_rag,
        "prompt_policy": prompt_policy,
        "context_used": {
            "routing_text": routing_text,
            "recent_messages": _recent_context_summary(normalized_messages),
            "memory": memory_summary,
            "rag_reason": rag_reason,
        },
    }

    log_event({
        "type": "router_context_decision",
        "decision": {
            "route": decision["route"],
            "model": decision["model"],
            "confidence": decision["confidence"],
            "source": decision["source"],
            "needs_rag": decision["needs_rag"],
            "prompt_policy": decision["prompt_policy"],
        },
        "context": decision["context_used"],
        "latency": round(time.time() - start, 4),
    }, trace_id)

    return decision
