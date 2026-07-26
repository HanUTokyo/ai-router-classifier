import requests
try:
    from .logger import log_event
    from .rule_engine import RuleEngine
except ImportError:
    from logger import log_event
    from rule_engine import RuleEngine
import os
import time

OLLAMA_URL = os.getenv("ROUTER_OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("ROUTER_MODEL", "qwen2.5:0.2b")


_FALLBACK_RULE_ENGINE = None


def get_fallback_rule_engine():
    global _FALLBACK_RULE_ENGINE
    if _FALLBACK_RULE_ENGINE is None:
        _FALLBACK_RULE_ENGINE = RuleEngine()
    return _FALLBACK_RULE_ENGINE

SYSTEM_PROMPT = """
你是一个严格的分类器（router）。

任务：将用户输入分类为以下三类之一：

code:
涉及编程、代码编写、调试、SQL、API、脚本、技术实现的问题

reason:
涉及数学、逻辑推理、原因分析、判断、解释“为什么”的问题

chat:
普通聊天、简单问答、寒暄、非技术讨论

分类规则（非常重要）：
- 只要涉及代码/SQL/API，一律归为 code（优先级最高）
- 只有在“纯分析/推理且不涉及代码”时才归为 reason
- 其他全部归为 chat

输出要求（必须遵守）：
- 只能输出一个单词
- 不允许输出解释
- 不允许输出句子
- 不允许包含标点或多余内容

错误示例（禁止）：
"这是code类型"
"我认为是reason"
"chat类型问题"
"code，因为..."

正确示例：
code
reason
chat
"""

def normalize_output(raw: str):
    raw = raw.strip().lower()

    # 只取第一行 + 去掉多余内容
    raw = raw.split("\n")[0].strip()

    return raw

def map_output(raw: str):
    # 强匹配
    if raw in ["code", "reason", "chat"]:
        return raw, 0.9

    # 弱匹配（容错）
    if "code" in raw.split():
        return "code", 0.6
    if "reason" in raw.split():
        return "reason", 0.6
    if "chat" in raw.split():
        return "chat", 0.6

    return None, 0.0

def call_llm(text):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload, timeout=10)
    res.raise_for_status()
    data = res.json()

    # Ollama在模型不存在时会返回 {"error": "..."}，避免后续 KeyError('message')。
    if "error" in data:
        raise ValueError(f"ollama_error:{data['error']}")

    message = data.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError("invalid_ollama_payload:missing_message_content")

    return message["content"]

def model_route(text, trace_id=None):
    start = time.time()

    try:
        for attempt in range(3):
            attempt_start = time.time()

            raw = call_llm(text)
            normalized = normalize_output(raw)
            result, confidence = map_output(normalized)

            attempt_latency = time.time() - attempt_start

            # 🔥 每一次 attempt 都记录
            log_event({
                "type": "llm_attempt",
                "model": MODEL,
                "input_text": text,
                "raw_output": raw,
                "normalized_output": normalized,
                "mapped_result": result,
                "confidence": confidence,
                "attempt": attempt + 1,
                "attempt_latency": round(attempt_latency, 4)
            }, trace_id)

            if result:
                total_latency = time.time() - start

                log_event({
                    "type": "llm_success",
                    "model": MODEL,
                    "final_route": result,
                    "confidence": confidence,
                    "attempt": attempt + 1,
                    "total_latency": round(total_latency, 4)
                }, trace_id)

                return result, confidence

        # fallback
        fallback_rule_data = get_fallback_rule_engine().match(text)
        fallback_result = fallback_rule_data["label"] or "chat"
        fallback_conf = fallback_rule_data["confidence"] if fallback_rule_data["label"] else 0.3

        log_event({
            "type": "llm_fallback",
            "input_text": text,
            "fallback_route": fallback_result,
            "fallback_confidence": fallback_conf,
            "fallback_hits": fallback_rule_data.get("hits", [])
        }, trace_id)

        return fallback_result, fallback_conf

    except Exception as e:
        latency = time.time() - start

        log_event({
            "type": "llm_error",
            "model": MODEL,
            "error": str(e),
            "latency": round(latency, 4)
        }, trace_id)

        return "chat", 0.2
