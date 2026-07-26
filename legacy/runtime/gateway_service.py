import os
import json
import re
import requests

from memory_manager import (
    append_long_term_memory_pair,
    extract_auto_memory_pairs_from_text,
    extract_kv_pairs_from_text,
    parse_memory_pairs,
    recall_memory_value,
    read_long_term_memory,
    read_today_memory,
    write_daily_memory,
)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.31.216:11434/api/chat")
ROUTER_ROUTE_URL = os.getenv("ROUTER_ROUTE_URL", "http://localhost:8001/route")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:e4b")
REASON_MODEL = os.getenv("REASON_MODEL", "deepseek-r1:8b")
CODE_MODEL = os.getenv("CODE_MODEL", "deepseek-coder:6.7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
TOOL_CALL_MODEL = os.getenv("TOOL_CALL_MODEL", CHAT_MODEL)
TOOL_CALL_SUPPORTED_MODELS = {
    model.strip()
    for model in os.getenv("TOOL_CALL_SUPPORTED_MODELS", "").split(",")
    if model.strip()
}
UPSTREAM_MEMORY_MODE = os.getenv("UPSTREAM_MEMORY_MODE", "local").strip().lower()

GEMMA_NUM_CTX = int(os.getenv("GEMMA_NUM_CTX", "131072"))
GEMMA_TEMPERATURE = float(os.getenv("GEMMA_TEMPERATURE", "1.0"))
GEMMA_TOP_P = float(os.getenv("GEMMA_TOP_P", "0.95"))
GEMMA_TOP_K = int(os.getenv("GEMMA_TOP_K", "64"))

GEMMA_OPTIONS = {
    "temperature": GEMMA_TEMPERATURE,
    "top_p": GEMMA_TOP_P,
    "top_k": GEMMA_TOP_K,
    "num_ctx": GEMMA_NUM_CTX,
}

VALID_MEMORY_MODES = {"anythingllm", "local", "hybrid"}

# Gemma 4 may emit internal thought blocks in these wrappers; remove them from history.
_THINK_BLOCK_RE = re.compile(
    r"<\|channel\|?>thought\n.*?(?:<\|channel\|>|<channel\|>)",
    re.DOTALL,
)


def _sanitize_recent_memory_lines(text: str) -> str:
    noise_prefixes = (
        "Conversation info",
        "Sender",
        "Please save this",
        "Reply only",
        "```",
        "{",
        "}",
        '"',
    )
    cleaned = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "untrusted metadata" in line.lower():
            continue
        if any(line.startswith(prefix) for prefix in noise_prefixes):
            continue
        cleaned.append(line)
    if not cleaned:
        return "(empty)"
    return "\n".join(cleaned[-30:])


def _build_memory_system_prompt() -> dict:
    long_mem = read_long_term_memory()
    today_mem = read_today_memory()
    memory_pairs = parse_memory_pairs(long_mem)

    if memory_pairs:
        pair_lines = [f"- {k}={v}" for k, v in memory_pairs.items()]
        long_mem_block = "\n".join(pair_lines[-50:])
    else:
        long_mem_block = "(empty)"

    today_block = _sanitize_recent_memory_lines(today_mem)

    memory_block = (
        "You have access to persistent memory.\n"
        "Use it as facts about the user when relevant.\n"
        "If memory conflicts with the current user message, trust the current message.\n\n"
        "Long-term memory (key=value):\n"
        f"{long_mem_block}\n\n"
        "Recent memory:\n"
        f"{today_block}"
    ).strip()

    return {"role": "system", "content": memory_block}


def _inject_memory_system_prompt(messages: list) -> list:
    if _memory_mode() != "local":
        return messages

    memory_prompt = _build_memory_system_prompt()["content"]
    merged_messages = []
    merged = False

    for msg in messages:
        if not merged and msg.get("role") == "system":
            merged_content = f"{memory_prompt}\n\n---\n\n{msg.get('content', '')}".strip()
            merged_messages.append({"role": "system", "content": merged_content})
            merged = True
        else:
            merged_messages.append(msg)

    if not merged:
        return [{"role": "system", "content": memory_prompt}] + merged_messages

    return merged_messages


def _memory_mode() -> str:
    if UPSTREAM_MEMORY_MODE in VALID_MEMORY_MODES:
        return UPSTREAM_MEMORY_MODE
    return "local"


def get_memory_mode() -> str:
    return _memory_mode()


def _should_write_local_memory() -> bool:
    return _memory_mode() in {"local", "hybrid"}


def _should_answer_from_local_memory() -> bool:
    return _memory_mode() == "local"


def _apply_memory_write_rules(last_user_message: str) -> None:
    if not _should_write_local_memory():
        return

    if not last_user_message:
        return

    user_text = last_user_message.lower()

    # Rule 1: explicit save intent.
    if "save" in user_text or "记住" in user_text:
        write_daily_memory(last_user_message)

    # Rule 2: explicit key-value memory, line by line.
    kv_pairs = extract_kv_pairs_from_text(last_user_message)
    for key, value in kv_pairs:
        append_long_term_memory_pair(key, value)

    # Rule 3: minimal automatic memory extraction.
    auto_pairs = extract_auto_memory_pairs_from_text(last_user_message)
    for key, value in auto_pairs:
        append_long_term_memory_pair(key, value)


def _memory_lookup_stream(answer: str):
    def iter_events():
        yield None, json.dumps({"message": {"content": answer}, "done": False}, ensure_ascii=False)
        yield None, json.dumps({"done": True}, ensure_ascii=False)

    return iter_events()


def _strip_thinking_blocks(messages: list) -> list:
    cleaned_messages = []
    for msg in messages:
        if msg.get("role") != "assistant":
            cleaned_messages.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            cleaned_messages.append(msg)
            continue

        cleaned_content = _THINK_BLOCK_RE.sub("", content).strip()
        cleaned_msg = dict(msg)
        cleaned_msg["content"] = cleaned_content
        cleaned_messages.append(cleaned_msg)

    return cleaned_messages


def choose_model(route_label: str) -> str:
    model_map = {
        "code": CODE_MODEL,
        "reason": REASON_MODEL,
        "chat": CHAT_MODEL,
    }
    return model_map.get(route_label, CHAT_MODEL)


def ask_ollama(model: str, text: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "options": GEMMA_OPTIONS,
        "stream": False,
    }
    res = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    res.raise_for_status()
    data = res.json()
    return data["message"]["content"]


def ask_ollama_messages(model: str, messages: list) -> str:
    """Send full message history to Ollama (for multi-turn or system prompt support)."""
    message = ask_ollama_messages_detail(model, messages)
    return message.get("content", "")


def _build_ollama_chat_payload(
    model: str,
    messages: list,
    stream: bool,
    tools: list | None = None,
) -> dict:
    cleaned_messages = _strip_thinking_blocks(messages)
    payload = {
        "model": model,
        "messages": cleaned_messages,
        "options": GEMMA_OPTIONS,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    return payload


def ask_ollama_messages_detail(
    model: str,
    messages: list,
    tools: list | None = None,
    tool_choice=None,
) -> dict:
    """Send full message history to Ollama and return the raw assistant message."""
    _ = tool_choice
    payload = _build_ollama_chat_payload(
        model,
        messages,
        stream=False,
        tools=tools,
    )
    res = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    res.raise_for_status()
    data = res.json()

    message = data.get("message")
    if not isinstance(message, dict):
        raise ValueError("invalid_ollama_payload:missing_message")
    return message


def ask_ollama_messages_stream(
    model: str,
    messages: list,
    tools: list | None = None,
    tool_choice=None,
):
    """Stream full message history to Ollama and yield decoded JSON events."""
    _ = tool_choice
    payload = _build_ollama_chat_payload(
        model,
        messages,
        stream=True,
        tools=tools,
    )
    res = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=None,
        stream=True,
    )
    res.raise_for_status()

    def iter_events():
        try:
            for line in res.iter_lines(decode_unicode=True):
                if not line:
                    continue
                yield res, line
        finally:
            res.close()

    return iter_events()


def _extract_routing_text(messages: list) -> str:
    routing_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            routing_text = msg.get("content", "")
            break
    if not routing_text:
        routing_text = messages[-1].get("content", "") if messages else ""
    return routing_text


def resolve_route_for_messages(messages: list):
    routing_text = _extract_routing_text(messages)

    try:
        route_result = route_by_dispatcher_api(routing_text)
    except Exception as exc:
        route_result = {
            "route": "chat",
            "confidence": 0.2,
            "source": f"router_unavailable:{exc}",
        }

    route_label = route_result.get("route", "chat")
    model = choose_model(route_label)
    source = route_result.get("source", "router_api")

    return route_result, route_label, model, source


def _resolve_execution_model(route_model: str, tools_present: bool):
    if not tools_present:
        return route_model, "route_model"
    if route_model in TOOL_CALL_SUPPORTED_MODELS:
        return route_model, "route_model_tool_supported"
    return TOOL_CALL_MODEL, "tool_call_model"


def _prepare_routed_request(
    messages: list,
    last_user_message: str,
    tools: list | None,
    force_route_label: str | None = None,
):
    if force_route_label:
        route_label = force_route_label
        route_result = {
            "route": route_label,
            "confidence": 1.0,
            "source": "forced_debug_model",
        }
        route_model = choose_model(route_label)
        source = "forced_debug_model"
    else:
        route_result, route_label, route_model, source = resolve_route_for_messages(messages)

    model, model_source = _resolve_execution_model(route_model, bool(tools))
    if model_source != "route_model":
        source = f"{source}+{model_source}"

    messages_for_model = _inject_memory_system_prompt(messages)

    return route_result, route_label, model, source, messages_for_model


def ask_with_routing_messages(
    messages: list,
    trace_id: str,
    tools: list | None = None,
    tool_choice=None,
    force_route_label: str | None = None,
):
    """Ollama-compatible routing: extract last user message for routing,
    then call selected model with full message history."""
    _ = trace_id

    last_user_message = _extract_routing_text(messages)
    _apply_memory_write_rules(last_user_message)

    memory_value = (
        recall_memory_value(last_user_message)
        if _should_answer_from_local_memory()
        else None
    )
    if memory_value is not None:
        return {
            "route": "chat",
            "model": "memory",
            "confidence": 1.0,
            "source": "memory_lookup",
            "answer": memory_value,
            "memory_mode": _memory_mode(),
        }

    route_result, route_label, model, source, messages_for_model = _prepare_routed_request(
        messages,
        last_user_message,
        tools,
        force_route_label=force_route_label,
    )

    try:
        message = ask_ollama_messages_detail(
            model,
            messages_for_model,
            tools=tools,
            tool_choice=tool_choice,
        )
    except Exception as exc:
        source = "model_error_fallback"
        if model != CHAT_MODEL:
            try:
                message = ask_ollama_messages_detail(
                    CHAT_MODEL,
                    messages_for_model,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                model = CHAT_MODEL
            except Exception:
                message = {"content": f"Ollama request failed: {exc}"}
        else:
            message = {"content": f"Ollama request failed: {exc}"}

    return {
        "route": route_label,
        "model": model,
        "confidence": route_result.get("confidence", 0.0),
        "source": source,
        "answer": message.get("content", ""),
        "tool_calls": message.get("tool_calls"),
        "memory_mode": _memory_mode(),
    }


def ask_with_routing_messages_stream(
    messages: list,
    trace_id: str,
    tools: list | None = None,
    tool_choice=None,
    force_route_label: str | None = None,
):
    _ = trace_id

    last_user_message = _extract_routing_text(messages)
    _apply_memory_write_rules(last_user_message)

    memory_value = (
        recall_memory_value(last_user_message)
        if _should_answer_from_local_memory()
        else None
    )
    if memory_value is not None:
        return {
            "route": "chat",
            "model": "memory",
            "confidence": 1.0,
            "source": "memory_lookup",
            "stream": _memory_lookup_stream(memory_value),
            "memory_mode": _memory_mode(),
        }

    route_result, route_label, model, source, messages_for_model = _prepare_routed_request(
        messages,
        last_user_message,
        tools,
        force_route_label=force_route_label,
    )

    try:
        stream = ask_ollama_messages_stream(
            model,
            messages_for_model,
            tools=tools,
            tool_choice=tool_choice,
        )
    except Exception as exc:
        source = "model_error_fallback"
        if model != CHAT_MODEL:
            stream = ask_ollama_messages_stream(
                CHAT_MODEL,
                messages_for_model,
                tools=tools,
                tool_choice=tool_choice,
            )
            model = CHAT_MODEL
        else:
            raise exc

    return {
        "route": route_label,
        "model": model,
        "confidence": route_result.get("confidence", 0.0),
        "source": source,
        "stream": stream,
        "memory_mode": _memory_mode(),
    }


def route_by_dispatcher_api(text: str):
    payload = {"message": text}
    res = requests.post(ROUTER_ROUTE_URL, json=payload, timeout=10)
    res.raise_for_status()
    data = res.json()

    if not isinstance(data, dict):
        raise ValueError("invalid_dispatcher_response")

    return {
        "route": data.get("route", "chat"),
        "confidence": data.get("confidence", 0.0),
        "source": data.get("source", "router_api"),
    }


def ask_with_routing(text: str, trace_id: str):
    _ = trace_id

    _apply_memory_write_rules(text)
    memory_value = recall_memory_value(text) if _should_answer_from_local_memory() else None
    if memory_value is not None:
        return {
            "route": "chat",
            "model": "memory",
            "confidence": 1.0,
            "source": "memory_lookup",
            "answer": memory_value,
            "memory_mode": _memory_mode(),
        }

    try:
        route_result = route_by_dispatcher_api(text)
    except Exception as exc:
        route_result = {
            "route": "chat",
            "confidence": 0.2,
            "source": f"router_unavailable:{exc}",
        }

    route_label = route_result.get("route", "chat")
    model = choose_model(route_label)

    messages_with_memory = _inject_memory_system_prompt(
        [{"role": "user", "content": text}]
    )

    source = route_result.get("source", "router_api")
    try:
        answer = ask_ollama_messages(model, messages_with_memory)
    except Exception as exc:
        source = "model_error_fallback"
        if model != CHAT_MODEL:
            try:
                answer = ask_ollama_messages(CHAT_MODEL, messages_with_memory)
                model = CHAT_MODEL
            except Exception:
                answer = f"Ollama request failed: {exc}"
        else:
            answer = f"Ollama request failed: {exc}"

    return {
        "route": route_label,
        "model": model,
        "confidence": route_result.get("confidence", 0.0),
        "source": source,
        "answer": answer,
        "memory_mode": _memory_mode(),
    }
