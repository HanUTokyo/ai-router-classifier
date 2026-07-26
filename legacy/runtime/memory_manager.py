import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

MEM_ROOT = "/opt/ai/Router/memory"
LONG_TERM_FILE = os.path.join(MEM_ROOT, "MEMORY.md")
DAILY_DIR = os.path.join(MEM_ROOT, "daily")

os.makedirs(MEM_ROOT, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)


def read_long_term_memory() -> str:
    if not os.path.exists(LONG_TERM_FILE):
        return ""
    with open(LONG_TERM_FILE, "r", encoding="utf-8") as f:
        return f.read()


def read_today_memory() -> str:
    path = os.path.join(DAILY_DIR, f"{datetime.now().date()}.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_daily_memory(text: str) -> None:
    path = os.path.join(DAILY_DIR, f"{datetime.now().date()}.md")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{text}\n")


def write_long_term_memory(text: str) -> None:
    with open(LONG_TERM_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{text}\n")


def append_long_term_memory_pair(key: str, value: str) -> None:
    key_clean = key.strip()
    value_clean = value.strip()
    if not key_clean or not value_clean:
        return
    write_long_term_memory(f"{key_clean}={value_clean}")


def parse_memory_pairs(text: str) -> Dict[str, str]:
    """Parse key=value lines from memory text. Later entries override earlier ones."""
    pairs: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([^=\n]{1,120}?)\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip()
        if key and value and not _is_question_like_value(value):
            pairs[key] = value
    return pairs


def extract_kv_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        m = re.match(r"^\s*([^=\n]{1,120}?)\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip()
        if key and value and not _is_question_like_value(value):
            pairs.append((key, value))
    return pairs


def extract_auto_memory_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract simple long-term memory from common user statements."""
    results: Dict[str, str] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for content in lines:
        patterns = [
            r"^我叫\s*([^，。,.!?？\n]{1,40})",
            r"^我的名字是\s*([^，。,.!?？\n]{1,40})",
        ]

        for pattern in patterns:
            m = re.match(pattern, content)
            if m:
                results["名字"] = m.group(1).strip()
                break

        m = re.match(r"^我喜欢(?:吃)?\s*([^，。,.!?？\n]{1,40})", content)
        if m:
            candidate = m.group(1).strip()
            if not _is_question_like_value(candidate):
                results["喜欢"] = candidate

        m = re.match(r"^我住在\s*([^，。,.!?？\n]{1,40})", content)
        if m:
            candidate = m.group(1).strip()
            if not _is_question_like_value(candidate):
                results["居住地"] = candidate

    return list(results.items())


def _is_question_like_value(value: str) -> bool:
    v = value.strip()
    if v in {"?", "？"}:
        return True
    question_tokens = ["什么", "几", "多少", "?", "？"]
    return any(token in v for token in question_tokens)


def _extract_lookup_key(user_text: str) -> Optional[str]:
    text = user_text.strip()

    patterns = [
        r"^\s*([\u4e00-\u9fffA-Za-z0-9_\-]{1,40})\s*等于(?:什么|几|多少)\s*[？?]?\s*$",
        r"^\s*([\u4e00-\u9fffA-Za-z0-9_\-]{1,40})\s*是(?:什么|几|多少)\s*[？?]?\s*$",
        r"^\s*([\u4e00-\u9fffA-Za-z0-9_\-]{1,40})\s*=\s*[？?]?\s*$",
    ]

    for pattern in patterns:
        m = re.match(pattern, text)
        if m:
            return m.group(1).strip()

    return None


def recall_memory_value(user_text: str) -> Optional[str]:
    """Return remembered value for questions like '香蕉等于什么' if present."""
    text = user_text.strip()
    memory_text = read_long_term_memory()
    pairs = parse_memory_pairs(memory_text)

    # Natural-language preference recall.
    if re.match(r"^\s*我喜欢(?:吃)?什么\s*[？?]?\s*$", text):
        return pairs.get("喜欢")

    key = _extract_lookup_key(text)
    if not key:
        return None

    return pairs.get(key)
