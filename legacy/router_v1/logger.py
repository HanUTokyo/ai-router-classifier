import logging
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger():
    logger = logging.getLogger("ai_gateway")
    logger.setLevel(logging.INFO)

    # ❗防止重复 handler
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(f"{LOG_DIR}/router.log")
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# ✅ 单一 logger
logger = setup_logger()

def log_event(data: dict, trace_id=None):
    event = {
        "ts": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        **data
    }

    if trace_id:
        event["trace_id"] = trace_id

    logger.info(json.dumps(event, ensure_ascii=False))