"""
Prometheus 指标收集器

📊 三个核心指标：
  1️⃣ llm_latency_seconds - Histogram，记录每次调用的延迟分布
  2️⃣ llm_calls_total - Counter，记录总调用次数  
  3️⃣ llm_errors_total - Counter，记录错误次数
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps
import time
from typing import Optional, Callable, Any

# 创建自定义 registry（避免与其他应用冲突）
REGISTRY = CollectorRegistry()

# ========================================
# 🟢 1️⃣ llm_latency_seconds - 延迟直方图
# ========================================
# 分布桶：0.05s, 0.1s, 0.5s, 1s, 2s, 5s, 10s
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM 调用延迟（秒）',
    labelnames=['model', 'endpoint', 'status'],
    buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY
)

# ========================================
# 🟢 2️⃣ llm_calls_total - 总调用次数
# ========================================
llm_calls_total = Counter(
    'llm_calls_total',
    '总 LLM 调用次数',
    labelnames=['model', 'endpoint'],
    registry=REGISTRY
)

# ========================================
# 🔴 3️⃣ llm_errors_total - 错误次数
# ========================================
llm_errors_total = Counter(
    'llm_errors_total',
    'LLM 调用错误总数',
    labelnames=['model', 'endpoint', 'error_type'],
    registry=REGISTRY
)

# ========================================
# 🟠 其他有用的指标
# ========================================
llm_tokens_total = Counter(
    'llm_tokens_total',
    'LLM 处理的 token 总数',
    labelnames=['model', 'type'],  # type: input/output
    registry=REGISTRY
)

llm_active_requests = Gauge(
    'llm_active_requests',
    '当前进行中的 LLM 请求数',
    labelnames=['model', 'endpoint'],
    registry=REGISTRY
)

llm_queue_length = Gauge(
    'llm_queue_length',
    'LLM 请求队列长度',
    labelnames=['model'],
    registry=REGISTRY
)


def record_llm_call(model: str, endpoint: str = "default"):
    """
    装饰器：记录 LLM 调用指标
    
    用法：
        @record_llm_call(model="qwen", endpoint="chat")
        def call_llm(prompt):
            return llm.chat(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            llm_calls_total.labels(model=model, endpoint=endpoint).inc()
            llm_active_requests.labels(model=model, endpoint=endpoint).inc()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # 记录延迟（成功）
                llm_latency_seconds.labels(
                    model=model,
                    endpoint=endpoint,
                    status="success"
                ).observe(elapsed)
                
                return result
                
            except TimeoutError as e:
                elapsed = time.time() - start_time
                llm_errors_total.labels(
                    model=model,
                    endpoint=endpoint,
                    error_type="timeout"
                ).inc()
                llm_latency_seconds.labels(
                    model=model,
                    endpoint=endpoint,
                    status="timeout"
                ).observe(elapsed)
                raise
                
            except Exception as e:
                elapsed = time.time() - start_time
                error_type = type(e).__name__
                llm_errors_total.labels(
                    model=model,
                    endpoint=endpoint,
                    error_type=error_type
                ).inc()
                llm_latency_seconds.labels(
                    model=model,
                    endpoint=endpoint,
                    status="error"
                ).observe(elapsed)
                raise
                
            finally:
                llm_active_requests.labels(model=model, endpoint=endpoint).dec()
        
        return wrapper
    return decorator


def record_llm_tokens(model: str, input_tokens: int = 0, output_tokens: int = 0):
    """记录 token 计数"""
    if input_tokens > 0:
        llm_tokens_total.labels(model=model, type='input').inc(input_tokens)
    if output_tokens > 0:
        llm_tokens_total.labels(model=model, type='output').inc(output_tokens)


def record_llm_error(model: str, endpoint: str, error_type: str):
    """手动记录错误"""
    llm_errors_total.labels(
        model=model,
        endpoint=endpoint,
        error_type=error_type
    ).inc()


def get_metrics() -> bytes:
    """获取 Prometheus 格式的指标"""
    return generate_latest(REGISTRY)
