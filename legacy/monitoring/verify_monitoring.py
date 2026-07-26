#!/usr/bin/env python3
"""
快速验证 LLM 监控系统
验证指标端点和 Prometheus 配置
"""

import requests
import json
import time
import sys
from typing import Dict, List, Tuple

# 服务端点
SERVICES = {
    "mlx-openai-agent": "http://localhost:8000",
    "local-agent": "http://localhost:8001",
    "router-classifier": "http://localhost:8080",
    "prometheus": "http://localhost:9090",
    "grafana": "http://localhost:3001",
}

def check_service_health(name: str, url: str) -> Tuple[bool, str]:
    """检查服务健康状态"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        if response.status_code == 200:
            return True, "✅ 健康"
        else:
            return False, f"❌ 状态码: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ 无法连接"
    except Exception as e:
        return False, f"❌ 错误: {str(e)}"

def check_metrics_endpoint(name: str, url: str) -> Tuple[bool, int]:
    """检查 /metrics 端点"""
    try:
        response = requests.get(f"{url}/metrics", timeout=2)
        if response.status_code == 200:
            # 验证是否包含 Prometheus 格式的指标
            content = response.text
            if "llm_latency_seconds" in content or "# HELP" in content:
                return True, len(content)
            else:
                return False, 0
        else:
            return False, response.status_code
    except Exception as e:
        return False, 0

def check_prometheus_targets() -> Tuple[bool, List[Dict]]:
    """检查 Prometheus 目标"""
    try:
        response = requests.get(
            "http://localhost:9090/api/v1/targets",
            timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            targets = data.get("data", {}).get("activeTargets", [])
            return True, targets
        else:
            return False, []
    except Exception as e:
        return False, []

def main():
    print("=" * 60)
    print("🤖 LLM 监控系统验证")
    print("=" * 60)
    
    # 1. 检查服务健康状态
    print("\n📋 1️⃣ 检查服务健康状态")
    print("-" * 60)
    
    for name, url in SERVICES.items():
        if name in ["prometheus", "grafana"]:
            continue
        healthy, msg = check_service_health(name, url)
        status = "✅" if healthy else "❌"
        print(f"{status} {name:20} {msg}")
    
    # 2. 检查 /metrics 端点
    print("\n📊 2️⃣ 检查 /metrics 端点")
    print("-" * 60)
    
    for name, url in SERVICES.items():
        if name in ["prometheus", "grafana"]:
            continue
        has_metrics, size = check_metrics_endpoint(name, url)
        status = "✅" if has_metrics else "❌"
        if has_metrics:
            print(f"{status} {name:20} 返回 {size:,} 字节")
        else:
            print(f"{status} {name:20} 端点不可用")
    
    # 3. 检查 Prometheus 目标
    print("\n🔍 3️⃣ 检查 Prometheus 目标")
    print("-" * 60)
    
    has_targets, targets = check_prometheus_targets()
    if has_targets:
        print(f"✅ 发现 {len(targets)} 个活跃目标:")
        for target in targets:
            labels = target.get("labels", {})
            job = labels.get("job", "unknown")
            instance = labels.get("instance", "unknown")
            health = target.get("health", "unknown")
            health_icon = "✅" if health == "up" else "❌"
            print(f"   {health_icon} {job:20} {instance}")
    else:
        print("❌ 无法连接到 Prometheus")
    
    # 4. 检查指标是否被收集
    print("\n📈 4️⃣ 检查指标收集")
    print("-" * 60)
    
    metrics_to_check = [
        "llm_latency_seconds_bucket",
        "llm_calls_total",
        "llm_errors_total",
        "llm_active_requests",
        "llm_tokens_total",
    ]
    
    try:
        response = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": "up"},
            timeout=2
        )
        if response.status_code == 200:
            print("✅ Prometheus 响应正常")
            
            # 检查每个指标
            for metric in metrics_to_check:
                try:
                    response = requests.get(
                        "http://localhost:9090/api/v1/query",
                        params={"query": metric},
                        timeout=2
                    )
                    if response.status_code == 200:
                        data = response.json()
                        result_count = len(data.get("data", {}).get("result", []))
                        if result_count > 0:
                            print(f"   ✅ {metric:30} 已收集 ({result_count} 条时间序列)")
                        else:
                            print(f"   ⚠️  {metric:30} 尚未收集数据")
                except:
                    print(f"   ❌ {metric:30} 查询失败")
        else:
            print("❌ Prometheus 响应异常")
    except Exception as e:
        print(f"❌ 无法连接到 Prometheus: {e}")
    
    # 5. 访问 URL
    print("\n🌐 5️⃣ 快速链接")
    print("-" * 60)
    print("📊 Prometheus: http://localhost:9090")
    print("📈 Grafana:    http://localhost:3001 (admin/admin)")
    print("🔍 Query: http://localhost:9090/graph")
    
    print("\n💡 下一步:")
    print("1. 生成一些 LLM 调用来创建指标数据")
    print("2. 在 Grafana 中查看 LLM Monitoring Dashboard")
    print("3. 使用 PromQL 进行自定义查询")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
