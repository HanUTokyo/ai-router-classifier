"""
Gateway Client for openclaw
客户端库：让 openclaw 通过网关访问 Ollama 大模型
"""
import os
import requests
from typing import Optional


class GatewayClient:
    """网关智能路由客户端"""

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        初始化网关客户端

        Args:
            gateway_url: 网关地址，默认从 GATEWAY_URL 环境变量读取，
                        格式为 http://ip:port (不包括 /ask 路径)
                        示例: http://10.0.3.1:8000
            timeout: 请求超时秒数，默认 120
        """
        self.gateway_url = gateway_url or os.getenv(
            "GATEWAY_URL", "http://10.0.3.1:8000"
        )
        self.ask_endpoint = f"{self.gateway_url}/ask"
        self.timeout = timeout

    def ask(self, message: str) -> str:
        """
        发送问题到网关并获取答案

        Args:
            message: 用户问题

        Returns:
            大模型回答

        Raises:
            requests.RequestException: 网络请求失败
            ValueError: 网关返回无效响应
        """
        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        payload = {"message": message}

        try:
            resp = requests.post(
                self.ask_endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Gateway request failed: {e}"
            ) from e

        data = resp.json()

        # 验证返回结构
        if "answer" not in data:
            raise ValueError(f"Invalid gateway response: {data}")

        return data["answer"]

    def ask_with_routing(self, message: str) -> dict:
        """
        发送问题并获取完整路由与答案信息

        Args:
            message: 用户问题

        Returns:
            {
                "route": "code|reason|chat",
                "model": "model_name",
                "confidence": 0.0-1.0,
                "source": "router_source",
                "answer": "answer_text"
            }

        Raises:
            requests.RequestException: 网络请求失败
        """
        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        payload = {"message": message}

        try:
            resp = requests.post(
                self.ask_endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Gateway request failed: {e}"
            ) from e

        return resp.json()

    def health(self) -> bool:
        """
        检查网关健康状态

        Returns:
            True if gateway is healthy
        """
        try:
            health_url = f"{self.gateway_url}/health"
            resp = requests.get(health_url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# 便捷函数供快速使用
_default_client = None


def get_gateway_client(gateway_url: Optional[str] = None) -> GatewayClient:
    """获取全局网关客户端实例（单例）"""
    global _default_client
    if _default_client is None:
        _default_client = GatewayClient(gateway_url=gateway_url)
    return _default_client


def ask(message: str, gateway_url: Optional[str] = None) -> str:
    """快速调用: 发送问题到网关并获取答案"""
    client = get_gateway_client(gateway_url=gateway_url)
    return client.ask(message)


def ask_with_routing(
    message: str, gateway_url: Optional[str] = None
) -> dict:
    """快速调用: 发送问题并获取完整路由信息"""
    client = get_gateway_client(gateway_url=gateway_url)
    return client.ask_with_routing(message)
