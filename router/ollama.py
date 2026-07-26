from __future__ import annotations

import json
from typing import Any

import httpx

from .settings import OllamaSettings
from .types import RouteLabel, SmallModelOutput


class OllamaError(RuntimeError):
    selected_model: str | None = None


class OllamaTimeout(OllamaError):
    pass


class OllamaUnavailable(OllamaError):
    pass


class ClassifierOutputError(OllamaError):
    pass


class OllamaClient:
    def __init__(
        self,
        settings: OllamaSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.read_timeout_seconds,
            pool=settings.connect_timeout_seconds,
        )
        self.client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def installed_models(self) -> dict[str, str]:
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Timed out while checking Ollama models") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailable(f"Unable to query Ollama models: {exc}") from exc
        models: dict[str, str] = {}
        for item in payload.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models[str(name)] = str(item.get("digest") or "")
        return models

    async def ensure_models(self, required: set[str]) -> dict[str, str]:
        installed = await self.installed_models()
        missing = sorted(model for model in required if model not in installed)
        if missing:
            raise OllamaUnavailable(
                "Configured Ollama models are not installed: " + ", ".join(missing)
            )
        return installed

    async def classify(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        few_shots: list[tuple[str, RouteLabel]] | None = None,
        max_attempts: int,
    ) -> RouteLabel:
        messages = [{"role": "system", "content": system_prompt}]
        for example_prompt, example_route in few_shots or []:
            messages.extend(
                [
                    {"role": "user", "content": example_prompt},
                    {
                        "role": "assistant",
                        "content": json.dumps({"route": example_route.value}),
                    },
                ]
            )
        messages.append({"role": "user", "content": user_prompt})
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid. Return only the required "
                            'JSON object, for example {"route":"chat"}.'
                        ),
                    }
                )
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "string",
                            "enum": ["code", "reason", "chat"],
                        }
                    },
                    "required": ["route"],
                    "additionalProperties": False,
                },
                "options": {
                    "temperature": 0,
                    "num_predict": 48,
                },
            }
            try:
                data = await self._post_json("/api/chat", payload)
                content = data.get("message", {}).get("content", "")
                parsed = json.loads(content)
                return SmallModelOutput.model_validate(parsed).route
            except (
                json.JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                last_error = exc
                continue
        raise ClassifierOutputError(
            f"Classifier returned invalid structured output after {max_attempts} attempts"
        ) from last_error

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        data = await self._post_json("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaUnavailable("Ollama response is missing message")
        return data

    async def open_chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
    ) -> httpx.Response:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        stream_timeout = httpx.Timeout(
            connect=self.settings.connect_timeout_seconds,
            read=self.settings.stream_timeout_seconds,
            write=self.settings.read_timeout_seconds,
            pool=self.settings.connect_timeout_seconds,
        )
        request = self.client.build_request(
            "POST",
            "/api/chat",
            json=payload,
            timeout=stream_timeout,
        )
        try:
            response = await self.client.send(request, stream=True)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Ollama stream timed out before starting") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama stream failed: {exc}") from exc

    async def _post_json(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = await self.client.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaUnavailable("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise OllamaUnavailable("Ollama returned a non-object response")
        if data.get("error"):
            raise OllamaUnavailable(str(data["error"]))
        return data
