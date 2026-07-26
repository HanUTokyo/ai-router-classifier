from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .classifier import RouterClassifier
from .ollama import OllamaClient
from .settings import Settings
from .storage import ExplicitMemory
from .types import Message, RouteDecision


@dataclass
class GatewayResult:
    decision: RouteDecision
    selected_model: str
    upstream: dict[str, Any]


@dataclass
class GatewayStream:
    decision: RouteDecision
    selected_model: str
    upstream: httpx.Response


class ThinGateway:
    def __init__(
        self,
        settings: Settings,
        classifier: RouterClassifier,
        ollama: OllamaClient,
    ):
        self.settings = settings
        self.classifier = classifier
        self.ollama = ollama
        self.memory = ExplicitMemory(
            settings.storage.memory_path,
            settings.storage.memory_enabled,
        )

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GatewayResult:
        decision, model, prepared = await self._prepare(messages, tools)
        upstream = await self.ollama.chat(
            model=model,
            messages=prepared,
            options=_options(temperature, top_p, max_tokens),
            tools=tools,
            think=self.settings.gateway.think,
        )
        return GatewayResult(
            decision=decision,
            selected_model=model,
            upstream=upstream,
        )

    async def stream(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GatewayStream:
        decision, model, prepared = await self._prepare(messages, tools)
        upstream = await self.ollama.open_chat_stream(
            model=model,
            messages=prepared,
            options=_options(temperature, top_p, max_tokens),
            tools=tools,
            think=self.settings.gateway.think,
        )
        return GatewayStream(
            decision=decision,
            selected_model=model,
            upstream=upstream,
        )

    async def _prepare(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[RouteDecision, str, list[dict[str, Any]]]:
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].role == "user"
            ),
            None,
        )
        if last_user_index is None:
            raise ValueError("At least one user message is required")
        last_user = _content_text(messages[last_user_index].content)
        context = messages[:last_user_index][-6:]
        decision = await self.classifier.classify(last_user, context)

        route_model = self.settings.route_model(decision.route.value)
        if tools and route_model not in self.settings.gateway.tool_supported_models:
            model = self.settings.gateway.tool_model
        else:
            model = route_model

        self.memory.capture_if_explicit(last_user)
        prepared = [_message_dict(message) for message in messages]
        memory_prompt = self.memory.prompt()
        if memory_prompt:
            prepared = [{"role": "system", "content": memory_prompt}, *prepared]
        return decision, model, prepared


def _message_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": _content_text(message.content),
    }
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {
                "text",
                "input_text",
            }:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _options(
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    return options
