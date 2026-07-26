from __future__ import annotations

import json

import httpx
import pytest

from router.ollama import ClassifierOutputError, OllamaClient
from router.settings import Settings
from router.types import RouteLabel


@pytest.mark.asyncio
async def test_invalid_classifier_output_is_retried_once():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        content = "not json" if len(calls) == 1 else '{"route":"reason"}'
        return httpx.Response(200, json={"message": {"content": content}})

    client = OllamaClient(
        Settings.load().ollama,
        transport=httpx.MockTransport(handler),
    )
    try:
        route = await client.classify(
            model="test",
            system_prompt="classify",
            user_prompt="why",
            max_attempts=2,
        )
    finally:
        await client.close()

    assert route == RouteLabel.REASON
    assert len(calls) == 2
    assert calls[0]["options"]["temperature"] == 0
    assert calls[0]["think"] is False
    assert calls[0]["format"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_repeated_invalid_output_raises_typed_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": '{"route":"unsupported"}'}},
        )

    client = OllamaClient(
        Settings.load().ollama,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ClassifierOutputError):
            await client.classify(
                model="test",
                system_prompt="classify",
                user_prompt="request",
                max_attempts=2,
            )
    finally:
        await client.close()
