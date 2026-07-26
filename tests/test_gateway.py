from __future__ import annotations

import pytest

from router.gateway import ThinGateway
from router.ollama import OllamaUnavailable
from router.types import Message, RouteDecision, RouteLabel, RouteSource


class CodeClassifier:
    async def classify(self, *_args, **_kwargs):
        return RouteDecision(
            route=RouteLabel.CODE,
            confidence=None,
            source=RouteSource.SMALL_MODEL,
            classifier_model="classifier",
            classifier_version="test",
            degraded=False,
            latency_ms=1.0,
        )


class FailingOllama:
    async def chat(self, **_kwargs):
        raise OllamaUnavailable("failed")


@pytest.mark.asyncio
async def test_gateway_error_retains_selected_route_model(settings):
    gateway = ThinGateway(settings, CodeClassifier(), FailingOllama())

    with pytest.raises(OllamaUnavailable) as captured:
        await gateway.complete(
            messages=[Message(role="user", content="write code")]
        )

    assert captured.value.selected_model == settings.gateway.code_model
