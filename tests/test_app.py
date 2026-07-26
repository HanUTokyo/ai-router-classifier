from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from router.app import create_app
from router.ollama import OllamaUnavailable


def mock_transport(
    *,
    fail_answer: bool = False,
    timeout_answer: bool = False,
    incomplete_stream: bool = False,
    missing_classifier: bool = False,
):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            models = [
                "gemma4:e4b",
                "deepseek-r1:8b",
                "deepseek-coder:6.7b",
            ]
            if not missing_classifier:
                models.append("qwen2.5:0.5b")
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": model, "digest": f"digest-{index}"}
                        for index, model in enumerate(models)
                    ]
                },
            )
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["model"] == "qwen2.5:0.5b":
            assert payload["think"] is False
            return httpx.Response(
                200,
                json={"message": {"content": '{"route":"chat"}'}},
            )
        if timeout_answer:
            raise httpx.ReadTimeout("slow", request=request)
        if fail_answer:
            return httpx.Response(500, json={"error": "broken"})
        if payload.get("stream"):
            assert payload["think"] is False
            body = '{"message":{"role":"assistant","content":"Hello"},"done":false}\n'
            if not incomplete_stream:
                body += '{"done":true,"eval_count":1,"prompt_eval_count":2}\n'
            return httpx.Response(200, text=body)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Hello"},
                "done": True,
            },
        )

    return httpx.MockTransport(handler), calls


def test_route_and_openai_non_stream(settings):
    transport, calls = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        route = client.post("/route", json={"message": "你好"})
        completion = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-router",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )

    assert route.status_code == 200
    assert route.json()["route"] == "chat"
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["message"]["content"] == "Hello"
    assert calls[-1]["model"] == "gemma4:e4b"


def test_blank_route_message_is_rejected(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post("/route", json={"message": "   "})

    assert response.status_code == 422


def test_external_request_id_is_not_reused(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.get(
            "/health/live",
            headers={"X-Request-ID": "user@example.com"},
        )

    assert response.headers["x-request-id"] != "user@example.com"
    uuid.UUID(response.headers["x-request-id"])


def test_startup_refuses_missing_required_model(settings):
    strict = settings.model_copy(
        update={
            "classifier": settings.classifier.model_copy(
                update={"strict_model_check": True}
            )
        }
    )
    transport, _ = mock_transport(missing_classifier=True)
    app = create_app(strict, transport=transport)

    with pytest.raises(OllamaUnavailable, match="not installed"):
        with TestClient(app):
            pass


def test_api_key_protects_non_health_endpoints(settings):
    keyed = settings.model_copy(
        update={
            "server": settings.server.model_copy(
                update={"api_key": "test-secret"}
            )
        }
    )
    transport, _ = mock_transport()
    app = create_app(keyed, transport=transport)

    with TestClient(app) as client:
        denied = client.post("/route", json={"message": "hello"})
        allowed = client.post(
            "/route",
            json={"message": "hello"},
            headers={"Authorization": "Bearer test-secret"},
        )
        live = client.get("/health/live")

    assert denied.status_code == 401
    uuid.UUID(denied.headers["x-request-id"])
    assert allowed.status_code == 200
    assert live.status_code == 200


def test_metrics_exposes_prometheus_content_type(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "router_http_requests_total" in response.text


def test_openai_stream_terminates(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["x-ai-route"] == "chat"
    assert '"content": "Hello"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_ollama_stream_is_real_ndjson(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert '"done":true' in response.text


def test_incomplete_streams_end_with_explicit_error(settings):
    transport, _ = mock_transport(incomplete_stream=True)
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        openai = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )
        ollama = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )

    assert "event: error" in openai.text
    assert openai.text.endswith("data: [DONE]\n\n")
    assert '"error": "Upstream stream failed"' in ollama.text
    assert '"done": true' in ollama.text


def test_upstream_http_failure_is_502_not_fake_answer(settings):
    transport, _ = mock_transport(fail_answer=True)
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_failed"


def test_upstream_timeout_is_504(settings):
    transport, _ = mock_transport(timeout_answer=True)
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


def test_feedback_is_secure_and_responses_api_is_explicit(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        feedback = client.post(
            "/route/feedback",
            json={"message": "hello", "expected_route": "chat"},
        )
        unsupported = client.post("/v1/responses", json={"input": "hello"})
        schema = client.get("/openapi.json")

    assert feedback.status_code == 200
    assert unsupported.status_code == 501
    assert "/v1/responses" not in schema.json()["paths"]
    assert os.stat(settings.storage.feedback_path).st_mode & 0o777 == 0o600


def test_ask_alias_has_deprecation_headers(settings):
    transport, _ = mock_transport()
    app = create_app(settings, transport=transport)

    with TestClient(app) as client:
        response = client.post("/ask", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["deprecation"] == "true"
    assert response.headers["sunset"] == "Thu, 01 Oct 2026 00:00:00 GMT"
    assert response.headers["link"] == (
        '</v1/chat/completions>; rel="successor-version"'
    )

    failing_transport, _ = mock_transport(fail_answer=True)
    failing_app = create_app(settings, transport=failing_transport)
    with TestClient(failing_app) as client:
        failed = client.post("/ask", json={"message": "hello"})

    assert failed.status_code == 502
    assert failed.headers["deprecation"] == "true"


def test_tools_use_tool_model_and_unsupported_contract_is_rejected(settings):
    transport, calls = mock_transport()
    app = create_app(settings, transport=transport)
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with TestClient(app) as client:
        supported = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "tools": [tool],
                "tool_choice": "auto",
            },
        )
        unsupported = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "response_format": {"type": "json_object"},
            },
        )

    assert supported.status_code == 200
    assert calls[-1]["model"] == settings.gateway.tool_model
    assert calls[-1]["tools"] == [tool]
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_response_format"
