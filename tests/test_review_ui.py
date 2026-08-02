from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from router.app import create_app


def _write_dataset(path):
    records = [
        {
            "id": "case-1",
            "dataset_id": "test-dev-v2",
            "text": "请帮我写一个 Python 函数",
            "expected_route": "code",
            "language": "zh",
            "split": "dev",
            "tags": ["seed"],
            "verified": False,
            "review_status": "pending_human_review",
            "context": [],
        },
        {
            "id": "case-2",
            "dataset_id": "test-dev-v2",
            "text": "帮我分析这两个方案的利弊",
            "expected_route": "reason",
            "language": "zh",
            "split": "dev",
            "tags": ["seed"],
            "verified": False,
            "review_status": "pending_human_review",
            "context": [{"role": "user", "content": "方案 A 和方案 B"}],
        },
        {
            "id": "case-3",
            "dataset_id": "test-dev-v2",
            "text": "你好",
            "expected_route": "chat",
            "language": "zh",
            "split": "test",
            "tags": ["seed"],
            "verified": False,
            "review_status": "pending_human_review",
            "context": [],
        },
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_dataset(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_review_page_and_binary_review_workflow(settings):
    _write_dataset(settings.storage.review_dataset_path)
    app = create_app(settings)

    with TestClient(app) as client:
        page = client.get("/review")
        state = client.get("/review/api/state", params={"split": "dev"})

        assert page.status_code == 200
        assert "Router 人工审核" in page.text
        assert page.headers["cache-control"] == "no-store"
        assert state.status_code == 200
        assert state.json()["progress"] == {
            "total": 2,
            "reviewed": 0,
            "correct": 0,
            "incorrect": 0,
            "remaining": 2,
            "percent": 0.0,
            "ready_for_evaluation": False,
        }
        assert state.json()["dataset_id"] == "test-dev-v2"
        assert state.json()["case"]["id"] == "case-1"
        assert state.json()["case"]["proposed_route"] == "code"
        assert "tags" not in state.json()["case"]
        first_revision = state.json()["case"]["revision"]

        accepted = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "correct",
                "reviewer": "Kai",
                "revision": first_revision,
            },
        )
        second_revision = accepted.json()["case"]["revision"]
        rejected = client.post(
            "/review/api/cases/case-2",
            json={
                "result": "incorrect",
                "reviewer": "Kai",
                "revision": second_revision,
            },
        )

    assert accepted.status_code == 200
    assert accepted.json()["progress"]["correct"] == 1
    assert accepted.json()["case"]["id"] == "case-2"
    assert rejected.status_code == 200
    assert rejected.json()["progress"] == {
        "total": 2,
        "reviewed": 2,
        "correct": 1,
        "incorrect": 1,
        "remaining": 0,
        "percent": 100.0,
        "ready_for_evaluation": False,
    }
    assert rejected.json()["case"] is None

    records = _read_dataset(settings.storage.review_dataset_path)
    assert records[0]["expected_route"] == "code"
    assert records[0]["review_result"] == "correct"
    assert records[0]["review_status"] == "verified"
    assert records[0]["verified"] is True
    assert records[1]["expected_route"] == "reason"
    assert records[1]["review_result"] == "incorrect"
    assert records[1]["review_status"] == "rejected"
    assert records[1]["verified"] is False
    assert os.stat(settings.storage.review_dataset_path).st_mode & 0o777 == 0o600


def test_review_rejects_invalid_or_conflicting_votes(settings):
    _write_dataset(settings.storage.review_dataset_path)
    app = create_app(settings)

    with TestClient(app) as client:
        revision = client.get("/review/api/state").json()["case"]["revision"]
        invalid = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "code",
                "reviewer": "Kai",
                "revision": revision,
            },
        )
        blank = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "correct",
                "reviewer": " ",
                "revision": revision,
            },
        )
        first = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "correct",
                "reviewer": "Kai",
                "revision": revision,
            },
        )
        idempotent = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "correct",
                "reviewer": "Kai",
                "revision": revision,
            },
        )
        conflict = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "incorrect",
                "reviewer": "Other",
                "revision": revision,
            },
        )
        missing = client.post(
            "/review/api/cases/unknown",
            json={
                "result": "correct",
                "reviewer": "Kai",
                "revision": revision,
            },
        )

    assert invalid.status_code == 422
    assert blank.status_code == 422
    assert first.status_code == 200
    assert idempotent.status_code == 200
    assert conflict.status_code == 409
    assert missing.status_code == 404


def test_review_api_is_not_published_as_router_contract(settings):
    _write_dataset(settings.storage.review_dataset_path)
    app = create_app(settings)

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert "/review" not in schema["paths"]
    assert "/review/api/state" not in schema["paths"]


def test_stale_review_page_cannot_overwrite_changed_case(settings):
    _write_dataset(settings.storage.review_dataset_path)
    app = create_app(settings)

    with TestClient(app) as client:
        state = client.get("/review/api/state").json()
        stale_revision = state["case"]["revision"]
        records = _read_dataset(settings.storage.review_dataset_path)
        records[0]["text"] = "内容已经改变"
        settings.storage.review_dataset_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        response = client.post(
            "/review/api/cases/case-1",
            json={
                "result": "correct",
                "reviewer": "Kai",
                "revision": stale_revision,
            },
        )

    assert response.status_code == 409
