from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..classifier import normalize_classification_text
from ..review_ui import REVIEW_PAGE, ReviewConflictError, ReviewDatasetError
from ..storage import content_hash
from ..types import FeedbackRequest
from .runtime import RuntimeServices


class ReviewSubmission(BaseModel):
    result: Literal["correct", "incorrect"]
    reviewer: str = Field(min_length=1, max_length=100)
    revision: str = Field(min_length=16, max_length=64)

    @field_validator("reviewer")
    @classmethod
    def require_reviewer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reviewer must not be blank")
        return value.strip()


def build_control_router(services: RuntimeServices) -> APIRouter:
    router = APIRouter()
    selected = services.settings
    classifier = services.classifier
    feedback_store = services.feedback_store
    review_store = services.review_store

    @router.get("/review", include_in_schema=False)
    async def review_page():
        return HTMLResponse(
            REVIEW_PAGE,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; connect-src 'self'; "
                    "img-src 'self' data:; frame-ancestors 'none'"
                ),
                "X-Frame-Options": "DENY",
            },
        )

    @router.get("/review/api/state", include_in_schema=False)
    async def review_state(
        split: Literal["dev", "test"] = "dev",
        after: str | None = None,
    ):
        try:
            return review_store.state(split=split, after=after)
        except ReviewDatasetError:
            return JSONResponse(
                status_code=503,
                content={"detail": "Review dataset is unavailable or invalid."},
                headers={"Cache-Control": "no-store"},
            )

    @router.post("/review/api/cases/{case_id}", include_in_schema=False)
    async def review_case(case_id: str, submission: ReviewSubmission):
        try:
            split = review_store.record(
                case_id=case_id,
                result=submission.result,
                reviewer=submission.reviewer,
                revision=submission.revision,
            )
            state = review_store.state(split=split, after=case_id)
            return JSONResponse(
                content=state,
                headers={"Cache-Control": "no-store"},
            )
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Review case was not found."},
            )
        except ReviewConflictError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except ReviewDatasetError:
            return JSONResponse(
                status_code=503,
                content={"detail": "Review dataset is unavailable or invalid."},
            )

    @router.post("/route/feedback")
    async def route_feedback(req: FeedbackRequest, request: Request):
        text = req.message.strip()
        classification_text, _ = normalize_classification_text(text)
        evidence = classifier.rules.evaluate(classification_text)
        hard_label = evidence.unique_hard_label
        if hard_label is not None and hard_label != req.expected_route:
            rule_signal = "hard_rule_negative"
        elif hard_label == req.expected_route:
            rule_signal = "hard_rule_positive"
        elif req.actual_route is not None and req.actual_route != req.expected_route:
            rule_signal = "candidate_rule_gap"
        else:
            rule_signal = "human_label_only"
        feedback_store.append(
            {
                "request_id": request.state.request_id,
                "message": text if selected.storage.capture_review_text else None,
                "message_hash": content_hash(text),
                "expected_route": req.expected_route.value,
                "actual_route": (
                    req.actual_route.value if req.actual_route is not None else None
                ),
                "tags": req.tags,
                "comment": req.comment,
                "verified": True,
                "source": "manual_feedback",
                "rule_signal": rule_signal,
                "rule_hits": [
                    hit.model_dump(mode="json")
                    for hit in evidence.hard_hits + evidence.weak_hits
                ],
                "rule_version": classifier.rules.version,
                "rule_digest": classifier.rules.digest,
                "classifier_model": selected.classifier.model,
                "classifier_version": selected.classifier.version,
                "prompt_version": selected.classifier.prompt_version,
            }
        )
        return {"ok": True, "stored": True, "rule_signal": rule_signal}

    return router
