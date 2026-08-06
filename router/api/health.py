from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ..observability import metrics_payload
from ..ollama import OllamaError, OllamaUnavailable
from .runtime import RuntimeServices


def build_health_router(services: RuntimeServices) -> APIRouter:
    router = APIRouter()
    selected = services.settings
    ollama = services.ollama
    classifier = services.classifier

    @router.get("/health/live")
    async def health_live():
        return {"status": "ok", "service": "ai-router"}

    @router.get("/health/ready")
    async def health_ready():
        try:
            installed = await ollama.ensure_models(selected.required_models)
            identity_valid = classifier.bind_model_digest(
                installed.get(selected.classifier.model)
            )
            if selected.classifier.strict_model_check and not identity_valid:
                raise OllamaUnavailable(
                    "Installed classifier model digest does not match the "
                    "accepted calibration receipt"
                )
        except OllamaError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        return {
            "status": "ready",
            "classifier_model": selected.classifier.model,
            "calibration": {
                "version": classifier.calibrator.version,
                "validated": classifier.calibrator.validated,
                "identity_valid": classifier.calibrator.identity_valid,
                "prompt_digest": classifier.prompt_digest,
                "expected_model_digest": (
                    classifier.calibrator.expected_model_digest
                ),
                "installed_model_digest": (
                    classifier.calibrator.installed_model_digest
                ),
                "hard_rules_enabled": classifier.calibrator.hard_rules_enabled,
                "weak_fallback_enabled": (
                    classifier.calibrator.weak_fallback_enabled
                ),
            },
            "shadow_rules": {
                "version": (
                    classifier.shadow_rules.version
                    if classifier.shadow_rules is not None
                    else None
                ),
                "loaded": classifier.shadow_rules is not None,
                "error": classifier.shadow_rules_error,
            },
            "rules": {
                "version": classifier.rules.version,
                "digest": classifier.rules.digest,
            },
            "models": {
                model: installed.get(model, "")
                for model in sorted(selected.required_models)
            },
        }

    @router.get("/health")
    async def health():
        return await health_ready()

    @router.get("/metrics")
    async def metrics():
        body, content_type = metrics_payload()
        return Response(content=body, headers={"Content-Type": content_type})

    return router
