"""Endpoint mô tả trạng thái dịch vụ."""

from fastapi import APIRouter, Request

from edupulse.api.schemas import Health
from edupulse.engine.memory import current_rss_mb

router = APIRouter(prefix="/api", tags=["meta"])


# `async def` để health check không phải chờ luồng trong threadpool đang bận suy luận.
@router.get("/health")
async def health(request: Request) -> Health:
    settings = request.app.state.settings
    analyzer = getattr(request.app.state, "analyzer", None)
    manifest = None if analyzer is None else analyzer.bundle.manifest
    return Health(
        status="ok",
        bundle_version=settings.bundle_version if manifest is None else manifest.bundle_version,
        model_version=None if manifest is None else manifest.model_version,
        stage=None if manifest is None else manifest.stage,
        ner_enabled=settings.ner,
        rss_mb=current_rss_mb(),
    )
