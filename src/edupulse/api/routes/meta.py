"""Endpoint mô tả dịch vụ: sức khoẻ, cấu hình cho giao diện, model card, file mẫu."""

from fastapi import APIRouter, Request, Response

from edupulse.api.errors import not_implemented
from edupulse.api.schemas import AppConfig, ErrorResponse, Health
from edupulse.bundle_schema import ModelCard
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


@router.get("/config", summary="Giới hạn, ngân sách duyệt, câu mẫu và trạng thái demo")
def app_config() -> AppConfig:
    raise not_implemented()


@router.get(
    "/model-card",
    responses={404: {"model": ErrorResponse, "description": "model_card_unavailable"}},
    summary="Model card của bundle đang phục vụ",
)
def model_card() -> ModelCard:
    raise not_implemented()


@router.get(
    "/sample.csv",
    response_class=Response,
    responses={
        200: {
            "description": "File mẫu 200 câu, dựng lúc build image.",
            "content": {"text/csv": {"schema": {"type": "string"}}},
        },
        404: {"model": ErrorResponse, "description": "demo_unavailable"},
    },
    summary="Tải file mẫu",
)
def sample_csv() -> Response:
    raise not_implemented()
