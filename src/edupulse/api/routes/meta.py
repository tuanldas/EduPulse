"""Endpoint mô tả dịch vụ: sức khoẻ, cấu hình cho giao diện, model card, file mẫu."""

from fastapi import APIRouter, Request, Response

from edupulse.api.deps import AnalyzerDep, DemoDep, ModelCardDep, SettingsDep
from edupulse.api.errors import ApiError
from edupulse.api.limits import MAX_GROUPS
from edupulse.api.schemas import (
    AppConfig,
    BudgetOption,
    ErrorResponse,
    Health,
    Limits,
    ModelInfo,
    UiSample,
)
from edupulse.bundle_schema import ModelCard
from edupulse.config import (
    AUTO_XAI_TOP,
    BUDGETS,
    DEFAULT_BUDGET,
    HEARTBEAT_S,
    MAX_CHARS,
    MAX_ROWS,
    MAX_UPLOAD_MB,
)
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


SAMPLE_FILENAME = "edupulse-file-mau.csv"


@router.get("/config", summary="Giới hạn, ngân sách duyệt, câu mẫu và trạng thái demo")
async def app_config(analyzer: AnalyzerDep, settings: SettingsDep, demo: DemoDep) -> AppConfig:
    manifest = analyzer.bundle.manifest
    efficiency = analyzer.calibration.efficiency
    return AppConfig(
        model=ModelInfo(
            model_version=manifest.model_version,
            bundle_version=manifest.bundle_version,
            stage=manifest.stage,
        ),
        ner_enabled=settings.ner,
        limits=Limits(
            max_chars=MAX_CHARS,
            max_upload_mb=MAX_UPLOAD_MB,
            max_rows=MAX_ROWS,
            max_groups=MAX_GROUPS,
            xai_max_words=analyzer.xai_max_words,
            auto_xai_top=AUTO_XAI_TOP,
        ),
        budgets=[
            BudgetOption(budget=b, default=b == DEFAULT_BUDGET, efficiency=efficiency[b])
            for b in BUDGETS
        ],
        default_budget=DEFAULT_BUDGET,
        heartbeat_s=HEARTBEAT_S,
        ui_samples=[UiSample(text=s.text, no_diacritics=s.no_diacritics) for s in demo.ui_samples],
        demo_available=demo.available,
        demo_reason=demo.reason,
        source_url=settings.source_url,
    )


@router.get(
    "/model-card",
    responses={404: {"model": ErrorResponse, "description": "model_card_unavailable"}},
    summary="Model card của bundle đang phục vụ",
)
async def model_card(card: ModelCardDep) -> ModelCard:
    return card


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
async def sample_csv(demo: DemoDep) -> Response:
    if not demo.available:
        raise ApiError(404, "demo_unavailable", demo.reason or "Chưa có file mẫu.")
    return Response(
        demo.sample_csv,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{SAMPLE_FILENAME}"'},
    )
