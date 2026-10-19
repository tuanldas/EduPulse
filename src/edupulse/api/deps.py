"""Phụ thuộc dùng chung của các route: bộ phân tích đã nạp lúc khởi động."""

from typing import Annotated

from fastapi import Depends, Request

from edupulse.api.errors import ApiError
from edupulse.api.jobs import JobStore
from edupulse.bundle_schema import InsightBackground, ModelCard
from edupulse.config import Settings
from edupulse.demo_data import DemoData
from edupulse.engine.service import Analyzer


def get_analyzer(request: Request) -> Analyzer:
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        # Chỉ xảy ra khi app chạy mà chưa qua lifespan (ví dụ TestClient không vào `with`).
        raise ApiError(503, "engine_unavailable", "Mô hình chưa sẵn sàng.")
    return analyzer


AnalyzerDep = Annotated[Analyzer, Depends(get_analyzer)]


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_demo(request: Request) -> DemoData:
    demo = getattr(request.app.state, "demo", None)
    if demo is None:
        raise ApiError(503, "engine_unavailable", "Dữ liệu demo chưa được nạp.")
    return demo


def get_model_card(request: Request) -> ModelCard:
    card = getattr(request.app.state, "model_card", None)
    if card is None:
        raise ApiError(404, "model_card_unavailable", "Bundle không có model card hợp lệ.")
    return card


def get_background(request: Request) -> InsightBackground:
    background = getattr(request.app.state, "background", None)
    if background is None:
        raise ApiError(503, "insights_unavailable", "Bundle không có số liệu nền để tính insight.")
    return background


SettingsDep = Annotated[Settings, Depends(get_settings)]
DemoDep = Annotated[DemoData, Depends(get_demo)]
ModelCardDep = Annotated[ModelCard, Depends(get_model_card)]
BackgroundDep = Annotated[InsightBackground, Depends(get_background)]


def get_store(request: Request) -> JobStore:
    store = getattr(request.app.state, "jobs", None)
    if store is None:
        raise ApiError(503, "engine_unavailable", "Kho batch chưa sẵn sàng.")
    return store


StoreDep = Annotated[JobStore, Depends(get_store)]
