"""Phụ thuộc dùng chung của các route: bộ phân tích đã nạp lúc khởi động."""

from typing import Annotated

from fastapi import Depends, Request

from edupulse.api.errors import ApiError
from edupulse.engine.service import Analyzer


def get_analyzer(request: Request) -> Analyzer:
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        # Chỉ xảy ra khi app chạy mà chưa qua lifespan (ví dụ TestClient không vào `with`).
        raise ApiError(503, "engine_unavailable", "Mô hình chưa sẵn sàng.")
    return analyzer


AnalyzerDep = Annotated[Analyzer, Depends(get_analyzer)]
