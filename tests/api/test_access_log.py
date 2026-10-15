"""Nhật ký truy cập chỉ có method, khuôn route, status và thời gian; không query, id hay thân."""

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from edupulse.api.access_log import AccessLogMiddleware
from edupulse.api.errors import install_error_handlers

SECRET_QUERY = "token=BI-MAT-QUERY&group=Nhom+Rieng"
SECRET_ID = "id-phien-bi-mat-123"
SECRET_BODY = "Thay Nam rat te, so dien thoai 0912345678"


@pytest.fixture
def access_lines(caplog: pytest.LogCaptureFixture) -> Callable[[], list[str]]:
    """Trả hàm đọc các dòng nhật ký truy cập đã ghi tới thời điểm gọi."""
    caplog.set_level(logging.INFO, logger="edupulse")

    def lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "edupulse.access"]

    return lines


def test_logs_method_route_template_status_and_duration(
    client: TestClient, access_lines: Callable[[], list[str]]
):
    client.get("/api/health")
    (line,) = access_lines()
    assert re.fullmatch(r"GET /api/health 200 \d+ms", line)


def test_query_string_ids_and_body_never_reach_the_log(
    client: TestClient, access_lines: Callable[[], list[str]], caplog: pytest.LogCaptureFixture
):
    client.get(f"/api/health?{SECRET_QUERY}")
    client.get(f"/api/batches/{SECRET_ID}/results?{SECRET_QUERY}")
    client.post(f"/api/batches/{SECRET_ID}/run", json={"text": SECRET_BODY})
    client.post("/api/health", content=SECRET_BODY)

    everything = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("BI-MAT", "Nhom", SECRET_ID, "Thay Nam", "0912345678"):
        assert secret not in everything
    assert len(access_lines()) == 4


def test_unknown_api_paths_are_logged_as_unmatched(
    client: TestClient, access_lines: Callable[[], list[str]]
):
    client.get(f"/api/{SECRET_ID}")
    assert access_lines()[-1].startswith("GET unmatched 404 ")


def test_wrong_method_keeps_the_route_template(
    client: TestClient, access_lines: Callable[[], list[str]]
):
    client.post("/api/health")
    assert re.fullmatch(r"POST /api/health 405 \d+ms", access_lines()[-1])


@pytest.mark.model
def test_analyze_and_explain_requests_never_log_the_sentence(
    live_client: TestClient,
    access_lines: Callable[[], list[str]],
    caplog: pytest.LogCaptureFixture,
):
    live_client.post(f"/api/analyze?{SECRET_QUERY}", json={"text": SECRET_BODY})
    live_client.post("/api/explain", json={"text": SECRET_BODY})
    live_client.post("/api/analyze", json={"text": "x" * 600})  # 422

    lines = access_lines()
    assert re.fullmatch(r"POST /api/analyze 200 \d+ms", lines[0])
    assert re.fullmatch(r"POST /api/explain 200 \d+ms", lines[1])
    assert re.fullmatch(r"POST /api/analyze 422 \d+ms", lines[2])
    everything = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("BI-MAT", "Nhom", "Thay Nam", "0912345678", "x" * 20):
        assert secret not in everything


def test_static_requests_are_grouped_instead_of_logging_the_path(
    client: TestClient, web_dir: Path, access_lines: Callable[[], list[str]]
):
    client.get("/")
    client.get(f"/{SECRET_ID}/")
    found, missing = access_lines()
    assert found.startswith("GET static 200 ")
    assert missing.startswith("GET static 404 ")
    assert SECRET_ID not in found + missing


def test_router_prefix_from_include_router_is_kept_in_the_template(
    access_lines: Callable[[], list[str]],
):
    router = APIRouter()

    @router.get("/items/{item_id}")
    def read_item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.add_middleware(AccessLogMiddleware)

    assert TestClient(app).get(f"/api/items/{SECRET_ID}").status_code == 200
    (line,) = access_lines()
    assert re.fullmatch(r"GET /api/items/\{item_id\} 200 \d+ms", line)


def test_unhandled_error_is_logged_as_500_without_the_exception_message(
    access_lines: Callable[[], list[str]], caplog: pytest.LogCaptureFixture
):
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/api/boom")
    def boom() -> None:
        raise RuntimeError(f"hỏng khi xử lý {SECRET_BODY}")

    app.add_middleware(AccessLogMiddleware)

    response = TestClient(app, raise_server_exceptions=False).post("/api/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": {"code": "internal_error", "message": "Lỗi máy chủ."}}

    assert re.fullmatch(r"POST /api/boom 500 \d+ms", access_lines()[-1])
    error_lines = [r.getMessage() for r in caplog.records if r.name == "edupulse.api"]
    assert error_lines == ["Lỗi không xử lý RuntimeError tại /api/boom"]
    assert "Thay Nam" not in "\n".join(record.getMessage() for record in caplog.records)
