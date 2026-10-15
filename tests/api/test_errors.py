"""Mọi lỗi API có dạng `{"detail": {"code", "message"}}` và không lặp lại dữ liệu người dùng."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from edupulse.api.errors import ApiError, install_error_handlers
from edupulse.api.schemas import ErrorResponse


class Payload(BaseModel):
    text: str = Field(min_length=1, max_length=5)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/echo")
    def echo(payload: Payload) -> Payload:
        return payload

    @app.get("/busy")
    def busy() -> None:
        raise ApiError(429, "busy", "Hàng đợi đầy.", headers={"Retry-After": "5"})

    @app.get("/gone")
    def gone() -> None:
        raise ApiError(404, "batch_expired", "Phiên đã hết hạn.")

    @app.get("/teapot")
    def teapot() -> None:
        raise ApiError(418, "teapot", "Tôi là ấm trà.")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("hỏng")

    return TestClient(app, raise_server_exceptions=False)


def test_api_error_keeps_its_own_code_message_and_headers(client: TestClient):
    response = client.get("/busy")
    assert response.status_code == 429
    assert response.json() == {"detail": {"code": "busy", "message": "Hàng đợi đầy."}}
    assert response.headers["retry-after"] == "5"


def test_api_error_code_wins_over_the_generic_code_for_the_status(client: TestClient):
    assert client.get("/gone").json()["detail"]["code"] == "batch_expired"
    assert client.get("/teapot").json()["detail"]["code"] == "teapot"


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("GET", "/khong-co", 404, "not_found"),
        ("GET", "/echo", 405, "method_not_allowed"),
    ],
)
def test_framework_errors_get_the_standard_shape(client, method, path, status, code):
    response = client.request(method, path)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    ErrorResponse.model_validate(response.json())


def test_validation_error_reports_where_but_never_echoes_the_input(client: TestClient):
    secret = "văn bản góp ý rất dài của sinh viên"
    response = client.post("/echo", json={"text": secret})
    assert response.status_code == 422
    body = response.json()["detail"]
    assert body["code"] == "validation_error"
    assert body["fields"] == [{"loc": ["body", "text"], "type": "string_too_long"}]
    assert secret not in response.text
    ErrorResponse.model_validate(response.json())


def test_validation_error_for_missing_body_field(client: TestClient):
    response = client.post("/echo", json={})
    assert response.status_code == 422
    assert response.json()["detail"]["fields"] == [{"loc": ["body", "text"], "type": "missing"}]


def test_unhandled_exception_becomes_a_generic_500_without_details(client: TestClient):
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": {"code": "internal_error", "message": "Lỗi máy chủ."}}
    assert "hỏng" not in response.text
