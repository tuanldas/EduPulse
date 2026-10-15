"""Kiểm tra kiểu dữ liệu của hợp đồng API mà không cần mô hình."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from edupulse.api import schemas
from edupulse.api.errors import install_error_handlers
from edupulse.config import MAX_CHARS, MAX_ROWS


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/analyze")
    def analyze(body: schemas.AnalyzeRequest) -> dict[str, object]:
        return {"text": body.text, "compare": body.compare}

    @app.post("/run")
    def run(body: schemas.RunRequest) -> dict[str, object]:
        return body.model_dump()

    @app.post("/export")
    def export(body: schemas.ExportRequest) -> dict[str, object]:
        return body.model_dump()

    @app.post("/insights")
    def insights(body: schemas.InsightsRequest) -> dict[str, object]:
        return body.model_dump()

    return TestClient(app)


def test_text_is_stripped_before_it_is_measured(client: TestClient):
    response = client.post("/analyze", json={"text": "  xin chào \n"})
    assert response.json() == {"text": "xin chào", "compare": False}


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_has_its_own_error_code(client: TestClient, text: str):
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "text_empty"
    assert "params" not in detail


def test_too_long_text_has_its_own_error_code_and_the_limit(client: TestClient):
    response = client.post("/analyze", json={"text": "a" * (MAX_CHARS + 1)})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "text_too_long"
    assert detail["params"] == {"max_chars": MAX_CHARS}
    assert "a" * 20 not in response.text  # lỗi không lặp lại câu đã gửi
    assert client.post("/analyze", json={"text": "a" * MAX_CHARS}).status_code == 200


def test_length_is_measured_after_stripping(client: TestClient):
    padded = " " * 50 + "a" * MAX_CHARS + " " * 50
    assert client.post("/analyze", json={"text": padded}).status_code == 200


def test_a_wrong_type_is_a_generic_validation_error(client: TestClient):
    detail = client.post("/analyze", json={"text": 5}).json()["detail"]
    assert detail["code"] == "validation_error"


def test_request_models_reject_unknown_fields(client: TestClient):
    response = client.post("/run", json={"text_col": "Góp ý", "review_buget": 10})
    assert response.status_code == 422
    assert response.json()["detail"]["fields"] == [
        {"loc": ["body", "review_buget"], "type": "extra_forbidden"}
    ]


def test_run_request_defaults_and_budget_choices(client: TestClient):
    assert client.post("/run", json={"text_col": "Góp ý"}).json() == {
        "text_col": "Góp ý",
        "group_col": None,
        "review_budget": 10,
    }
    for budget in (5, 10, 20, 30):
        assert (
            client.post("/run", json={"text_col": "x", "review_budget": budget}).status_code == 200
        )
    for budget in (0, 7, 50):
        assert (
            client.post("/run", json={"text_col": "x", "review_budget": budget}).status_code == 422
        )


def test_run_request_needs_a_non_empty_text_column(client: TestClient):
    assert client.post("/run", json={"text_col": ""}).status_code == 422
    assert client.post("/run", json={}).status_code == 422


def test_export_request_limits(client: TestClient):
    assert client.post("/export", json={}).json() == {"kind": "all", "overrides": []}
    ok = {"kind": "corrected", "overrides": [{"index": 3, "sentiment": "negative"}]}
    assert client.post("/export", json=ok).status_code == 200
    bad_label = {"overrides": [{"index": 3, "sentiment": "angry"}]}
    assert client.post("/export", json=bad_label).status_code == 422
    negative_index = {"overrides": [{"index": -1}]}
    assert client.post("/export", json=negative_index).status_code == 422
    too_many = {"overrides": [{"index": i} for i in range(MAX_ROWS + 1)]}
    assert client.post("/export", json=too_many).status_code == 422
    assert client.post("/export", json={"kind": "everything"}).status_code == 422


def test_insights_request_groups_are_optional_and_bounded(client: TestClient):
    assert client.post("/insights", json={}).json() == {"groups": None}
    assert client.post("/insights", json={"groups": ["Nhóm A"]}).status_code == 200
    assert client.post("/insights", json={"groups": [""]}).status_code == 422
    assert client.post("/insights", json={"groups": ["x" * 101]}).status_code == 422
    assert client.post("/insights", json={"groups": [str(i) for i in range(31)]}).status_code == 422


def test_response_models_cannot_be_built_with_missing_fields():
    with pytest.raises(ValidationError):
        schemas.Flag(needs_review=True)  # type: ignore[call-arg]
