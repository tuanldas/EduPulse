"""OpenAPI đã commit (`apps/web/openapi.json`) phải khớp với mã: giao diện sinh kiểu từ file đó."""

import json
from typing import Any, get_args

from edupulse.api import schemas
from edupulse.api.export_openapi import DEFAULT_OUT, build_openapi, render
from edupulse.config import BUDGETS
from edupulse.insights import models as insight_models
from edupulse.labels import LABELS


def test_committed_openapi_matches_the_code():
    assert DEFAULT_OUT.is_file(), "thiếu apps/web/openapi.json: chạy `make gen-api`"
    committed = DEFAULT_OUT.read_text(encoding="utf-8")
    assert committed == render(build_openapi()), (
        "apps/web/openapi.json không khớp mã nguồn: chạy `make gen-api` rồi commit cả "
        "openapi.json và src/lib/api/schema.d.ts"
    )


def test_review_budget_literal_matches_the_configured_budgets():
    assert get_args(schemas.ReviewBudget) == BUDGETS


def test_label_literals_match_the_label_order():
    assert get_args(schemas.SentimentLabel) == LABELS["sentiment"]
    assert get_args(schemas.TopicLabel) == LABELS["topic"]
    assert schemas.SentimentLabel is insight_models.SentimentLabel
    assert schemas.TopicLabel is insight_models.TopicLabel


def _operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (path, method, operation)
        for path, item in spec["paths"].items()
        for method, operation in item.items()
    ]


def test_operation_ids_are_unique_and_named_after_the_handlers():
    operations = _operations(build_openapi())
    ids = [operation["operationId"] for _, _, operation in operations]
    assert len(ids) == len(set(ids))
    # Không có tên tự sinh kiểu `analyze_api_analyze_post`: tên hàm làm operationId.
    assert all("_api_" not in operation_id for operation_id in ids)


def test_every_route_documents_the_real_validation_error_body():
    spec = build_openapi()
    assert "HTTPValidationError" not in spec["components"]["schemas"]
    for path, method, operation in _operations(spec):
        for status, response in operation["responses"].items():
            if status != "422":
                continue
            ref = response["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorResponse"), (path, method)


def _subclasses(base: type) -> set[type]:
    found: set[type] = set()
    for sub in base.__subclasses__():
        found |= {sub} | _subclasses(sub)
    return found


def test_response_models_mark_every_property_as_required():
    """Trường có mặc định vẫn luôn được gửi, nên kiểu TypeScript không được để `field?:`."""
    schemas_by_name = build_openapi()["components"]["schemas"]
    checked = 0
    for model in _subclasses(insight_models.ApiModel):
        schema = schemas_by_name.get(model.__name__)
        if schema is None:  # chưa có route nào dùng
            continue
        checked += 1
        assert set(schema["properties"]) <= set(schema.get("required", [])), model.__name__
    assert checked > 20


def test_exported_json_is_valid_and_deterministic():
    first = render(build_openapi())
    assert json.loads(first)["info"]["title"] == "EduPulse API"
    assert first == render(build_openapi())
