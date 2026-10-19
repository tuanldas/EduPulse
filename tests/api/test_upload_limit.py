"""Middleware giới hạn thân yêu cầu: chặn theo Content-Length, và đếm byte khi chunked.

`UploadLimitMiddleware` lo `POST /api/batches` (file); `JsonBodyLimitMiddleware` lo thân JSON của
mọi POST/DELETE còn lại. App thật xếp cả hai, nên các test route khác đều chạy qua cặp đó.
"""

import asyncio
import json
from collections.abc import Iterable

import pytest
from fastapi.testclient import TestClient

from edupulse.api.main import create_app
from edupulse.api.schemas import ExportRequest, LabelOverride
from edupulse.api.upload_limit import (
    MAX_JSON_BYTES,
    MAX_UPLOAD_BYTES,
    MULTIPART_OVERHEAD,
    JsonBodyLimitMiddleware,
    UploadLimitMiddleware,
)
from edupulse.config import MAX_ROWS, Settings

LIMIT = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD
KB = 1 << 10
MB = 1 << 20


def app_stack(app):
    """Hai lớp giới hạn xếp như trong `create_app`."""
    return JsonBodyLimitMiddleware(UploadLimitMiddleware(app))


async def echo_size_app(scope, receive, send):
    """Ứng dụng đọc hết thân yêu cầu như Starlette khi parse multipart rồi trả số byte nhận được."""
    total = 0
    status = 200
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            status = 400  # giống FastAPI khi thân bị cắt giữa chừng
            break
        total += len(message.get("body", b""))
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": status, "headers": []})
    await send({"type": "http.response.body", "body": str(total).encode()})


def call(
    chunks: Iterable[bytes] = (),
    *,
    headers: Iterable[tuple[str, str]] = (),
    method: str = "POST",
    path: str = "/api/batches",
    scope_type: str = "http",
    stack=UploadLimitMiddleware,
):
    """Chạy middleware một lần; trả (các message đã gửi, số lần receive được gọi)."""
    sent: list[dict] = []
    calls = 0
    pending = iter(chunks)

    async def receive():
        nonlocal calls
        calls += 1
        chunk = next(pending, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
    }
    asyncio.run(stack(echo_size_app)(scope, receive, send))
    return sent, calls


def status_of(sent: list[dict]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def body_of(sent: list[dict]) -> bytes:
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def test_declared_length_over_the_limit_is_rejected_without_reading_the_body():
    sent, receive_calls = call(
        [b"x" * MB] * 3, headers=[("Content-Length", str(50 * MB)), ("Content-Type", "multipart/x")]
    )
    assert receive_calls == 0
    assert status_of(sent) == 413
    detail = json.loads(body_of(sent))["detail"]
    assert detail == {
        "code": "file_too_large",
        "message": "File vượt quá 10MB.",
        "params": {"max_mb": 10},
    }
    headers = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    assert headers[b"content-type"] == b"application/json"
    assert headers[b"connection"] == b"close"
    assert int(headers[b"content-length"]) == len(body_of(sent))


def test_the_declared_limit_is_inclusive():
    sent, _ = call([b"x"], headers=[("Content-Length", str(LIMIT))])
    assert status_of(sent) == 200
    sent, receive_calls = call([b"x"], headers=[("Content-Length", str(LIMIT + 1))])
    assert status_of(sent) == 413 and receive_calls == 0


def test_chunked_upload_is_cut_off_when_it_passes_the_limit():
    sent, receive_calls = call([b"x" * MB] * 50)  # không có Content-Length
    assert status_of(sent) == 413
    assert receive_calls <= 12  # dừng đọc sau ~11MB, không đọc hết 50MB
    # Phản hồi 400/200 mà ứng dụng cố gửi sau đó bị bỏ: chỉ có đúng một phản hồi.
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert json.loads(body_of(sent))["detail"]["code"] == "file_too_large"


def test_a_lying_content_length_cannot_hide_a_large_body():
    sent, receive_calls = call([b"x" * MB] * 50, headers=[("Content-Length", "1000")])
    assert status_of(sent) == 413 and receive_calls <= 12


def test_an_unparseable_content_length_falls_back_to_counting():
    sent, receive_calls = call([b"x" * MB] * 50, headers=[("Content-Length", "nhieu")])
    assert status_of(sent) == 413 and receive_calls <= 12


def test_a_body_exactly_at_the_limit_passes_and_one_byte_more_does_not():
    at_limit = [b"x" * MB] * (LIMIT // MB) + [b"x" * (LIMIT % MB)]
    sent, _ = call(at_limit)
    assert status_of(sent) == 200 and body_of(sent) == str(LIMIT).encode()
    sent, _ = call([*at_limit, b"x"])
    assert status_of(sent) == 413


def test_a_normal_small_upload_reaches_the_app_untouched():
    sent, _ = call([b"a" * 1000, b"b" * 500], headers=[("Content-Length", "1500")])
    assert status_of(sent) == 200 and body_of(sent) == b"1500"


@pytest.mark.parametrize("path", ["/api/batches", "/api/batches/"])
def test_both_spellings_of_the_upload_path_are_guarded(path: str):
    sent, _ = call(headers=[("Content-Length", str(50 * MB))], path=path)
    assert status_of(sent) == 413


def test_the_upload_limiter_alone_leaves_every_other_route_to_the_json_limiter():
    sent, _ = call(
        [b"x" * 100], headers=[("Content-Length", str(50 * MB))], method="POST", path="/api/analyze"
    )
    assert status_of(sent) == 200  # UploadLimitMiddleware chỉ lo đường dẫn upload


# --- Thân JSON: 512KB cho mọi POST/DELETE ngoài upload ------------------------------------------

JSON_ROUTES = [
    ("POST", "/api/analyze"),
    ("POST", "/api/explain"),
    ("POST", "/api/batches/abc/run"),
    ("POST", "/api/batches/abc/insights"),
    ("POST", "/api/batches/abc/export"),
    ("POST", "/api/batches/abc/heartbeat"),
    ("POST", "/api/batches/sample"),
    ("DELETE", "/api/batches/abc"),
]


@pytest.mark.parametrize(("method", "path"), JSON_ROUTES)
def test_a_json_body_declared_over_512kb_is_refused_without_reading_it(method: str, path: str):
    sent, receive_calls = call(
        [b"x" * 100],
        headers=[("Content-Length", str(50 * MB))],
        method=method,
        path=path,
        stack=app_stack,
    )
    assert status_of(sent) == 413 and receive_calls == 0
    detail = json.loads(body_of(sent))["detail"]
    assert detail == {
        "code": "payload_too_large",
        "message": "Dữ liệu gửi lên quá lớn.",
        "params": {"max_kb": 512},
    }
    headers = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    assert headers[b"content-type"] == b"application/json" and headers[b"connection"] == b"close"


def test_the_json_limit_is_inclusive_and_is_512kb():
    assert MAX_JSON_BYTES == 512 * KB
    at_limit = [b"x" * KB] * 512
    sent, _ = call(
        at_limit,
        headers=[("Content-Length", str(MAX_JSON_BYTES))],
        path="/api/analyze",
        stack=app_stack,
    )
    assert status_of(sent) == 200 and body_of(sent) == str(MAX_JSON_BYTES).encode()
    sent, receive_calls = call(
        [b"x"],
        headers=[("Content-Length", str(MAX_JSON_BYTES + 1))],
        path="/api/analyze",
        stack=app_stack,
    )
    assert status_of(sent) == 413 and receive_calls == 0
    sent, _ = call([*at_limit, b"x"], path="/api/analyze", stack=app_stack)  # chunked, hơn 1 byte
    assert status_of(sent) == 413


@pytest.mark.parametrize(("method", "path"), JSON_ROUTES)
def test_a_chunked_json_body_is_cut_off_early_and_never_read_to_the_end(method: str, path: str):
    # 250MB nếu đọc hết: đúng kịch bản đo được (thân ~250MB làm container chết).
    sent, receive_calls = call([b"x" * MB] * 250, method=method, path=path, stack=app_stack)
    assert status_of(sent) == 413
    assert receive_calls <= 3  # dừng ở khúc đầu tiên vượt 512KB
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]


def test_a_lying_content_length_cannot_hide_a_large_json_body():
    sent, receive_calls = call(
        [b"x" * MB] * 50, headers=[("Content-Length", "100")], path="/api/analyze", stack=app_stack
    )
    assert status_of(sent) == 413 and receive_calls <= 3


@pytest.mark.parametrize("path", ["/api/batches", "/api/batches/"])
def test_the_upload_path_keeps_its_own_limit_and_error_under_the_json_limiter(path: str):
    sent, _ = call(
        [b"x" * 100], headers=[("Content-Length", str(5 * MB))], path=path, stack=app_stack
    )
    assert status_of(sent) == 200  # 5MB là file hợp lệ, không bị trần JSON
    sent, receive_calls = call(
        [b"x"], headers=[("Content-Length", str(50 * MB))], path=path, stack=app_stack
    )
    assert status_of(sent) == 413 and receive_calls == 0
    assert json.loads(body_of(sent))["detail"]["code"] == "file_too_large"


@pytest.mark.parametrize("path", ["/api/batches", "/api/batches/abc", "/api/health", "/api/config"])
def test_reading_never_hits_a_body_limit(path: str):
    sent, _ = call(
        [b"x" * 100],
        headers=[("Content-Length", str(50 * MB))],
        method="GET",
        path=path,
        stack=app_stack,
    )
    assert status_of(sent) == 200


def test_the_largest_legitimate_json_request_fits_with_plenty_of_room():
    overrides = [
        LabelOverride(index=MAX_ROWS - 1, sentiment="negative", topic="training_program")
        for _ in range(MAX_ROWS)
    ]
    body = json.dumps(ExportRequest(kind="corrected", overrides=overrides).model_dump())
    assert len(body) < MAX_JSON_BYTES // 3  # xuất file 2.000 dòng chỉnh nhãn ~140KB


def test_the_real_app_refuses_oversized_json_before_routing(settings: Settings):
    client = TestClient(create_app(settings))  # không chạy lifespan: 413 đến từ middleware
    big = b"a" * (600 * KB)
    for method, path in JSON_ROUTES:
        response = client.request(
            method, path, content=big, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413, (method, path)
        assert response.json()["detail"]["code"] == "payload_too_large"
        assert response.headers["x-content-type-options"] == "nosniff"  # qua middleware ngoài
    assert client.get("/api/openapi.json").status_code == 200  # đọc thì không bị trần


def test_non_http_scopes_pass_straight_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(UploadLimitMiddleware(app)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
