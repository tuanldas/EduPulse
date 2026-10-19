"""Middleware giới hạn upload: chặn theo Content-Length, và đếm byte khi chunked."""

import asyncio
import json
from collections.abc import Iterable

import pytest

from edupulse.api.upload_limit import (
    MAX_UPLOAD_BYTES,
    MULTIPART_OVERHEAD,
    UploadLimitMiddleware,
)

LIMIT = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD
MB = 1 << 20


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
    asyncio.run(UploadLimitMiddleware(echo_size_app)(scope, receive, send))
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


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/batches"),
        ("POST", "/api/analyze"),
        ("POST", "/api/batches/abc/run"),
        ("POST", "/api/batches/sample"),
        ("DELETE", "/api/batches"),
    ],
)
def test_other_routes_are_not_limited(method: str, path: str):
    sent, _ = call(
        [b"x" * 100], headers=[("Content-Length", str(50 * MB))], method=method, path=path
    )
    assert status_of(sent) == 200


def test_non_http_scopes_pass_straight_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(UploadLimitMiddleware(app)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
