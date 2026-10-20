"""Giới hạn dung lượng thân yêu cầu ở tầng ASGI, TRƯỚC khi Starlette hay FastAPI đọc nó.

Starlette đọc hết thân multipart (file lớn hơn 1MB được ghi ra file tạm) trước khi handler chạy, và
`max_part_size` không áp cho file. FastAPI cũng đọc nguyên thân JSON vào RAM trước khi kiểm tra bất
cứ gì. Nếu chỉ kiểm trong handler thì một yêu cầu 50MB (hay 250MB JSON) vẫn bị nhận hết. Hai lớp
dưới đây chặn sớm:

- `UploadLimitMiddleware` cho `POST /api/batches` (file tải lên, 10MB);
- `JsonBodyLimitMiddleware` cho mọi POST/PUT/PATCH/DELETE còn lại (thân JSON, 512KB).

Cả hai làm giống nhau:

- `Content-Length` khai quá giới hạn -> 413 ngay, KHÔNG đọc thân yêu cầu;
- không có `Content-Length` (chunked) hoặc khai sai -> đếm byte nhận được, vượt thì trả 413 và cho
  ứng dụng thấy kết nối đã đóng, nên thân còn lại không bao giờ được đọc.

Giới hạn upload là dung lượng file cộng phần overhead của multipart; kiểm tra chính xác dung lượng
file nằm ở handler (`file_too_large`). Phản hồi 413 đi qua các middleware ngoài (header an toàn,
nhật ký), và chỉ ghi khuôn đường dẫn như mọi route khác.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edupulse.config import MAX_UPLOAD_MB

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024  # ranh giới, tên trường và header của phần multipart
UPLOAD_PATH = "/api/batches"
# Thân JSON lớn nhất hợp lệ là yêu cầu xuất file với 2.000 chỉnh nhãn (~140KB).
MAX_JSON_BYTES = 512 * 1024
JSON_METHODS = ("POST", "PUT", "PATCH", "DELETE")

_TOO_LARGE = {
    "detail": {
        "code": "file_too_large",
        "message": f"File vượt quá {MAX_UPLOAD_MB}MB.",
        "params": {"max_mb": MAX_UPLOAD_MB},
    }
}
_JSON_TOO_LARGE = {
    "detail": {
        "code": "payload_too_large",  # cùng mã với lỗi 413 chung của Starlette (errors.py)
        "message": "Dữ liệu gửi lên quá lớn.",
        "params": {"max_kb": MAX_JSON_BYTES // 1024},
    }
}


def _declared_length(scope: Scope) -> int | None:
    """Giá trị `Content-Length` hợp lệ đầu tiên; thiếu hoặc không phải số thì coi như chưa khai."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return length if length >= 0 else None
    return None


class UploadLimitMiddleware:
    """Trần thân yêu cầu cho một đường dẫn (`path`), hoặc cho mọi đường dẫn trừ `exclude`."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        path: str | None = UPLOAD_PATH,
        exclude: str | None = None,
        methods: tuple[str, ...] = ("POST",),
        max_bytes: int = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD,
        detail: dict = _TOO_LARGE,
    ) -> None:
        self.app = app
        self.path = path
        self.exclude = exclude
        self.methods = methods
        self.max_bytes = max_bytes
        self.detail = detail

    def _applies(self, scope: Scope) -> bool:
        if scope["type"] != "http" or scope["method"] not in self.methods:
            return False
        target = scope["path"].rstrip("/")
        return target != self.exclude and (self.path is None or target == self.path)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(self.detail, ensure_ascii=False).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            # Thân yêu cầu có thể chưa đọc hết: đóng kết nối thay vì đọc nốt để tái sử dụng.
            (b"connection", b"close"),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    await self._reject(send)
                    # Ứng dụng thấy máy khách ngắt kết nối nên dừng đọc và bỏ phần thân còn lại.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            # Sau khi đã trả 413, những gì ứng dụng cố gửi tiếp (vd. lỗi 400 vì thân bị cắt) bị bỏ.
            if not rejected:
                await send(message)

        await self.app(scope, limited_receive, guarded_send)


class JsonBodyLimitMiddleware(UploadLimitMiddleware):
    """Trần `MAX_JSON_BYTES` cho thân của mọi POST/PUT/PATCH/DELETE ngoài đường dẫn upload."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_JSON_BYTES) -> None:
        super().__init__(
            app,
            path=None,
            exclude=UPLOAD_PATH,
            methods=JSON_METHODS,
            max_bytes=max_bytes,
            detail=_JSON_TOO_LARGE,
        )
