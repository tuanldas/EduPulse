"""Giới hạn dung lượng upload ở tầng ASGI, TRƯỚC khi Starlette parse multipart.

Starlette đọc hết thân multipart (file lớn hơn 1MB được ghi ra file tạm) trước khi handler chạy, và
`max_part_size` không áp cho file. Nếu chỉ kiểm trong handler thì một yêu cầu 50MB vẫn bị nhận hết.
Middleware này chặn sớm cho `POST /api/batches`:

- `Content-Length` khai quá giới hạn -> 413 ngay, KHÔNG đọc thân yêu cầu;
- không có `Content-Length` (chunked) hoặc khai sai -> đếm byte nhận được, vượt thì trả 413 và cho
  ứng dụng thấy kết nối đã đóng, nên thân còn lại không bao giờ được đọc.

Giới hạn ở đây là dung lượng file cộng phần overhead của multipart; kiểm tra chính xác dung lượng
file nằm ở handler (`file_too_large`). Phản hồi 413 đi qua các middleware ngoài (header an toàn,
nhật ký), và chỉ ghi khuôn đường dẫn như mọi route khác.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edupulse.config import MAX_UPLOAD_MB

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024  # ranh giới, tên trường và header của phần multipart
UPLOAD_PATH = "/api/batches"

_TOO_LARGE = {
    "detail": {
        "code": "file_too_large",
        "message": f"File vượt quá {MAX_UPLOAD_MB}MB.",
        "params": {"max_mb": MAX_UPLOAD_MB},
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
    def __init__(
        self,
        app: ASGIApp,
        *,
        path: str = UPLOAD_PATH,
        max_bytes: int = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD,
    ) -> None:
        self.app = app
        self.path = path
        self.max_bytes = max_bytes

    def _applies(self, scope: Scope) -> bool:
        return (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == self.path
        )

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps(_TOO_LARGE, ensure_ascii=False).encode("utf-8")
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
