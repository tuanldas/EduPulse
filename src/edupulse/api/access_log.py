"""Nhật ký truy cập tối giản: `METHOD khuôn-đường-dẫn status thời-gian`.

Uvicorn chạy với `--no-access-log` vì log mặc định của nó ghi cả query string. Ở đây chỉ ghi
khuôn đường dẫn của route đã khớp (vd `/api/batches/{batch_id}/results`), không ghi id thật,
query hay thân yêu cầu, nên log không bao giờ chứa nội dung góp ý của sinh viên.
"""

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edupulse.api.paths import is_api_path

logger = logging.getLogger("edupulse.access")

STATIC = "static"  # yêu cầu rơi vào bản web export tĩnh
UNMATCHED = "unmatched"  # không route nào khớp (đường dẫn API sai, hoặc chưa có bản web)


def route_template(scope: Scope) -> str:
    """Khuôn đường dẫn của route đã khớp, an toàn để ghi log vì không chứa giá trị người dùng."""
    # FastAPI ghi khuôn đầy đủ (gồm tiền tố của include_router) và đánh dấu route web tĩnh trong
    # scope["fastapi"]; `route.path` bên dưới chỉ là phương án dự phòng khi các khoá đó đổi.
    fastapi_scope = scope.get("fastapi")
    if isinstance(fastapi_scope, dict):
        if "frontend_path" in fastapi_scope:
            return UNMATCHED if is_api_path(scope["path"]) else STATIC
        path_format = getattr(fastapi_scope.get("effective_route_context"), "path_format", None)
        if isinstance(path_format, str):
            return path_format
    return getattr(scope.get("route"), "path", None) or UNMATCHED


class AccessLogMiddleware:
    """ASGI thuần: đo thời gian rồi ghi một dòng cho mỗi yêu cầu HTTP."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500  # giữ nguyên nếu ứng dụng ném lỗi trước khi gửi phản hồi

        async def send_and_track(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_track)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            logger.info("%s %s %d %dms", scope["method"], route_template(scope), status, elapsed_ms)
