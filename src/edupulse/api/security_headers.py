"""Header an toàn cho mọi phản hồi.

Ứng dụng xử lý góp ý của sinh viên nên đường dẫn có thể mang id phiên (`?batch=`): không gửi
Referer sang trang ngoài, không cho nhúng vào khung của trang khác, và không cho trình duyệt hay
proxy lưu phản hồi của `/api`.
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edupulse.api.paths import is_api_path

_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class SecurityHeadersMiddleware:
    """ASGI thuần: thêm header nếu phản hồi chưa tự đặt."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        no_store = is_api_path(scope["path"])

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _HEADERS.items():
                    headers.setdefault(name, value)
                if no_store:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)
