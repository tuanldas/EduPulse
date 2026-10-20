"""Header an toàn cho mọi phản hồi.

Ứng dụng xử lý góp ý của sinh viên nên đường dẫn có thể mang id phiên (`?batch=`): không gửi
Referer sang trang ngoài, không cho nhúng vào khung của trang khác, và không cho trình duyệt hay
proxy lưu phản hồi của `/api`.

Chuyển hướng (vd. `/batch` -> `/batch/`) được Starlette dựng thành URL tuyệt đối từ header Host do
client gửi; middleware đổi `Location` thành đường dẫn tương đối để không phản chiếu Host lạ và không
hạ https xuống http khi đứng sau proxy. App không chuyển hướng sang trang khác nên bỏ scheme và
host là an toàn.
"""

from urllib.parse import urlsplit, urlunsplit

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edupulse.api.paths import is_api_path

_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def relative_location(location: str) -> str:
    """URL chuyển hướng tuyệt đối -> đường dẫn tương đối (giữ query và fragment)."""
    parts = urlsplit(location)
    if not parts.scheme and not parts.netloc:
        return location
    return urlunsplit(("", "", parts.path or "/", parts.query, parts.fragment))


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
                if "location" in headers:
                    headers["location"] = relative_location(headers["location"])
            await send(message)

        await self.app(scope, receive, send_with_headers)
