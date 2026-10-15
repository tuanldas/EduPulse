"""Lỗi API luôn có dạng JSON `{"detail": {"code", "message"}}`.

`code` là khoá ổn định để giao diện dịch; `message` chỉ để người phát triển đọc. Thông điệp lỗi và
log lỗi không bao giờ chứa nội dung người dùng gửi lên (câu, tên cột, id): chỉ ghi loại ngoại lệ
và khuôn đường dẫn.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException

from edupulse.api.access_log import route_template
from edupulse.config import MAX_CHARS

logger = logging.getLogger("edupulse.api")

# Mã trạng thái do Starlette/FastAPI tự sinh (không qua ApiError) → (code, message).
_STATUS_ERRORS = {
    400: ("bad_request", "Yêu cầu không hợp lệ."),
    404: ("not_found", "Không tìm thấy tài nguyên."),
    405: ("method_not_allowed", "Phương thức không được hỗ trợ."),
    413: ("payload_too_large", "Dữ liệu gửi lên quá lớn."),
    415: ("unsupported_media_type", "Loại dữ liệu không được hỗ trợ."),
    429: ("busy", "Máy chủ đang bận, hãy thử lại sau."),
}
_FALLBACK_ERROR = ("http_error", "Yêu cầu không thành công.")


class ApiError(StarletteHTTPException):
    """Lỗi nghiệp vụ có mã ổn định; handler bên dưới trả đúng `{"detail": {code, message}}`.

    `params` là các số nguyên đi kèm (vd. số dòng của file) để giao diện tự ghép câu thông báo;
    không đưa vào đó dữ liệu người dùng dạng chuỗi.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
        params: dict[str, int] | None = None,
    ) -> None:
        detail: dict[str, object] = {"code": code, "message": message}
        if params:
            detail["params"] = params
        super().__init__(status_code=status_code, detail=detail, headers=headers)


def not_implemented() -> ApiError:
    """Route đã có trong hợp đồng nhưng chưa làm xong: lỗi rõ ràng, không trả dữ liệu giả."""
    return ApiError(501, "not_implemented", "Chức năng này chưa được cài đặt.")


def _is_api_detail(detail: object) -> bool:
    return isinstance(detail, dict) and "code" in detail and "message" in detail


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> Response:
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=exc.headers)
    detail = exc.detail
    if not _is_api_detail(detail):
        code, message = _STATUS_ERRORS.get(exc.status_code, _FALLBACK_ERROR)
        detail = {"code": code, "message": message}
    return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=exc.headers)


# Loại lỗi do validator tuỳ biến phát ra (xem schemas.Text) -> (mã lỗi, thông báo, số đi kèm).
_CODED_VALIDATION_ERRORS: dict[str, tuple[str, dict[str, int] | None]] = {
    "text_empty": ("Câu không được để trống.", None),
    "text_too_long": (f"Câu dài hơn {MAX_CHARS} ký tự.", {"max_chars": MAX_CHARS}),
}


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Chỉ trả vị trí và loại lỗi: `input` của pydantic chứa nguyên văn dữ liệu người dùng gửi.
    errors = exc.errors()
    fields = [{"loc": list(error["loc"]), "type": error["type"]} for error in errors]
    detail: dict[str, object] = {
        "code": "validation_error",
        "message": "Dữ liệu gửi lên không hợp lệ.",
        "fields": fields,
    }
    coded = next((e["type"] for e in errors if e["type"] in _CODED_VALIDATION_ERRORS), None)
    if coded is not None:
        message, params = _CODED_VALIDATION_ERRORS[coded]
        detail.update(code=coded, message=message)
        if params:
            detail["params"] = params
    return JSONResponse({"detail": detail}, status_code=422)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Lỗi không xử lý %s tại %s", type(exc).__name__, route_template(request.scope))
    detail = {"code": "internal_error", "message": "Lỗi máy chủ."}
    return JSONResponse({"detail": detail}, status_code=500)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
