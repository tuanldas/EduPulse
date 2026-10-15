"""Mô hình dữ liệu của API: lỗi và thông tin sức khoẻ.

Giá trị phân loại (`Literal`) khoá bằng tiếng Anh; giao diện ánh xạ sang tiếng Việt.
"""

from typing import Literal

from pydantic import BaseModel


class FieldError(BaseModel):
    """Một trường không hợp lệ; không kèm giá trị gửi lên để không lặp lại nội dung người dùng."""

    loc: list[str | int]
    type: str


class ErrorBody(BaseModel):
    code: str  # khoá ổn định để giao diện dịch
    message: str  # chỉ để người phát triển đọc
    fields: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]
    bundle_version: str
    ner_enabled: bool
    rss_mb: float  # bộ nhớ thường trú của tiến trình
