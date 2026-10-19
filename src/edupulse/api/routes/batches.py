"""Phân tích hàng loạt: tải file, chạy nền, theo dõi, kết quả, giải thích, insight, xuất file.

Endpoint khai báo bằng `def` (chạy trong threadpool). Trạng thái batch nằm trong RAM của đúng một
worker (xem `jobs`); id là chuỗi ngẫu nhiên 128 bit, còn `demo` là batch chỉ đọc dựng từ dữ liệu
demo.
"""

from typing import Annotated

from fastapi import APIRouter, File, Path, Response, UploadFile

from edupulse.api import schemas
from edupulse.api.errors import not_implemented
from edupulse.api.limits import DEMO_BATCH_ID

router = APIRouter(prefix="/api", tags=["batches"])

BatchId = Annotated[str, Path(max_length=64, description=f"Id batch, hoặc `{DEMO_BATCH_ID}`.")]
RowIndex = Annotated[int, Path(ge=0)]


def _errors(codes: dict[int, str]) -> dict[int | str, dict[str, object]]:
    """Khai báo các lỗi có thể gặp: thân lỗi chuẩn, mô tả liệt kê các `detail.code` của từng mã."""
    return {
        status: {"model": schemas.ErrorResponse, "description": description}
        for status, description in codes.items()
    }


NOT_FOUND = {404: "batch_not_found (không có hoặc đã hết hạn)"}


@router.post(
    "/batches",
    status_code=201,
    responses=_errors(
        {
            413: "file_too_large",
            415: "unsupported_file_type",
            422: "too_many_rows (params n_rows, max_rows) | no_rows | decode_failed",
            429: "busy (đã đủ số batch, hoặc hàng đợi đầy)",
        }
    ),
    summary="Tải file CSV/XLSX lên",
)
def upload_batch(
    file: Annotated[UploadFile, File(description="CSV hoặc XLSX, tối đa 10MB và 2.000 dòng.")],
) -> schemas.BatchUpload:
    raise not_implemented()


@router.post(
    "/batches/sample",
    status_code=201,
    responses=_errors({404: "demo_unavailable", 429: "busy"}),
    summary="Tạo batch từ file mẫu dựng lúc build image",
)
def create_sample_batch() -> schemas.BatchUpload:
    raise not_implemented()


@router.post(
    "/batches/{batch_id}/run",
    status_code=202,
    responses=_errors(
        {
            **NOT_FOUND,
            409: "already_running | batch_readonly",
            422: "invalid_column | no_text_column | too_many_groups (params n_groups, max_groups)",
            429: "busy (hàng đợi chạy đầy)",
        }
    ),
    summary="Chạy phân tích nền",
)
def run_batch(batch_id: BatchId, body: schemas.RunRequest) -> schemas.BatchStatus:
    raise not_implemented()


@router.get("/batches/{batch_id}", responses=_errors(NOT_FOUND), summary="Trạng thái và tiến độ")
def batch_status(batch_id: BatchId) -> schemas.BatchStatus:
    raise not_implemented()


@router.post(
    "/batches/{batch_id}/heartbeat",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Gia hạn batch khi tab còn mở",
)
def heartbeat(batch_id: BatchId) -> Response:
    raise not_implemented()


@router.post(
    "/batches/{batch_id}/close",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Xoá batch khi đóng tab (navigator.sendBeacon chỉ gửi được POST)",
)
def close_batch(batch_id: BatchId) -> Response:
    raise not_implemented()


@router.delete(
    "/batches/{batch_id}",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Xoá batch",
)
def delete_batch(batch_id: BatchId) -> Response:
    raise not_implemented()


@router.get(
    "/batches/{batch_id}/results",
    responses=_errors({**NOT_FOUND, 409: "not_done | batch_readonly"}),
    summary="Kết quả từng dòng và thống kê",
)
def batch_results(batch_id: BatchId) -> schemas.ResultsResponse:
    raise not_implemented()


@router.get(
    "/batches/{batch_id}/rows/{index}/explain",
    responses=_errors(
        {
            404: "batch_not_found | row_not_found",
            409: "not_done | batch_readonly",
            422: "xai_too_long (params n_words, max_words) | no_content",
        }
    ),
    summary="Giải thích một dòng (tính khi mở nếu chưa có sẵn)",
)
def explain_row(batch_id: BatchId, index: RowIndex) -> schemas.ExplainResponse:
    raise not_implemented()


@router.post(
    "/batches/{batch_id}/insights",
    responses=_errors({**NOT_FOUND, 409: "not_done"}),
    summary="Dashboard & insight, lọc nhóm qua thân yêu cầu",
)
def batch_insights(batch_id: BatchId, body: schemas.InsightsRequest) -> schemas.InsightsResponse:
    raise not_implemented()


@router.post(
    "/batches/{batch_id}/export",
    response_class=Response,
    responses={
        **_errors(
            {
                **NOT_FOUND,
                409: "not_done | batch_readonly",
                422: "invalid_override (index không có trong batch, hoặc trùng)",
            }
        ),
        200: {
            "description": "CSV UTF-8 có BOM, mở thẳng được bằng Excel.",
            "content": {"text/csv": {"schema": {"type": "string"}}},
        },
    },
    summary="Xuất CSV: toàn bộ kết quả hoặc chỉ nhãn đã sửa",
)
def export_batch(batch_id: BatchId, body: schemas.ExportRequest) -> Response:
    raise not_implemented()
