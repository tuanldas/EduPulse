"""Phân tích hàng loạt: tải file, chạy nền, theo dõi, kết quả, giải thích, insight, xuất file.

Endpoint khai báo bằng `def` (chạy trong threadpool). Trạng thái batch nằm trong RAM của đúng một
worker (xem `jobs`); id là chuỗi ngẫu nhiên 128 bit, còn `demo` là batch chỉ đọc dựng từ dữ liệu
demo.
"""

from typing import Annotated

from fastapi import APIRouter, File, Path, Response, UploadFile

from edupulse.api import schemas
from edupulse.api.batch_view import results_response, status_response, upload_response
from edupulse.api.deps import AnalyzerDep, BackgroundDep, DemoDep, StoreDep
from edupulse.api.errors import ApiError
from edupulse.api.export import (
    ALL_FILENAME,
    CORRECTED_FILENAME,
    CSV_MEDIA_TYPE,
    render_all,
    render_corrected,
    resolve_overrides,
)
from edupulse.api.jobs import Batch
from edupulse.api.limits import DEMO_BATCH_ID
from edupulse.api.parsing import is_illustrative, parse_upload
from edupulse.api.routes.analyze import NO_CONTENT_MESSAGE, to_explain_response
from edupulse.api.upload_limit import MAX_UPLOAD_BYTES
from edupulse.config import MAX_UPLOAD_MB
from edupulse.engine.explain import ExplanationUnavailable
from edupulse.insights.build import build_insights

router = APIRouter(prefix="/api", tags=["batches"])

BatchId = Annotated[str, Path(max_length=64, description=f"Id batch, hoặc `{DEMO_BATCH_ID}`.")]
RowIndex = Annotated[int, Path(ge=0)]


SAMPLE_FILENAME = "edupulse-file-mau.csv"
INSIGHTS_CACHE_SIZE = 16  # số cách lọc nhóm được nhớ cho mỗi batch


def _finished(batch: Batch) -> Batch:
    """Batch đã chạy xong và có kết quả để đọc; demo và batch chưa xong thì 409."""
    if batch.readonly:
        raise ApiError(409, "batch_readonly", "Batch demo chỉ có insight, không có kết quả dòng.")
    if batch.state != "done" or batch.result is None:
        raise ApiError(409, "not_done", "Batch chưa phân tích xong.")
    return batch


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
    store: StoreDep,
) -> schemas.BatchUpload:
    # Middleware đã chặn thân quá lớn trước khi parse; ở đây kiểm đúng dung lượng của file.
    too_large = ApiError(
        413, "file_too_large", f"File vượt quá {MAX_UPLOAD_MB}MB.", params={"max_mb": MAX_UPLOAD_MB}
    )
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise too_large
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise too_large
    filename = file.filename or ""
    table = parse_upload(filename, data)
    del data
    batch = store.create(
        filename, table, illustrative=is_illustrative("upload", None, table.columns)
    )
    return upload_response(batch, table)


@router.post(
    "/batches/sample",
    status_code=201,
    responses=_errors({404: "demo_unavailable", 429: "busy"}),
    summary="Tạo batch từ file mẫu dựng lúc build image",
)
def create_sample_batch(demo: DemoDep, store: StoreDep) -> schemas.BatchUpload:
    if not demo.available:
        raise ApiError(404, "demo_unavailable", demo.reason or "Chưa có file mẫu.")
    table = parse_upload("sample.csv", demo.sample_csv)
    batch = store.create(SAMPLE_FILENAME, table, kind="sample", illustrative=True)
    return upload_response(batch, table)


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
def run_batch(batch_id: BatchId, body: schemas.RunRequest, store: StoreDep) -> schemas.BatchStatus:
    batch = store.get(batch_id)
    snapshot = store.run(
        batch, text_col=body.text_col, group_col=body.group_col, budget=body.review_budget
    )
    return status_response(snapshot)


@router.get("/batches/{batch_id}", responses=_errors(NOT_FOUND), summary="Trạng thái và tiến độ")
def batch_status(batch_id: BatchId, store: StoreDep) -> schemas.BatchStatus:
    return status_response(store.snapshot(store.get(batch_id)))


@router.post(
    "/batches/{batch_id}/heartbeat",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Gia hạn batch khi tab còn mở",
)
def heartbeat(batch_id: BatchId, store: StoreDep) -> Response:
    store.get(batch_id)  # tra cứu cũng gia hạn TTL; id lạ hoặc hết hạn thì 404
    return Response(status_code=204)


@router.post(
    "/batches/{batch_id}/close",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Xoá batch khi đóng tab (navigator.sendBeacon chỉ gửi được POST)",
)
def close_batch(batch_id: BatchId, store: StoreDep) -> Response:
    store.remove(batch_id)
    return Response(status_code=204)


@router.delete(
    "/batches/{batch_id}",
    status_code=204,
    responses=_errors(NOT_FOUND),
    summary="Xoá batch",
)
def delete_batch(batch_id: BatchId, store: StoreDep) -> Response:
    store.remove(batch_id)
    return Response(status_code=204)


@router.get(
    "/batches/{batch_id}/results",
    responses=_errors({**NOT_FOUND, 409: "not_done | batch_readonly"}),
    summary="Kết quả từng dòng và thống kê",
)
def batch_results(
    batch_id: BatchId, store: StoreDep, analyzer: AnalyzerDep
) -> schemas.ResultsResponse:
    batch = _finished(store.get(batch_id))
    assert batch.result is not None
    return results_response(batch, batch.result, analyzer.xai_max_words)


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
def explain_row(
    batch_id: BatchId, index: RowIndex, store: StoreDep, analyzer: AnalyzerDep
) -> schemas.ExplainResponse:
    batch = _finished(store.get(batch_id))
    result = batch.result
    assert result is not None
    if index >= len(result.rows):
        raise ApiError(404, "row_not_found", "Batch không có dòng này.")
    row = result.rows[index]
    if row.analysis is None:
        raise ApiError(422, "no_content", NO_CONTENT_MESSAGE)
    with batch.lock:
        explanation = result.explanations.get(index)
    if explanation is None:
        try:
            explanation = analyzer.explain_prepared(row.prepared)
        except ExplanationUnavailable as exc:
            raise ApiError(
                422,
                exc.code,
                f"Câu có {exc.n_words} từ, giải thích chỉ hỗ trợ tối đa {exc.max_words} từ.",
                params={"n_words": exc.n_words, "max_words": exc.max_words},
            ) from exc
        with batch.lock:
            explanation = result.explanations.setdefault(index, explanation)
    return to_explain_response(explanation)


@router.post(
    "/batches/{batch_id}/insights",
    responses=_errors(
        {**NOT_FOUND, 409: "not_done", 503: "insights_unavailable (bundle thiếu số liệu nền)"}
    ),
    summary="Dashboard & insight, lọc nhóm qua thân yêu cầu",
)
def batch_insights(
    batch_id: BatchId, body: schemas.InsightsRequest, store: StoreDep, background: BackgroundDep
) -> schemas.InsightsResponse:
    batch = store.get(batch_id)
    if not batch.readonly:
        _finished(batch)
    key = tuple(sorted(set(body.groups or ())))
    with batch.lock:
        cached = batch.insights_cache.get(key)
    if cached is not None:
        return cached
    insights = build_insights(
        store.insight_rows(batch),
        background,
        source="demo" if batch.readonly else "batch",
        illustrative_groups=batch.illustrative_groups,
        groups=key,
    )
    with batch.lock:
        if len(batch.insights_cache) >= INSIGHTS_CACHE_SIZE:
            batch.insights_cache.pop(next(iter(batch.insights_cache)))
        batch.insights_cache[key] = insights
    return insights


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
def export_batch(
    batch_id: BatchId, body: schemas.ExportRequest, store: StoreDep, analyzer: AnalyzerDep
) -> Response:
    batch = _finished(store.get(batch_id))
    result = batch.result
    assert result is not None
    corrections = resolve_overrides(result, body.overrides)
    if body.kind == "corrected":
        data = render_corrected(result, corrections, analyzer.bundle.manifest.model_version)
        filename = CORRECTED_FILENAME
    else:
        data = render_all(batch, result, corrections)
        filename = ALL_FILENAME
    return Response(
        data,
        media_type=CSV_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
