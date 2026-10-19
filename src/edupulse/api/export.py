"""Xuất kết quả batch ra CSV: toàn bộ kết quả để đọc, hoặc chỉ nhãn đã sửa để huấn luyện lại.

- Mã hoá UTF-8 có BOM (`utf-8-sig`) nên Excel tiếng Việt mở thẳng đúng dấu.
- Chỉ có câu ĐÃ ẩn thông tin cá nhân; không bao giờ có dữ liệu gốc của file.
- Chống CSV/formula injection (OWASP): ô văn bản bắt đầu bằng `= + - @`, các dạng toàn góc
  `＝ ＋ － ＠`, TAB hoặc CR được thêm dấu `'` đứng trước. Người đọc lại file này (vd. script
  huấn luyện lại) dùng `unguard_cell` để bỏ dấu đó.
- Nhãn sửa đến từ giao diện (giữ ở client) qua `overrides`; máy chủ không lưu gì.
- Hai kiểu file: `all` (cột tiếng Việt, mọi dòng) và `corrected` (cột máy đọc được, chỉ các dòng có
  nhãn bị sửa, kèm `text_model` là câu đã chuẩn hoá mà mô hình thật sự nhận).
"""

import csv
import hashlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from edupulse.api.errors import ApiError
from edupulse.api.jobs import Batch, BatchResult, RowData
from edupulse.api.schemas import LabelOverride
from edupulse.labels import SENTIMENT_VI, TOPIC_VI

CSV_MEDIA_TYPE = "text/csv; charset=utf-8"
ALL_FILENAME = "edupulse-ket-qua.csv"
CORRECTED_FILENAME = "edupulse-nhan-da-sua.csv"

# Ký tự đầu ô khiến Excel/LibreOffice coi ô là công thức (hoặc lệnh DDE).
DANGEROUS_LEADS = ("=", "+", "-", "@", "\t", "\r", "＝", "＋", "－", "＠")
GUARD = "'"

SOURCE_VI = {"flag": "Gắn cờ", "audit": "Kiểm tra ngẫu nhiên", "none": ""}
REASON_SEPARATOR = "; "
NO_CONTENT_NOTE = "Không có nội dung để phân tích"

ALL_HEADER = (
    "STT",
    "Câu (đã ẩn thông tin cá nhân)",
    "Cảm xúc",
    "Tin cậy cảm xúc (%)",
    "Chủ đề",
    "Tin cậy chủ đề (%)",
    "Cần xem xét",
    "Lý do",
    "Nguồn",
    "Cảm xúc đã sửa",
    "Chủ đề đã sửa",
)
CORRECTED_HEADER = (
    "id_hash",
    "text_masked",
    "text_model",
    "model_version",
    "pred_sentiment",
    "conf_sentiment",
    "pred_topic",
    "conf_topic",
    "label_sentiment",
    "label_topic",
    "source",
    "reviewed",
)


def guard_cell(value: str) -> str:
    """Thêm `'` trước ô có thể bị hiểu là công thức."""
    return GUARD + value if value.startswith(DANGEROUS_LEADS) else value


def unguard_cell(value: str) -> str:
    """Ngược lại của `guard_cell`, cho người đọc lại file đã xuất."""
    if value.startswith(GUARD) and value[1:].startswith(DANGEROUS_LEADS):
        return value[1:]
    return value


@dataclass(frozen=True, slots=True)
class Correction:
    """Nhãn do người duyệt sửa (None = giữ nhãn mô hình); chỉ giữ giá trị KHÁC dự đoán."""

    sentiment: str | None
    topic: str | None


def resolve_overrides(
    result: BatchResult, overrides: Iterable[LabelOverride]
) -> dict[int, Correction]:
    """Kiểm tra `overrides` rồi trả `{chỉ số dòng: Correction}` chỉ gồm các nhãn thật sự đổi.

    Chỉ số phải thuộc một dòng có nhãn và không được lặp. Giá trị trùng nhãn dự đoán là xác nhận
    chứ không phải sửa nên bị bỏ.
    """
    corrections: dict[int, Correction] = {}
    seen: set[int] = set()
    for item in overrides:
        if item.index in seen:
            raise ApiError(422, "invalid_override", "Có dòng bị sửa nhãn hai lần.")
        seen.add(item.index)
        row = result.rows[item.index] if item.index < len(result.rows) else None
        if row is None or row.analysis is None:
            raise ApiError(
                422, "invalid_override", "Có dòng sửa nhãn không tồn tại hoặc không có nhãn."
            )
        sentiment = item.sentiment if item.sentiment != row.analysis.sentiment.label else None
        topic = item.topic if item.topic != row.analysis.topic.label else None
        if sentiment is not None or topic is not None:
            corrections[item.index] = Correction(sentiment, topic)
    return corrections


def _percent(confidence: float) -> str:
    return str(round(confidence * 100))


def _source(row: RowData) -> str:
    analysis = row.analysis
    if analysis is not None and analysis.flag.needs_review:
        return "flag"
    return "audit" if row.audit else "none"


def _write(rows: Iterable[Sequence[str]]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    for row in rows:
        writer.writerow([guard_cell(cell) for cell in row])
    return out.getvalue().encode("utf-8-sig")


def render_all(batch: Batch, result: BatchResult, corrections: dict[int, Correction]) -> bytes:
    """Mọi dòng của batch (kể cả dòng không có nhãn) kèm nhãn đã sửa nếu có."""
    has_groups = batch.group_col is not None
    header = [*ALL_HEADER, *([batch.group_col] if has_groups and batch.group_col else [])]
    lines: list[list[str]] = [header]
    for row in result.rows:
        analysis = row.analysis
        group = [row.group or ""] if has_groups else []
        if analysis is None:
            lines.append(
                [
                    str(row.source_row),
                    row.prepared.display_text,
                    "",
                    "",
                    "",
                    "",
                    "",
                    NO_CONTENT_NOTE,
                ]
                + ["", "", ""]
                + group
            )
            continue
        fix = corrections.get(row.index)
        lines.append(
            [
                str(row.source_row),
                row.prepared.display_text,
                SENTIMENT_VI[analysis.sentiment.label],
                _percent(analysis.sentiment.confidence),
                TOPIC_VI[analysis.topic.label],
                _percent(analysis.topic.confidence),
                "Có" if analysis.flag.needs_review else "Không",
                REASON_SEPARATOR.join(r.message for r in analysis.flag.reasons),
                SOURCE_VI[_source(row)],
                SENTIMENT_VI[fix.sentiment] if fix and fix.sentiment else "",
                TOPIC_VI[fix.topic] if fix and fix.topic else "",
                *group,
            ]
        )
    return _write(lines)


def render_corrected(
    result: BatchResult, corrections: dict[int, Correction], model_version: str
) -> bytes:
    """Chỉ các dòng có nhãn bị sửa, ở dạng máy đọc được để huấn luyện lại.

    `label_*` là nhãn cuối cùng (sửa nếu có, không thì nhãn dự đoán); `pred_*` giữ dự đoán gốc của
    mô hình để đo độ chính xác của vòng duyệt. `id_hash` là băm của `text_model` để khử trùng.
    """
    lines: list[list[str]] = [list(CORRECTED_HEADER)]
    for row in result.rows:
        fix = corrections.get(row.index)
        analysis = row.analysis
        if fix is None or analysis is None:
            continue
        lines.append(
            [
                hashlib.sha256(row.prepared.model_text.encode("utf-8")).hexdigest()[:16],
                row.prepared.display_text,
                row.prepared.model_text,
                model_version,
                analysis.sentiment.label,
                f"{analysis.sentiment.confidence:.4f}",
                analysis.topic.label,
                f"{analysis.topic.confidence:.4f}",
                fix.sentiment or analysis.sentiment.label,
                fix.topic or analysis.topic.label,
                _source(row),
                "1",
            ]
        )
    return _write(lines)
