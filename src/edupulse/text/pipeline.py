"""Điểm vào DUY NHẤT của tiền xử lý văn bản, cho cả `ml/` (huấn luyện) lẫn `api/` (phục vụ).

    raw -NFC-> ẩn PII --+--> display_text  ([TÊN] [SĐT]..., giữ hoa/thường)
                        +--> model_pre     (tên -> wzjwz<k>; PII khác xoá)
    model_pre -> nhận diện không dấu
    model_pre -> chữ thường -> dấu thanh -> token đặc biệt -> gộp lặp -> tách dấu câu
              -> RDRSegmenter -> model_text

`NORMALIZER_VERSION` đi theo dữ liệu đã tiền xử lý, bundle và API: đổi bất kỳ bước nào ở đây (hay
trong `normalize.py`, `pii.py`) thì phải tăng nó, chạy lại `make data` và huấn luyện lại.

Lịch sử:
- "1": bản đầu.
- "2": ẩn thêm tên sau "thầy giáo", "cô giáo", "hiệu trưởng", "trưởng khoa", "trợ giảng", "cố vấn"
  (bỏ qua chức danh đứng trước tên); MSSV chữ + số (B20DCCN123); SĐT bàn có ngoặc ((028) 3812
  3456); URL không có http của các mạng xã hội thông dụng; @tên tài khoản. Dữ liệu UIT-VSFC toàn
  chữ thường và không có PII nên "1" và "2" cho model_text giống hệt trên cả 16.175 câu: mô hình
  đã huấn luyện và số liệu test của bản "1" vẫn đúng nguyên vẹn (bằng chứng ghi trong
  `results/metrics/normalizer_v2_invariance.json`, do `python -m ml.normalizer_invariance` sinh).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from edupulse.config import load_settings
from edupulse.text.normalize import clean_text, is_no_diacritics, normalize_text
from edupulse.text.pii import mask
from edupulse.text.segment import get_segmenter

NORMALIZER_VERSION = "2"

# Chặn đầu vào bất thường trước khi chạy regex; API đã giới hạn 500 ký tự mỗi câu.
MAX_INPUT_CHARS = 5000


@dataclass(frozen=True, slots=True)
class Prepared:
    display_text: str  # câu đã ẩn PII để hiển thị/xuất, giữ hoa/thường
    model_text: str  # câu đưa vào mô hình: chữ thường, đã tách từ (từ ghép nối bằng "_")
    pii_types: tuple[str, ...]  # loại PII đã ẩn: name, phone, email, url, student_id, national_id
    no_diacritics: bool  # nghi gõ không dấu (mô hình kém hẳn, nên luôn gắn cờ)
    n_words: int  # số từ (token cách nhau bằng dấu cách) của model_text, gồm cả dấu câu


def _prepare(raw: str, use_ner: bool) -> Prepared:
    if not isinstance(raw, str):
        raise TypeError(f"cần chuỗi, nhận {type(raw).__name__}")
    if len(raw) > MAX_INPUT_CHARS:
        raise ValueError(f"văn bản dài {len(raw)} ký tự, tối đa {MAX_INPUT_CHARS}")
    pii = mask(clean_text(raw), use_ner=use_ner)
    model_text = get_segmenter().segment(normalize_text(pii.model))
    return Prepared(
        display_text=pii.display,
        model_text=model_text,
        pii_types=pii.types,
        no_diacritics=is_no_diacritics(pii.model),
        n_words=len(model_text.split()),
    )


def prepare(raw: str, use_ner: bool | None = None) -> Prepared:
    """Tiền xử lý một câu. `use_ner=None` thì theo `EDUPULSE_NER`."""
    return _prepare(raw, load_settings().ner if use_ner is None else use_ner)


def prepare_many(texts: Iterable[str], use_ner: bool | None = None) -> list[Prepared]:
    """Tiền xử lý nhiều câu (JVM nạp một lần, đọc cấu hình một lần)."""
    ner = load_settings().ner if use_ner is None else use_ner
    return [_prepare(text, ner) for text in texts]
