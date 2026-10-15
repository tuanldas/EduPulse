"""Ẩn thông tin cá nhân (PII) trước khi hiển thị, xuất file hay đưa vào mô hình.

Mỗi câu cho hai góc nhìn từ cùng một tập vị trí đã phát hiện:
- `display`: giữ hoa/thường, thay bằng nhãn `[TÊN] [SĐT] [EMAIL] [MSSV] [CCCD] [URL]`;
- `model`: tên -> `wzjwz<k>` (k phân biệt theo tên trong câu, như dữ liệu gốc), PII khác bị xoá.

Số điện thoại, email, URL, CCCD, MSSV được bắt bằng regex. Tên người chỉ bắt được khi viết hoa:
luật rẻ "từ xưng hô + 1-4 âm tiết Viết Hoa" luôn bật, NER của underthesea là tuỳ chọn và chỉ chạy
khi câu có chữ hoa. Tên viết thường ("thầy nam") KHÔNG bắt được: giới hạn đã biết, phải nói rõ
với người dùng (best effort).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

NAME = "name"
PHONE = "phone"
EMAIL = "email"
URL = "url"
STUDENT_ID = "student_id"
NATIONAL_ID = "national_id"

PLACEHOLDERS: dict[str, str] = {
    NAME: "[TÊN]",
    PHONE: "[SĐT]",
    EMAIL: "[EMAIL]",
    URL: "[URL]",
    STUDENT_ID: "[MSSV]",
    NATIONAL_ID: "[CCCD]",
}
NAME_TOKEN_PREFIX = "wzjwz"
MAX_NAME_WORDS = 4

# --- Regex (đầu số di động 03/05/07/08/09, 10 số từ 15/9/2018) ---------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?i)\b(?:https?://|www\.|(?:facebook|fb|zalo|tiktok)\.[a-z]+/)\S+")
_PHONE_MOBILE = re.compile(
    r"(?<![\d.])\(?(?:\+?84|0084|0)\)?[\s.\-]?[35789](?:[\s.\-]?\d){8}(?!\d)"
)
_PHONE_LANDLINE = re.compile(r"(?<![\d.])0[\s.\-]?2\d{1,2}(?:[\s.\-]?\d){7,8}(?!\d)")
# 3 số tỉnh + 1 số thế kỷ/giới + 2 số năm sinh + 6 số ngẫu nhiên.
_CCCD = re.compile(r"(?<![\d.])0\d{2}[0-3]\d{8}(?!\d)")
_NATIONAL_ID_CONTEXT = re.compile(
    r"(?i)\b(?:cccd|cmnd|cmt|căn\s*cước(?:\s*công\s*dân)?|chứng\s*minh(?:\s*nhân\s*dân)?)\b"
    r"\W{0,3}\d{9,12}(?!\d)"
)
_STUDENT_ID_CONTEXT = re.compile(
    r"(?i)\b(?:mssv|msv|mã\s*số\s*sinh\s*viên|mã\s*sv)\b\W{0,3}[a-z]{0,3}\d{5,11}[a-z0-9]*"
)
# Chuỗi 8-12 chữ số đứng riêng; lookbehind loại "2.500.000", "8.5", "12/10/2023", "it001".
_LONG_ID = re.compile(r"(?<![\w.,/])\d{8,12}(?!\w)")

# Thứ tự = độ ưu tiên khi hai mẫu trùng vị trí.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (URL, _URL),
    (EMAIL, _EMAIL),
    (STUDENT_ID, _STUDENT_ID_CONTEXT),
    (NATIONAL_ID, _NATIONAL_ID_CONTEXT),
    (NATIONAL_ID, _CCCD),
    (PHONE, _PHONE_MOBILE),
    (PHONE, _PHONE_LANDLINE),
    (STUDENT_ID, _LONG_ID),
)

# --- Luật tên: từ xưng hô + âm tiết Viết Hoa ---------------------------------------------------
_CUE = re.compile(
    r"(?<!\w)(?:(?:giảng|giáo)\s+viên|thầy|cô|anh|chị|bạn|em|ông|bà|(?:gv|ths|ts|pgs|gs)\.?)\s+",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]+")
_BLANKS = re.compile(r"[ \t]+")
# Từ Viết Hoa hay đứng sau từ xưng hô nhưng không phải tên: "Thầy Cô", "Cô Giáo", "Thầy Hiệu Trưởng"
_NOT_NAMES = frozenset(
    "thầy cô anh chị bạn em ông bà gv ths ts pgs gs giảng giáo viên sinh chủ nhiệm hiệu trưởng "
    "phó khoa bộ môn tổ trợ sư tiến sĩ thạc ơi ạ".split()
)
# Chữ in hoa toàn câu thì chữ hoa không còn báo hiệu tên.
_ALL_CAPS_RATIO = 0.8


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    type: str


@dataclass(frozen=True, slots=True)
class PiiResult:
    display: str  # giữ hoa/thường, PII thành nhãn [TÊN]...
    model: str  # tên -> wzjwz<k>, PII khác bị xoá
    types: tuple[str, ...]  # loại PII đã gặp, theo thứ tự xuất hiện, không lặp


def _overlaps(span: Span, taken: Iterable[Span]) -> bool:
    return any(span.start < other.end and other.start < span.end for other in taken)


def _regex_spans(text: str) -> list[Span]:
    accepted: list[Span] = []
    for pii_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            span = Span(match.start(), match.end(), pii_type)
            if not _overlaps(span, accepted):
                accepted.append(span)
    return accepted


def _is_mostly_upper(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return (
        len(letters) >= 3 and sum(ch.isupper() for ch in letters) / len(letters) > _ALL_CAPS_RATIO
    )


def _cue_name_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for cue in _CUE.finditer(text):
        position = end = cue.end()
        count = 0
        while count < MAX_NAME_WORDS:
            word = _WORD.match(text, position)
            if word is None or not word.group()[0].isupper():
                break
            if word.group().casefold() in _NOT_NAMES:
                break
            end = word.end()
            count += 1
            separator = _BLANKS.match(text, end)
            if separator is None:
                break
            position = separator.end()
        if count:
            spans.append(Span(cue.end(), end, NAME))
    return spans


def _ner_name_spans(text: str) -> list[Span]:
    """Tên người do underthesea NER nhận ra trong văn bản còn nguyên chữ hoa."""
    try:
        from underthesea import ner
    except ImportError:  # NER là tuỳ chọn
        return []
    spans: list[Span] = []
    cursor = 0
    for word, _pos, _chunk, label in ner(text):
        if not label.endswith("PER"):
            continue
        entity = word.replace("_", " ")
        start = text.find(entity, cursor)
        if start < 0:
            continue
        cursor = start + len(entity)
        spans.append(Span(start, cursor, NAME))
    return spans


def find_spans(text: str, use_ner: bool = False) -> list[Span]:
    """Các đoạn PII không chồng nhau, sắp theo vị trí. `text` nên đã qua `clean_text`."""
    spans = _regex_spans(text)
    if not _is_mostly_upper(text):
        for candidate in _cue_name_spans(text):
            if not _overlaps(candidate, spans):
                spans.append(candidate)
        if use_ner and any(ch.isupper() for ch in text):
            for candidate in _ner_name_spans(text):
                if not _overlaps(candidate, spans):
                    spans.append(candidate)
    return sorted(spans, key=lambda s: s.start)


def mask(text: str, use_ner: bool = False) -> PiiResult:
    """Ẩn PII theo hai góc nhìn. Mọi khoảng trắng thừa trong kết quả được gộp lại."""
    display: list[str] = []
    model: list[str] = []
    types: list[str] = []
    name_ids: dict[str, int] = {}
    position = 0
    for span in find_spans(text, use_ner):
        before = text[position : span.start]
        display.append(before)
        model.append(before)
        display.append(PLACEHOLDERS[span.type])
        if span.type == NAME:
            name_id = name_ids.setdefault(text[span.start : span.end].casefold(), len(name_ids) + 1)
            model.append(f" {NAME_TOKEN_PREFIX}{name_id} ")
        else:
            model.append(" ")
        if span.type not in types:
            types.append(span.type)
        position = span.end
    display.append(text[position:])
    model.append(text[position:])
    return PiiResult(
        display=re.sub(r"\s+", " ", "".join(display)).strip(),
        model=re.sub(r"\s+", " ", "".join(model)).strip(),
        types=tuple(types),
    )
