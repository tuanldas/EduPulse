"""Chuẩn hoá văn bản thô về dạng của UIT-VSFC trước khi tách từ; dùng chung cho train và serve.

Dữ liệu gốc đã là chữ thường, NFC, dấu câu tách bằng dấu cách, tên người thay bằng `wzjwz<N>`,
emoticon và vài ký hiệu thay bằng token chữ (`colonsmile`, `doubledot`, `vdotv`, `3dot0`...).
Văn bản người dùng gõ thì không như vậy, nên các hàm dưới đây đưa nó về cùng dạng.

Thứ tự trong `normalize_text` là hợp đồng, đừng đảo:
chữ thường -> dấu câu chuẩn -> dấu thanh -> token đặc biệt -> gộp ký tự lặp -> tách dấu câu.
- Emoticon phải đổi TRƯỚC khi tách từ, vì RDRSegmenter tách ":)" thành ":" và ")" rồi cắt câu.
- Dấu câu phải được tách bằng dấu cách TRƯỚC khi tách từ như trong dữ liệu gốc: nếu dính liền,
  RDRSegmenter cắt nhầm token chữ-số ("wzjwz155." thành "wzjwz 155 .", "9dot5." thành "9 dot 5 .").
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from itertools import groupby

# --- Làm sạch (áp cho cả bản hiển thị) -------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1b\x7f-\x9f]")
_FORMAT = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """NFC, bỏ ký tự điều khiển và ký tự vô hình, gộp mọi khoảng trắng thành một dấu cách."""
    text = unicodedata.normalize("NFC", text)
    text = _FORMAT.sub("", _CONTROL.sub(" ", text))
    return _SPACES.sub(" ", text).strip()


# --- Dấu thanh --------------------------------------------------------------------------------

# Bảng 15 cặp VinAI: vocab PhoBERT-v2 chỉ có kiểu mới (hoà, khoá, thuỷ...), không có kiểu cũ.
TONE_MAP: dict[str, str] = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}  # fmt: skip


def normalize_tone(text: str) -> str:
    """Đưa dấu thanh kiểu cũ (hòa, khóa, thủy) về kiểu mới (hoà, khoá, thuỷ). Nhận chữ thường."""
    for old, new in TONE_MAP.items():
        text = text.replace(old, new)
    return text


# --- Dấu câu ----------------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "″": '"',
        "‘": "'", "’": "'", "‚": "'", "′": "'", "´": "'", "`": "'",
        "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
        "！": "!", "？": "?", "，": ",", "：": ":", "；": ";",
        "（": "(", "）": ")", "．": ".",
    }
)  # fmt: skip


def normalize_punctuation(text: str) -> str:
    """Dấu nháy, gạch ngang và dấu toàn chiều rộng về ký tự ASCII như trong dữ liệu."""
    return text.translate(_PUNCT_TABLE)


# --- Token đặc biệt ---------------------------------------------------------------------------

# Emoticon: token ở cột phải là token có trong dữ liệu UIT-VSFC. Mỗi mẫu thay bằng " token ".
_EMOTICONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), f" {token} ")
    for pattern, token in (
        (r"[:=;]-?\){2,}", "colonsmilesmile"),
        (r"[:=;]-?\)", "colonsmile"),
        (r"[:=]-?\(+(?!\w)", "colonsad"),
        (r"(?<!\w)[:=]-?d(?!\w)", "colonbigsmile"),
        (r"(?<![\w<])<3+(?!\d)", "colonlove"),
        (r"(?<!\w):-?\*(?!\w)", "colonlove"),
        (r"(?<!\w):x(?!\w)", "colonlove"),
        (r"(?<!\w):v(?!\w)", "coloncontemn"),
        (r"(?<!\w):o(?!\w)", "colonsurprise"),
        (r"(?<!\w):3(?!\w)", "colonhihi"),
        (r"(?<!\w):>(?!\w)", "colonsmallsmile"),
    )
)

# Emoji và mọi ký tự ngoài mặt phẳng cơ bản (pyjnius xử lý cặp thay thế UTF-16 không tin cậy).
_EMOJI = re.compile("[\U00010000-\U0010ffff☀-➿⬀-⯿️‍]")
_VDOTV = re.compile(r"\bv\.v\b")
_ELLIPSIS = re.compile(r"\.{2,}|…+")
_TIME_COLON = re.compile(r"(?<=\d):(?=\d)")

# Dấu chấm dính liền hai bên được dữ liệu đổi thành "dot": số thập phân (3.0 -> 3dot0), tên miền
# hoặc mã lớp (uit.edu.vn -> uitdotedudotvn, it003.e21 -> it003dote21), ".net" -> "dotnet".
_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_DOMAIN = re.compile(r"(?<![\w.])[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+(?!\w)(?!\.\w)")
_LEADING_TLD = re.compile(r"(?<![\w.])\.(?=(?:net|com|vn|org|edu|io|info)(?!\w))")
_TLDS = frozenset(
    {"com", "net", "org", "edu", "gov", "vn", "info", "io", "biz", "app", "dev", "me"}
)
# Dấu chấm giữa hai chữ cái còn lại là lỗi gõ thiếu dấu cách ("tốt.em"): RDR không tách được.
_GLUED_DOT = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")


def _domain_to_dot(match: re.Match[str]) -> str:
    chunk = match.group(0)
    if chunk.rsplit(".", 1)[-1] in _TLDS or any(ch.isdigit() for ch in chunk):
        return chunk.replace(".", "dot")
    return chunk


def map_special_tokens(text: str) -> str:
    """Đổi emoticon, ký hiệu và dấu chấm/hai chấm đặc biệt thành token chữ như dữ liệu gốc.

    Nhận chữ thường. Thứ tự các bước quan trọng: emoticon trước ":" và "." để ":)" không bị ăn.
    """
    for pattern, token in _EMOTICONS:
        text = pattern.sub(token, text)
    text = _EMOJI.sub(" ", text)
    text = _VDOTV.sub(" vdotv ", text)
    text = _ELLIPSIS.sub(" dotdotdot ", text)
    text = _TIME_COLON.sub("doubledot", text)
    text = text.replace(":", " doubledot ")
    text = _DECIMAL_DOT.sub("dot", text)
    text = _DOMAIN.sub(_domain_to_dot, text)
    text = _LEADING_TLD.sub("dot", text)
    text = _GLUED_DOT.sub(" . ", text)
    return _SPACES.sub(" ", text).strip()


# --- Gộp ký tự lặp ----------------------------------------------------------------------------

_REPEATED_PUNCT = re.compile(r"([!?,])\1+")


@lru_cache(maxsize=4096)
def _base_letter(char: str) -> str | None:
    """Chữ cái gốc không dấu ("á" -> "a"); None nếu không phải chữ cái."""
    return unicodedata.normalize("NFD", char)[0] if char.isalpha() else None


def collapse_repeats(text: str) -> str:
    """Chữ cái lặp từ 3 lần trở lên thành 1 ("quáaaa" -> "quá"); "!!!" -> "!". Số giữ nguyên.

    Các chữ cùng gốc nhưng khác dấu cũng tính là lặp: "quáaaa" là "á" rồi ba chữ "a".
    """
    parts: list[str] = []
    for base, group in groupby(text, key=_base_letter):
        run = "".join(group)
        parts.append(run[0] if base is not None and len(run) >= 3 else run)
    return _REPEATED_PUNCT.sub(r"\1", "".join(parts))


# --- Tách dấu câu -----------------------------------------------------------------------------

# Dữ liệu gốc không có dấu câu nào dính liền chữ (trừ "50%", "64-bit", "he's").
_SPACED_PUNCT = re.compile(r'[.,!?;()"]')


def space_punctuation(text: str) -> str:
    """Đặt dấu cách quanh `. , ! ? ; ( ) "` như dữ liệu gốc; gộp khoảng trắng thừa."""
    return _SPACES.sub(" ", _SPACED_PUNCT.sub(r" \g<0> ", text)).strip()


# --- Gom lại ----------------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Chuỗi đã qua `clean_text` và ẩn PII -> chuỗi sẵn sàng cho bộ tách từ."""
    text = normalize_punctuation(text.lower())
    text = normalize_tone(text)
    text = map_special_tokens(text)
    return space_punctuation(collapse_repeats(text))


# --- Nhận diện chữ không dấu ------------------------------------------------------------------

_MARKED_LETTERS = frozenset("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
_ALNUM_RUN = re.compile(r"[^\W_]+")

DEFAULT_NO_DIACRITICS_RATIO = 0.15
MIN_SYLLABLES_FOR_DIACRITICS = 3


def diacritic_ratio(text: str) -> tuple[int, int]:
    """(số âm tiết có dấu, tổng số âm tiết) trong các cụm chữ cái thuần; bỏ `wzjwz<k>`, mã, số."""
    marked = total = 0
    for token in _ALNUM_RUN.findall(text.lower()):
        if not token.isalpha():
            continue
        total += 1
        if not _MARKED_LETTERS.isdisjoint(token):
            marked += 1
    return marked, total


def is_no_diacritics(
    text: str,
    threshold: float = DEFAULT_NO_DIACRITICS_RATIO,
    min_syllables: int = MIN_SYLLABLES_FOR_DIACRITICS,
) -> bool:
    """Đúng khi tỉ lệ âm tiết có dấu thanh/mũ dưới ngưỡng. Cần đủ âm tiết mới kết luận."""
    marked, total = diacritic_ratio(text)
    return total >= min_syllables and marked / total < threshold
