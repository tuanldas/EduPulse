"""N-gram (đơn và đôi) trên văn bản đã tách từ; MỘT hàm dùng chung cho đếm nền và insight.

Đếm nền (lúc đóng gói bundle) và đếm trên dữ liệu người dùng (lúc chạy) phải cắt token đúng như
nhau, nếu không prior log-odds sẽ lệch. Vì vậy cả hai gọi `ngrams`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from edupulse.text.pii import NAME_TOKEN_PREFIX


def _is_punctuation(token: str) -> bool:
    return not any(ch.isalnum() for ch in token)


def ngrams(text_rdr: str) -> list[str]:
    """Unigram rồi bigram của câu đã tách từ ("giảng_viên" giữ nguyên dấu `_`).

    - Token chỉ gồm dấu câu bị bỏ, và bigram không bắc qua dấu câu (không có "hay phòng" từ
      "hay . phòng").
    - Token tên người (`wzjwz<k>`) bị bỏ: chúng không phải từ khoá và định danh một người.
    - Chỉ đếm các n-gram cùng xuất hiện trong một đoạn liền, giữ nguyên thứ tự và số lần lặp.
    """
    unigrams: list[str] = []
    bigrams: list[str] = []
    previous: str | None = None
    for token in text_rdr.split():
        if _is_punctuation(token) or token.startswith(NAME_TOKEN_PREFIX):
            previous = None
            continue
        unigrams.append(token)
        if previous is not None:
            bigrams.append(f"{previous} {token}")
        previous = token
    return unigrams + bigrams


def count_ngrams(texts: Iterable[str]) -> Counter[str]:
    """Số lần xuất hiện của mỗi n-gram trên nhiều câu đã tách từ."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(ngrams(text))
    return counts
