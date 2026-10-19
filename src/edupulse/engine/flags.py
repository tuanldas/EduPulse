"""Cờ "cần xem xét" và lý do bằng lời, dựng trên `edupulse.scoring` (cùng luật với lúc hiệu chỉnh).

- Một câu (`/api/analyze`): cờ khi điểm < `tau` của ngân sách mặc định 10%.
- Cả lô: cờ ⌈B·n/100⌉ câu điểm thấp nhất (`scoring.rank_batch`).
- Câu gõ không dấu luôn được cờ (mô hình kém hẳn), và tính vào ngân sách trong một lô.
- Câu quá dài để giải thích (> XAI_MAX_WORDS từ) KHÔNG vì thế mà bị cờ.
- Cả lô còn có mẫu kiểm tra: một phần nhỏ câu KHÔNG cờ được chọn ngẫu nhiên (cố định theo seed của
  batch) để người duyệt đo độ chính xác không thiên lệch. Hàng đợi duyệt = cờ + mẫu kiểm tra,
  sắp độ tin cậy tăng dần.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from edupulse.labels import LABELS, LABELS_VI
from edupulse.scoring import flags_by_threshold, rank_batch

LOW_CONFIDENCE = "low_confidence"
NO_DIACRITICS = "no_diacritics"

TASK_VI = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}
TOP_CANDIDATES = 2


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class FlagReason:
    code: str
    task: str | None = None  # head không chắc, chỉ có với LOW_CONFIDENCE
    candidates: tuple[Candidate, ...] = ()  # các lớp cao nhất của head đó

    @property
    def message(self) -> str:
        """Lý do bằng tiếng Việt, dùng cho giao diện và cột "Lý do" của file xuất."""
        if self.code == NO_DIACRITICS:
            return "Câu có vẻ gõ không dấu nên mô hình kém chính xác, cần người xem xét"
        names = " và ".join(
            f"{LABELS_VI[self.task][c.label]} {round(c.probability * 100)}%"
            for c in self.candidates
        )
        return f"{TASK_VI[self.task]} không chắc — phân vân giữa {names}"


@dataclass(frozen=True, slots=True)
class Flag:
    needs_review: bool
    reasons: tuple[FlagReason, ...] = ()


NOT_FLAGGED = Flag(False)


def weakest_task(q_sentiment: float, q_topic: float) -> str:
    """Head có độ tin cậy thấp hơn (phân vị nhỏ hơn); bằng nhau thì lấy cảm xúc."""
    return "sentiment" if q_sentiment <= q_topic else "topic"


def top_candidates(
    task: str, probs: Sequence[float], k: int = TOP_CANDIDATES
) -> tuple[Candidate, ...]:
    ranked = sorted(zip(LABELS[task], probs, strict=True), key=lambda pair: -pair[1])
    return tuple(Candidate(label, float(p)) for label, p in ranked[:k])


def single_needs_review(score: float, no_diacritics: bool, tau: float) -> bool:
    return no_diacritics or bool(flags_by_threshold(score, tau))


def batch_needs_review(
    scores: ArrayLike, no_diacritics: ArrayLike, budget_pct: float
) -> NDArray[np.bool_]:
    return rank_batch(scores, budget_pct, forced=no_diacritics)


def build_flag(*, flagged: bool, no_diacritics: bool, task: str, probs: Sequence[float]) -> Flag:
    """Gắn lý do cho kết quả đã quyết định: gõ không dấu là lý do riêng, còn lại là tin cậy thấp."""
    if not flagged:
        return NOT_FLAGGED
    if no_diacritics:
        return Flag(True, (FlagReason(NO_DIACRITICS),))
    return Flag(True, (FlagReason(LOW_CONFIDENCE, task, top_candidates(task, probs)),))


def seed_from(text: str) -> int:
    """Seed ổn định (không phụ thuộc PYTHONHASHSEED) tính từ một chuỗi, vd. id của batch."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def audit_sample(flagged: ArrayLike, rate: float, seed: int) -> NDArray[np.bool_]:
    """Mẫu kiểm tra: ⌈rate · số câu không cờ⌉ câu không cờ, chọn ngẫu nhiên theo `seed`.

    Câu đã cờ không vào mẫu (chúng đã nằm trong hàng đợi duyệt). Cùng `flagged` và `seed` luôn cho
    cùng mẫu. Có câu không cờ thì có ít nhất một câu kiểm tra.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"tỉ lệ mẫu phải trong (0, 1], nhận {rate!r}")
    mask = np.asarray(flagged, dtype=bool)
    candidates = np.flatnonzero(~mask)
    chosen = np.zeros(mask.shape, dtype=bool)
    if candidates.size == 0:
        return chosen
    count = min(candidates.size, math.ceil(round(rate * candidates.size, 9)))
    picked = np.random.default_rng(seed).choice(candidates, size=count, replace=False)
    chosen[picked] = True
    return chosen


def review_order(scores: ArrayLike, in_review: ArrayLike) -> list[int]:
    """Chỉ số các câu trong hàng đợi duyệt, điểm tăng dần (ít chắc chắn trước); bằng điểm thì theo
    thứ tự dòng, nên thứ tự cố định giữa các lần gọi."""
    values = np.asarray(scores, dtype=np.float64)
    members = np.flatnonzero(np.asarray(in_review, dtype=bool))
    order = np.lexsort((members, values[members]))
    return [int(i) for i in members[order]]
