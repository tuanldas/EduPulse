"""Cờ "cần xem xét" và lý do bằng lời, dựng trên `edupulse.scoring` (cùng luật với lúc hiệu chỉnh).

- Một câu (`/api/analyze`): cờ khi điểm < `tau` của ngân sách mặc định 10%.
- Cả lô: cờ ⌈B·n/100⌉ câu điểm thấp nhất (`scoring.rank_batch`).
- Câu gõ không dấu luôn được cờ (mô hình kém hẳn), và tính vào ngân sách trong một lô.
- Câu quá dài để giải thích (> XAI_MAX_WORDS từ) KHÔNG vì thế mà bị cờ.
"""

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
