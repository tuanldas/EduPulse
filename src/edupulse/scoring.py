"""Độ tin cậy, điểm cờ, ngân sách duyệt: một bản cho `ml/calibrate.py` và `engine/flags.py`.

Chỉ dùng numpy (runtime không có scipy/torch). Luật cờ:

1. Mỗi head: xác suất `softmax(logits / T)`, độ tin cậy = xác suất lớn nhất.
2. `q` = phân vị của độ tin cậy trong phân phối trên tập val (bảng 101 điểm `cdf`).
3. Điểm cờ của một câu = `min(q_cảm_xúc, q_chủ_đề)`: thấp nghĩa là ít chắc chắn.
4. Ngân sách B%: `tau_B` sao cho khoảng B% câu val có điểm dưới `tau_B`.
   - Một câu: cờ nếu điểm < `tau_B` (`flags_by_threshold`).
   - Cả lô n câu: cờ ⌈B·n/100⌉ câu điểm thấp nhất (`rank_batch`); câu bắt buộc (gõ không dấu) được
     cờ trước và tính vào ngân sách.
Bảng "hiệu quả ước tính" tính bằng đúng `rank_batch` trên val, nên khớp luật chạy thật.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

CDF_POINTS = 101  # phân vị 0%, 1%, ..., 100%


def softmax_t(logits: ArrayLike, temperature: float = 1.0) -> NDArray[np.float64]:
    """Softmax theo hàng cuối với nhiệt độ T (T > 1 làm xác suất bớt tự tin)."""
    if not temperature > 0:
        raise ValueError(f"temperature phải > 0, nhận {temperature!r}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def confidence(probs: ArrayLike) -> NDArray[np.float64]:
    """Độ tin cậy = xác suất của lớp được dự đoán."""
    return np.asarray(probs, dtype=np.float64).max(axis=-1)


def make_cdf(confidences: ArrayLike, points: int = CDF_POINTS) -> list[float]:
    """Bảng độ tin cậy tại các phân vị 0..100% của một tập (thường là val)."""
    values = np.asarray(confidences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cần ít nhất một độ tin cậy")
    return [float(x) for x in np.quantile(values, np.linspace(0.0, 1.0, points))]


def quantile_rank(conf: ArrayLike, cdf: Sequence[float]) -> NDArray[np.float64]:
    """Phân vị (0..1) của độ tin cậy trong bảng `cdf`, nội suy tuyến tính.

    Các điểm trùng giá trị (đuôi bão hoà gần 1,0) lấy phân vị trung bình, để kết quả đơn điệu và
    không phụ thuộc cách `numpy.interp` chọn giữa các điểm trùng.
    """
    table = np.asarray(cdf, dtype=np.float64)
    levels = np.linspace(0.0, 1.0, len(table))
    unique, inverse = np.unique(table, return_inverse=True)
    mean_level = np.bincount(inverse, weights=levels) / np.bincount(inverse)
    return np.interp(np.asarray(conf, dtype=np.float64), unique, mean_level)


def flag_score(q_sentiment: ArrayLike, q_topic: ArrayLike) -> NDArray[np.float64]:
    """Điểm cờ: phân vị thấp hơn trong hai head."""
    return np.minimum(
        np.asarray(q_sentiment, dtype=np.float64), np.asarray(q_topic, dtype=np.float64)
    )


def _budget_count(pct: float, n: int) -> int:
    """⌈pct·n/100⌉; làm tròn 9 chữ số để sai số dấu phẩy động không đẩy lên một câu."""
    if not 0 < pct <= 100:
        raise ValueError(f"ngân sách phải trong (0, 100], nhận {pct!r}")
    return math.ceil(round(pct * n / 100, 9))


def threshold_for_budget(val_scores: ArrayLike, pct: float) -> float:
    """`tau` sao cho khoảng pct% câu val có điểm dưới `tau` (trung điểm hai điểm kề nhau)."""
    scores = np.sort(np.asarray(val_scores, dtype=np.float64))
    if scores.size == 0:
        raise ValueError("cần ít nhất một điểm val")
    k = _budget_count(pct, scores.size)
    if k >= scores.size:
        return math.inf
    return float((scores[k - 1] + scores[k]) / 2)


def flags_by_threshold(scores: ArrayLike, tau: float) -> NDArray[np.bool_]:
    """Luật cho từng câu độc lập: cờ khi điểm < tau."""
    return np.asarray(scores, dtype=np.float64) < tau


def rank_batch(scores: ArrayLike, pct: float, forced: ArrayLike | None = None) -> NDArray[np.bool_]:
    """Cờ ⌈pct·n/100⌉ câu điểm thấp nhất của một lô.

    `forced` (vd. câu gõ không dấu) luôn được cờ và tính vào ngân sách trước; vượt ngân sách thì
    vẫn cờ hết. Điểm bằng nhau xử lý theo thứ tự dòng, nên cùng lô luôn cho cùng kết quả.
    """
    values = np.asarray(scores, dtype=np.float64)
    must = (
        np.zeros(values.shape, dtype=bool)
        if forced is None
        else np.asarray(forced, dtype=bool).copy()
    )
    if must.shape != values.shape:
        raise ValueError("forced phải cùng kích thước với scores")
    remaining = _budget_count(pct, values.size) - int(must.sum())
    if remaining > 0:
        candidates = np.flatnonzero(~must)
        order = candidates[np.argsort(values[candidates], kind="stable")]
        must[order[:remaining]] = True
    return must


def efficiency(flags: ArrayLike, ok_sentiment: ArrayLike, ok_topic: ArrayLike) -> dict[str, float]:
    """Hiệu quả của một tập cờ so với nhãn vàng (dùng trên val để ước tính).

    - `flagged_share`: tỉ lệ câu bị cờ;
    - `caught_sentiment_errors` / `caught_topic_errors`: tỉ lệ lỗi của từng head nằm trong số câu cờ
      (0 nếu không có lỗi nào);
    - `accept_accuracy_both`: tỉ lệ câu đúng cả hai head trong phần KHÔNG cờ (1 nếu cờ hết);
    - `all_accuracy_both`: như trên khi không cờ gì, để so sánh.
    """
    flagged = np.asarray(flags, dtype=bool)
    ok_s = np.asarray(ok_sentiment, dtype=bool)
    ok_t = np.asarray(ok_topic, dtype=bool)
    if not (flagged.shape == ok_s.shape == ok_t.shape) or flagged.ndim != 1 or flagged.size == 0:
        raise ValueError(
            "flags, ok_sentiment, ok_topic phải là mảng 1 chiều cùng kích thước, không rỗng"
        )
    both = ok_s & ok_t

    def caught(ok: NDArray[np.bool_]) -> float:
        errors = ~ok
        return float((flagged & errors).sum() / errors.sum()) if errors.any() else 0.0

    accepted = ~flagged
    return {
        "flagged_share": float(flagged.mean()),
        "caught_sentiment_errors": caught(ok_s),
        "caught_topic_errors": caught(ok_t),
        "accept_accuracy_both": float(both[accepted].mean()) if accepted.any() else 1.0,
        "all_accuracy_both": float(both.mean()),
    }
