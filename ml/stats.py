"""Chỉ số phân loại từ ma trận nhầm lẫn, khoảng tin cậy bootstrap và kiểm định ghép cặp.

Chỉ dùng numpy (và scipy cho McNemar). Macro/weighted-F1 tính từ đếm nhầm lẫn theo công thức của
scikit-learn (`labels = 0..K-1`, `zero_division = 0`), nhưng vector hoá cho cả nghìn lần lấy mẫu:
2.000 vòng x 3.166 câu chạy trong vài giây. Mọi bootstrap dùng RNG có hạt giống cố định nên kết quả
tái lập từng chữ số; bootstrap ghép cặp dùng CÙNG chỉ số câu cho hai mô hình.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95


def confusion_counts(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Ma trận đếm [thật, dự đoán] kích thước K x K."""
    joint = np.asarray(y, dtype=np.int64) * n_classes + np.asarray(pred, dtype=np.int64)
    return np.bincount(joint, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_f1(counts: np.ndarray) -> np.ndarray:
    """F1 từng lớp từ đếm nhầm lẫn (cuối trục: [..., K, K] -> [..., K]); mẫu số 0 cho F1 = 0."""
    tp = np.diagonal(counts, axis1=-2, axis2=-1).astype(np.float64)
    support = counts.sum(axis=-1)
    predicted = counts.sum(axis=-2)
    denominator = support + predicted  # = 2tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    return per_class_f1(counts).mean(axis=-1)


def weighted_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    support = counts.sum(axis=-1).astype(np.float64)
    total = support.sum(axis=-1)
    return (per_class_f1(counts) * support).sum(axis=-1) / np.where(total > 0, total, 1.0)


def accuracy_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=(-2, -1))
    correct = np.diagonal(counts, axis1=-2, axis2=-1).sum(axis=-1)
    return correct / np.where(total > 0, total, 1)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(macro_f1_from_counts(confusion_counts(y, pred, n_classes)))


def bootstrap_indices(
    n: int, rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Chỉ số câu chọn lại có hoàn lại, hình [rounds, n]; cùng tham số cho cùng kết quả."""
    return np.random.default_rng(seed).integers(0, n, size=(rounds, n))


def resampled_counts(
    y: np.ndarray, pred: np.ndarray, n_classes: int, indices: np.ndarray
) -> np.ndarray:
    """Đếm nhầm lẫn của từng lần lấy mẫu, hình [rounds, K, K], bằng một lần bincount."""
    rounds = indices.shape[0]
    joint = (
        np.asarray(y, dtype=np.int64)[indices] * n_classes
        + np.asarray(pred, dtype=np.int64)[indices]
    )
    joint += (np.arange(rounds, dtype=np.int64) * n_classes * n_classes)[:, None]
    flat = np.bincount(joint.ravel(), minlength=rounds * n_classes * n_classes)
    return flat.reshape(rounds, n_classes, n_classes)


def metric_samples(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Chỉ số từng lần lấy mẫu: accuracy, macro_f1, weighted_f1 [R] và f1 từng lớp [R, K]."""
    return {
        "accuracy": accuracy_from_counts(counts),
        "macro_f1": macro_f1_from_counts(counts),
        "weighted_f1": weighted_f1_from_counts(counts),
        "per_class_f1": per_class_f1(counts),
    }


def percentile_interval(
    samples: np.ndarray, level: float = CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Khoảng phân vị đối xứng (vd. 2,5% và 97,5% cho mức 95%) của một dãy mẫu bootstrap."""
    tail = round((1.0 - level) / 2 * 100, 9)  # làm tròn để 0,95 cho đúng 2,5 chứ không 2,500…02
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_intervals(
    y: np.ndarray,
    pred: np.ndarray,
    n_classes: int,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Khoảng tin cậy 95% theo phân vị cho accuracy, macro-F1, weighted-F1 và F1 từng lớp."""
    indices = bootstrap_indices(len(y), rounds, seed)
    samples = metric_samples(resampled_counts(y, pred, n_classes, indices))
    result: dict[str, Any] = {"rounds": rounds, "seed": seed, "level": CONFIDENCE_LEVEL}
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        low, high = percentile_interval(samples[name])
        result[name] = {"low": low, "high": high}
    result["per_class_f1"] = [
        dict(zip(("low", "high"), percentile_interval(samples["per_class_f1"][:, k]), strict=True))
        for k in range(n_classes)
    ]
    return result


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """McNemar chính xác hai phía: b = A đúng B sai, c = A sai B đúng; p = binomtest(b, b+c)."""
    a, b_ = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    only_a, only_b = int((a & ~b_).sum()), int((~a & b_).sum())
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else float(binomtest(only_a, discordant, 0.5).pvalue)
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": p_value}


def paired_difference(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_classes: int,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A so với B trên cùng các câu: McNemar (accuracy) và bootstrap ghép cặp của hiệu macro-F1.

    `delta_macro_f1` = macroF1(A) - macroF1(B) trên toàn tập; khoảng 95% theo phân vị của hiệu qua
    các lần chọn lại CÙNG chỉ số câu cho hai mô hình; `bootstrap_p` hai phía =
    min(1, 2·min(tỉ lệ hiệu <= 0, tỉ lệ hiệu >= 0)).
    """
    counts_a = confusion_counts(y, pred_a, n_classes)
    counts_b = confusion_counts(y, pred_b, n_classes)
    indices = bootstrap_indices(len(y), rounds, seed)
    samples_a = macro_f1_from_counts(resampled_counts(y, pred_a, n_classes, indices))
    samples_b = macro_f1_from_counts(resampled_counts(y, pred_b, n_classes, indices))
    delta_samples = samples_a - samples_b
    low, high = percentile_interval(delta_samples)
    bootstrap_p = min(
        1.0, 2 * min(float((delta_samples <= 0).mean()), float((delta_samples >= 0).mean()))
    )
    mcnemar = mcnemar_exact(
        np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    )
    return {
        "n": len(y),
        "accuracy_a": float(accuracy_from_counts(counts_a)),
        "accuracy_b": float(accuracy_from_counts(counts_b)),
        "macro_f1_a": float(macro_f1_from_counts(counts_a)),
        "macro_f1_b": float(macro_f1_from_counts(counts_b)),
        "delta_macro_f1": float(macro_f1_from_counts(counts_a) - macro_f1_from_counts(counts_b)),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_p": bootstrap_p,
        "mcnemar_p": mcnemar["p_value"],
        "a_only_correct": mcnemar["a_only_correct"],
        "b_only_correct": mcnemar["b_only_correct"],
        "rounds": rounds,
        "seed": seed,
    }
