"""Hiệu chuẩn xác suất và ngưỡng cờ cho mô hình đã xuất, trên validation.

    python -m ml.calibrate                 # make calibrate (run 'best')

Đầu vào là logits val của CHÍNH bản int8 sẽ triển khai (`outputs/onnx/<run>/val_logits_int8.npz`),
vì lượng tử hoá đổi nhẹ logits; logits chạy từng câu một (batch 1) vì int8 động phụ thuộc lô, nên
ngưỡng là tham chiếu tất định cho phân tích một câu. Với mỗi head:
1. Nhiệt độ T tối ưu NLL trên toàn val. Chỉ dùng T nếu kiểm chéo (200 lần chia đôi val, fit T ở
   nửa A, đo ở nửa B) làm NLL giảm mà ECE không tệ đi; không thì giữ T = 1 (mô hình đa nhiệm
   thường đã gần hiệu chuẩn).
2. Bảng 101 phân vị của độ tin cậy (xác suất lớn nhất sau T) để đổi độ tin cậy mới thành phân vị.
Rồi điểm cờ = min(q_cảm_xúc, q_chủ_đề), ngưỡng cho từng ngân sách duyệt và bảng hiệu quả ước tính;
tất cả tính bằng `edupulse.scoring`, cùng mã mà API dùng khi chạy thật.
Đầu ra: `outputs/calibration/<run>/calibration.json` (schema `Calibration`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from edupulse.bundle_schema import BudgetEfficiency, Calibration, HeadCalibration
from edupulse.config import BUDGETS
from edupulse.labels import TASKS
from edupulse.scoring import (
    confidence,
    efficiency,
    flag_score,
    make_cdf,
    quantile_rank,
    rank_batch,
    softmax_t,
    threshold_for_budget,
)
from ml.common import (
    expected_calibration_error,
    negative_log_likelihood,
    resolve_run,
    settings,
    write_json,
)

CROSS_FIT_ROUNDS = 200
TEMPERATURE_BOUNDS = (0.05, 20.0)
SEED = 0


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T > 0 làm NLL nhỏ nhất (Guo et al. 2017)."""
    result = minimize_scalar(
        lambda t: negative_log_likelihood(softmax_t(logits, t), y),
        bounds=TEMPERATURE_BOUNDS,
        method="bounded",
    )
    return float(result.x)


def cross_fit(
    logits: np.ndarray, y: np.ndarray, rounds: int = CROSS_FIT_ROUNDS
) -> dict[str, float]:
    """Trung bình NLL/ECE ở nửa B trước và sau khi fit T ở nửa A, qua nhiều lần chia đôi."""
    rng = np.random.default_rng(SEED)
    before = {"nll": [], "ece": []}
    after = {"nll": [], "ece": []}
    for _ in range(rounds):
        order = rng.permutation(len(y))
        half = len(y) // 2
        a, b = order[:half], order[half:]
        t = fit_temperature(logits[a], y[a])
        for store, temperature in ((before, 1.0), (after, t)):
            probs = softmax_t(logits[b], temperature)
            store["nll"].append(negative_log_likelihood(probs, y[b]))
            store["ece"].append(expected_calibration_error(probs, y[b]))
    return {
        "nll_before": float(np.mean(before["nll"])),
        "nll_after": float(np.mean(after["nll"])),
        "ece_before": float(np.mean(before["ece"])),
        "ece_after": float(np.mean(after["ece"])),
    }


def choose_temperature(logits: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, Any]]:
    fitted = fit_temperature(logits, y)
    check = cross_fit(logits, y)
    useful = check["nll_after"] < check["nll_before"] and check["ece_after"] <= check["ece_before"]
    return (fitted if useful else 1.0), {"fitted": fitted, "used_fitted": useful, **check}


def calibrate(
    val_logits: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> tuple[Calibration, dict]:
    heads: dict[str, HeadCalibration] = {}
    ranks: dict[str, np.ndarray] = {}
    correct: dict[str, np.ndarray] = {}
    decisions: dict[str, Any] = {}
    for task in TASKS:
        logits = val_logits[task].astype(np.float64)
        y = targets[task]
        temperature, decision = choose_temperature(logits, y)
        probs = softmax_t(logits, temperature)
        cdf = make_cdf(confidence(probs))
        heads[task] = HeadCalibration(
            temperature=temperature,
            cdf=cdf,
            ece_before=expected_calibration_error(softmax_t(logits, 1.0), y),
            ece_after=expected_calibration_error(probs, y),
        )
        ranks[task] = quantile_rank(confidence(probs), cdf)
        correct[task] = probs.argmax(axis=1) == y
        decisions[task] = decision

    scores = flag_score(ranks["sentiment"], ranks["topic"])
    tau: dict[int, float] = {}
    table: dict[int, BudgetEfficiency] = {}
    for budget in BUDGETS:
        tau[budget] = threshold_for_budget(scores, budget)
        flags = rank_batch(scores, budget)
        table[budget] = BudgetEfficiency(
            **efficiency(flags, correct["sentiment"], correct["topic"])
        )
    calibration = Calibration(
        n=len(scores), sentiment=heads["sentiment"], topic=heads["topic"], tau=tau, efficiency=table
    )
    return calibration, decisions


def output_path(run: Path) -> Path:
    return settings().outputs_dir / "calibration" / run.name / "calibration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="best", help="'best', tên run hoặc thư mục checkpoint")
    parser.add_argument(
        "--report",
        choices=["test"],
        help="thay vì fit trên val: báo cáo hiệu quả ngân sách duyệt trên test bằng calibration "
        "đã đóng băng (ml.review_report); cần đã chấm test",
    )
    parser.add_argument(
        "--rehearsal", action="store_true", help="với --report: chạy trên dự đoán diễn tập (val)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        from ml.review_report import main as report_main

        return report_main(rehearsal=args.rehearsal)
    run = resolve_run(args.run)
    logits_path = settings().outputs_dir / "onnx" / run.name / "val_logits_int8.npz"
    if not logits_path.is_file():
        print(f"chưa có {logits_path} — chạy `make export` trước", file=sys.stderr)
        return 2
    data = np.load(logits_path)
    val_logits = {task: data[task] for task in TASKS}
    targets = {task: data[f"y_{task}"] for task in TASKS}
    calibration, decisions = calibrate(val_logits, targets)
    target = output_path(run)
    write_json(target, calibration.model_dump(mode="json"))
    print(json.dumps({"run": run.name, "temperature_decisions": decisions}, ensure_ascii=False))
    for budget in BUDGETS:
        e = calibration.efficiency[budget]
        print(
            f"ngân sách {budget:>2}%: tau={calibration.tau[budget]:.3f} · lỗi cảm xúc bắt "
            f"{e.caught_sentiment_errors:.0%} · lỗi chủ đề bắt {e.caught_topic_errors:.0%} · "
            f"phần không cờ đúng cả 2 {e.accept_accuracy_both:.1%}"
        )
    print(f"-> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
