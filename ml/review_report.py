"""Hiệu quả của ngân sách duyệt (5/10/20/30%) trên test, dùng calibration ĐÃ ĐÓNG BĂNG từ val.

    python -m ml.calibrate --report test [--rehearsal]        # make report

Không fit lại gì trên test: nhiệt độ T, bảng phân vị `cdf` và ngưỡng τ_B lấy từ `calibration.json`
của bản triển khai (tính trên val). Với mỗi ngân sách, bảng báo hai luật dùng đúng mã của engine
(`edupulse.scoring`):

- xếp hạng cả lô: cờ ⌈B·n/100⌉ câu điểm thấp nhất (luật của bảng val và của phân tích hàng loạt);
- ngưỡng từng câu: cờ khi điểm < τ_B (luật của phân tích một câu), cho thấy ngưỡng val tổng quát
  hoá sang test ra sao (tỉ lệ cờ thực có lệch ngân sách không).
Kèm ECE và NLL của từng head trên test trước và sau T. Chỉ đọc `outputs/test_preds/`.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from edupulse.bundle_schema import Calibration
from edupulse.config import BUDGETS, DEFAULT_BUDGET
from edupulse.labels import TASKS
from edupulse.scoring import (
    confidence,
    efficiency,
    flag_score,
    flags_by_threshold,
    quantile_rank,
    rank_batch,
    softmax_t,
)
from ml.common import (
    expected_calibration_error,
    negative_log_likelihood,
    settings,
    write_json,
)
from ml.eval_tables import TASK_VI
from ml.plotting import vn
from ml.preds import DEPLOY_KEY, EvalContext, context, is_done, load_preds, read_meta

EXIT_NOT_DONE = 2


def load_calibration(deploy_run: str) -> Calibration:
    """Calibration đã đóng băng của run được triển khai (schema `Calibration`)."""
    path = settings().outputs_dir / "calibration" / deploy_run / "calibration.json"
    return Calibration.model_validate_json(path.read_text(encoding="utf-8"))


def flag_scores(
    logits: dict[str, np.ndarray], calibration: Calibration
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Điểm cờ của từng câu (min của hai phân vị) và xác suất từng head, theo T và cdf đóng băng."""
    heads = {"sentiment": calibration.sentiment, "topic": calibration.topic}
    ranks: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    for task in TASKS:
        probs[task] = softmax_t(logits[task].astype(np.float64), heads[task].temperature)
        ranks[task] = quantile_rank(confidence(probs[task]), heads[task].cdf)
    return flag_score(ranks["sentiment"], ranks["topic"]), probs


def budget_rows(
    scores: np.ndarray, ok: dict[str, np.ndarray], tau: dict[int, float]
) -> dict[int, dict[str, Any]]:
    """Với mỗi ngân sách: hiệu quả của luật xếp hạng cả lô và của luật ngưỡng τ_B."""
    rows: dict[int, dict[str, Any]] = {}
    for budget in BUDGETS:
        batch = efficiency(rank_batch(scores, budget), ok["sentiment"], ok["topic"])
        by_threshold = efficiency(
            flags_by_threshold(scores, tau[budget]), ok["sentiment"], ok["topic"]
        )
        rows[budget] = {
            "rank_batch": batch,
            "threshold": {"tau": tau[budget], **by_threshold},
        }
    return rows


def calibration_quality(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict[str, float]:
    before = softmax_t(logits.astype(np.float64), 1.0)
    after = softmax_t(logits.astype(np.float64), temperature)
    return {
        "ece_before_t": expected_calibration_error(before, y),
        "ece_after_t": expected_calibration_error(after, y),
        "nll_before_t": negative_log_likelihood(before, y),
        "nll_after_t": negative_log_likelihood(after, y),
    }


def build_report(ctx: EvalContext, calibration: Calibration, deploy_run: str) -> dict[str, Any]:
    preds = load_preds(ctx.preds_dir, DEPLOY_KEY)
    scores, probs = flag_scores(preds.logits, calibration)
    ok = {task: probs[task].argmax(axis=1) == preds.y[task] for task in TASKS}
    heads = {"sentiment": calibration.sentiment, "topic": calibration.topic}
    return {
        "split": ctx.split,
        "rehearsal": ctx.rehearsal,
        "n": len(scores),
        "deploy_run": deploy_run,
        "default_budget": DEFAULT_BUDGET,
        "temperature": {task: heads[task].temperature for task in TASKS},
        "budgets": budget_rows(scores, ok, calibration.tau),
        "calibration_quality": {
            task: calibration_quality(preds.logits[task], preds.y[task], heads[task].temperature)
            for task in TASKS
        },
    }


def _percent(value: float) -> str:
    return vn(value * 100, 1)


def table_val(calibration: Calibration) -> str:
    lines = [
        f"Hiệu quả ngân sách duyệt trên validation (n = {vn(calibration.n, 0)}), luật xếp hạng cả "
        "lô, tính bằng đúng mã của engine. Cột 'độ chính xác khi không cờ' là mốc so sánh.",
        "",
        "| Ngân sách | Ngưỡng τ | Tỉ lệ cờ | Lỗi cảm xúc bị bắt | Lỗi chủ đề bị bắt "
        "| Chính xác phần không cờ (đúng cả hai) | Chính xác khi không cờ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        e = calibration.efficiency[budget]
        mark = " (mặc định)" if budget == DEFAULT_BUDGET else ""
        lines.append(
            f"| {budget}%{mark} | {vn(calibration.tau[budget], 4)} | {_percent(e.flagged_share)}% "
            f"| {_percent(e.caught_sentiment_errors)}% | {_percent(e.caught_topic_errors)}% "
            f"| {_percent(e.accept_accuracy_both)}% | {_percent(e.all_accuracy_both)}% |"
        )
    return "\n".join(lines) + "\n"


def table_test(report: dict[str, Any]) -> str:
    where = "test" if report["split"] == "test" else "validation (diễn tập, KHÔNG phải test)"
    lines = [
        f"Hiệu quả ngân sách duyệt trên tập {where} (n = {vn(report['n'], 0)}) với calibration "
        "đóng băng từ validation (T, phân vị độ tin cậy, ngưỡng τ). Không fit lại gì trên test.",
        "",
        "### Luật xếp hạng cả lô (cờ ⌈B·n/100⌉ câu điểm thấp nhất)",
        "",
        "| Ngân sách | Tỉ lệ cờ | Lỗi cảm xúc bị bắt | Lỗi chủ đề bị bắt "
        "| Chính xác phần không cờ (đúng cả hai) | Chính xác khi không cờ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        e = report["budgets"][budget]["rank_batch"]
        mark = " (mặc định)" if budget == report["default_budget"] else ""
        lines.append(
            f"| {budget}%{mark} | {_percent(e['flagged_share'])}% "
            f"| {_percent(e['caught_sentiment_errors'])}% | {_percent(e['caught_topic_errors'])}% "
            f"| {_percent(e['accept_accuracy_both'])}% | {_percent(e['all_accuracy_both'])}% |"
        )
    lines += [
        "",
        "### Luật ngưỡng từng câu (cờ khi điểm < τ, τ lấy từ validation)",
        "",
        "| Ngân sách | Ngưỡng τ | Tỉ lệ cờ thực | Lỗi cảm xúc bị bắt | Lỗi chủ đề bị bắt "
        "| Chính xác phần không cờ (đúng cả hai) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        e = report["budgets"][budget]["threshold"]
        lines.append(
            f"| {budget}% | {vn(e['tau'], 4)} | {_percent(e['flagged_share'])}% "
            f"| {_percent(e['caught_sentiment_errors'])}% | {_percent(e['caught_topic_errors'])}% "
            f"| {_percent(e['accept_accuracy_both'])}% |"
        )
    lines += [
        "",
        "### Hiệu chuẩn xác suất trên cùng tập",
        "",
        "| Tác vụ | T | ECE trước T | ECE sau T | NLL trước T | NLL sau T |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        q = report["calibration_quality"][task]
        temperature = vn(report["temperature"][task], 3)
        lines.append(
            f"| {TASK_VI[task]} | {temperature} | {vn(q['ece_before_t'], 4)} "
            f"| {vn(q['ece_after_t'], 4)} | {vn(q['nll_before_t'], 4)} "
            f"| {vn(q['nll_after_t'], 4)} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(ctx: EvalContext, report: dict[str, Any], calibration: Calibration) -> list[str]:
    paths = [
        ctx.metrics_path("review_budget_test.json"),
        ctx.table_path("review_budget_test.md"),
        ctx.table_path("review_budget_val.md"),
    ]
    write_json(paths[0], report)
    paths[1].parent.mkdir(parents=True, exist_ok=True)
    paths[1].write_text(table_test(report), encoding="utf-8")
    paths[2].write_text(table_val(calibration), encoding="utf-8")
    return [str(p) for p in paths]


def main(rehearsal: bool = False) -> int:
    ctx = context(rehearsal)
    if not is_done(ctx.preds_dir):
        print(f"calibrate: chưa chấm xong ({ctx.preds_dir} chưa có .done)", file=sys.stderr)
        return EXIT_NOT_DONE
    deploy_run = read_meta(ctx.preds_dir)["deploy_run"]
    calibration = load_calibration(deploy_run)
    report = build_report(ctx, calibration, deploy_run)
    for path in write_outputs(ctx, report, calibration):
        print(f"-> {path}")
    print(table_test(report))
    return 0
