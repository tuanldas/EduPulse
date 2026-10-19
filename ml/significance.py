"""Kiểm định ghép cặp trên dự đoán đã lưu: bản triển khai so với baseline và so với bản FP32.

    python -m ml.significance [--rehearsal]        # make report

Mỗi cặp (A, B) cho từng tác vụ: McNemar chính xác trên số câu hai mô hình bất đồng (kiểm
accuracy) và bootstrap ghép cặp của hiệu macro-F1 (2.000 vòng, cùng chỉ số câu cho hai mô hình,
hạt giống cố định; xem `ml.stats`). Chỉ đọc `outputs/test_preds/`, không suy luận lại mô hình.
Đầu ra: `metrics/significance.json` và `tables/significance.md`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from edupulse.labels import LABELS, TASKS
from ml import stats
from ml.common import write_json
from ml.eval_tables import TASK_VI, model_display
from ml.plotting import vn
from ml.preds import (
    BASELINES,
    DEPLOY_KEY,
    EvalContext,
    Preds,
    context,
    is_done,
    load_preds,
    read_meta,
)

EXIT_NOT_DONE = 2
ALPHA = 0.05
GROUPS = {
    "baseline": "Bản triển khai (int8) so với baseline TF-IDF",
    "fp32": "Bản triển khai (int8) so với PhoBERT FP32 cùng run",
}


def pairs(deploy_run: str) -> list[tuple[str, str, str]]:
    """(nhóm, khoá A, khoá B): A luôn là bản int8 đang triển khai."""
    return [
        *(("baseline", DEPLOY_KEY, name) for name in BASELINES),
        ("fp32", DEPLOY_KEY, deploy_run),
    ]


def compare(group: str, a: Preds, b: Preds, task: str) -> dict[str, Any]:
    """Một bản ghi so sánh (A so với B) cho một tác vụ."""
    if (a.y[task] != b.y[task]).any():
        raise ValueError(f"{a.key} và {b.key} không cùng nhãn thật ở {task}: lệch thứ tự dòng")
    result = stats.paired_difference(a.y[task], a.pred[task], b.pred[task], len(LABELS[task]))
    significant_accuracy = result["mcnemar_p"] < ALPHA
    return {
        "group": group,
        "task": task,
        "a": a.key,
        "b": b.key,
        **result,
        "significant_accuracy": significant_accuracy,
        "significant_macro_f1": result["ci95_low"] > 0 or result["ci95_high"] < 0,
    }


def run(ctx: EvalContext) -> dict[str, Any]:
    meta = read_meta(ctx.preds_dir)
    records: list[dict[str, Any]] = []
    cache: dict[str, Preds] = {}

    def preds_of(key: str) -> Preds:
        if key not in cache:
            cache[key] = load_preds(ctx.preds_dir, key)
        return cache[key]

    for group, key_a, key_b in pairs(meta["deploy_run"]):
        for task in TASKS:
            records.append(compare(group, preds_of(key_a), preds_of(key_b), task))
    return {
        "split": ctx.split,
        "rehearsal": ctx.rehearsal,
        "deploy_run": meta["deploy_run"],
        "n": meta["n"],
        "alpha": ALPHA,
        "bootstrap": {"rounds": stats.BOOTSTRAP_ROUNDS, "seed": stats.BOOTSTRAP_SEED},
        "records": records,
    }


def p_text(p_value: float) -> str:
    """p-value kiểu Việt; giá trị rất nhỏ ghi `< 0,0001` thay vì `0,0000` (bootstrap 2.000 vòng)."""
    return "< 0,0001" if p_value < 1e-4 else vn(p_value, 4)


def verdict(record: dict[str, Any]) -> str:
    def word(flag: bool) -> str:
        return "có" if flag else "không"

    return (
        f"acc: {word(record['significant_accuracy'])} · mF1: {word(record['significant_macro_f1'])}"
    )


def markdown(result: dict[str, Any]) -> str:
    where = "test" if result["split"] == "test" else "validation (diễn tập, KHÔNG phải test)"
    rounds, seed = vn(result["bootstrap"]["rounds"], 0), result["bootstrap"]["seed"]
    lines = [
        f"Kiểm định ghép cặp trên tập {where} (n = {vn(result['n'], 0)}). McNemar chính xác kiểm "
        "accuracy trên các câu hai mô hình bất đồng; ΔmF1 là hiệu macro-F1 (A − B, điểm phần "
        f"trăm) với khoảng tin cậy 95% theo bootstrap ghép cặp ({rounds} vòng, hạt giống {seed}, "
        "cùng chỉ số câu cho hai mô hình); 'có ý nghĩa' nghĩa là "
        f"p < {vn(result['alpha'], 2)} (McNemar) hoặc khoảng tin cậy không chứa 0 (bootstrap)."
    ]
    for group, title in GROUPS.items():
        records = [r for r in result["records"] if r["group"] == group]
        if not records:
            continue
        lines += [
            "",
            f"### {title}",
            "",
            "| Tác vụ | So sánh (A so với B) | Acc A | Acc B | A đúng, B sai | A sai, B đúng "
            "| p McNemar | ΔmF1 (điểm) | KTC 95% | p bootstrap | Có ý nghĩa |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in records:
            lines.append(
                f"| {TASK_VI[r['task']]} | {model_display(r['a'])} so với {model_display(r['b'])} "
                f"| {vn(r['accuracy_a'] * 100, 1)} | {vn(r['accuracy_b'] * 100, 1)} "
                f"| {r['a_only_correct']} | {r['b_only_correct']} | {p_text(r['mcnemar_p'])} "
                f"| {vn(r['delta_macro_f1'] * 100, 2)} "
                f"| [{vn(r['ci95_low'] * 100, 2)}; {vn(r['ci95_high'] * 100, 2)}] "
                f"| {p_text(r['bootstrap_p'])} | {verdict(r)} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(ctx: EvalContext, result: dict[str, Any]) -> list[str]:
    metrics = ctx.metrics_path("significance.json")
    table = ctx.table_path("significance.md")
    write_json(metrics, result)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(markdown(result), encoding="utf-8")
    return [str(metrics), str(table)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rehearsal", action="store_true", help="chạy trên dự đoán diễn tập")
    args = parser.parse_args(argv)
    ctx = context(args.rehearsal)
    if not is_done(ctx.preds_dir):
        print(f"significance: chưa chấm xong ({ctx.preds_dir} chưa có .done)", file=sys.stderr)
        return EXIT_NOT_DONE
    result = run(ctx)
    for path in write_outputs(ctx, result):
        print(f"-> {path}")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
