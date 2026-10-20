"""Chỉ số, bảng và hình của kết quả đánh giá, tính từ dự đoán đã lưu (`ml.preds`).

Không đọc dữ liệu hay mô hình, nên chạy lại bao nhiêu lần cũng không chạm test: `evaluate --final`
chỉ chấm một lần rồi ghi dự đoán, còn mọi bảng và hình sinh lại từ đó. Mỗi chỉ số chính kèm khoảng
tin cậy 95% theo bootstrap (2.000 vòng, hạt giống cố định, xem `ml.stats`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from edupulse.labels import LABELS, LABELS_VI, TASKS
from edupulse.scoring import softmax_t
from ml import stats
from ml.common import (
    expected_calibration_error,
    negative_log_likelihood,
    prediction_metrics,
    reliability_bins,
)
from ml.plotting import (
    ACCENT,
    COLORS,
    GRID,
    INK,
    MUTED,
    plt,
    save_figure_to,
    setup_style,
    vietnamese_axes,
    vn,
)
from ml.preds import (
    BASELINES,
    DEPLOY_KEY,
    Preds,
    available_keys,
    check_key,
    key_order,
    load_preds,
    seed_of,
)
from ml.train_baseline import DISPLAY as BASELINE_DISPLAY

TASK_VI = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}
DEPLOY_DISPLAY = "PhoBERT đa nhiệm · int8 (bản triển khai)"
SCALARS = ("accuracy", "macro_f1", "weighted_f1")
RELIABILITY_KEYS = (DEPLOY_KEY, "linearsvc")  # chỉ hai mô hình này cần biểu đồ độ tin cậy


def model_display(key: str) -> str:
    """Tên hiển thị của một khoá mô hình."""
    if key in BASELINES:
        return BASELINE_DISPLAY[key]
    if key == DEPLOY_KEY:
        return DEPLOY_DISPLAY
    return f"PhoBERT đa nhiệm · FP32 (seed {seed_of(check_key(key))})"


def preds_metrics(
    preds: Preds,
    task: str,
    temperature: float = 1.0,
    *,
    reliability: bool = False,
    intervals: bool = False,
) -> dict[str, Any]:
    """Chỉ số của một mô hình trên một tác vụ: nhãn cứng cho F1/CM, xác suất cho ECE/NLL.

    `intervals` thêm khoá `ci95` (bootstrap 95% của accuracy, macro-F1, weighted-F1, F1 từng lớp).
    """
    y = preds.y[task]
    probs = softmax_t(preds.logits[task].astype(np.float64), temperature)
    metrics = prediction_metrics(y, preds.pred[task], LABELS[task])
    metrics["ece"] = expected_calibration_error(probs, y)
    metrics["nll"] = negative_log_likelihood(probs, y)
    metrics["n"] = len(y)
    if reliability:
        metrics["reliability"] = reliability_bins(probs, y)
    if intervals:
        metrics["ci95"] = stats.bootstrap_intervals(y, preds.pred[task], len(LABELS[task]))
    return metrics


def collect_metrics(
    directory: Path, temperatures: dict[str, float], *, keys: list[str] | None = None
) -> dict[str, Any]:
    """Chỉ số mọi mô hình có dự đoán trong `directory`; bản int8 dùng nhiệt độ đã đóng băng."""
    models: dict[str, Any] = {}
    n = 0
    for key in keys or available_keys(directory):
        preds = load_preds(directory, key)
        models[key] = {
            task: preds_metrics(
                preds,
                task,
                temperatures[task] if key == DEPLOY_KEY else 1.0,
                reliability=key in RELIABILITY_KEYS,
                intervals=True,
            )
            for task in preds.tasks
        }
        n = len(next(iter(preds.y.values())))
    return {
        "n": n,
        "temperatures": temperatures,
        "bootstrap": {
            "rounds": stats.BOOTSTRAP_ROUNDS,
            "seed": stats.BOOTSTRAP_SEED,
            "level": stats.CONFIDENCE_LEVEL,
        },
        "models": models,
    }


# --- bảng chính -------------------------------------------------------------------------------


def _with_interval(value: float, interval: dict[str, float]) -> dict[str, float]:
    return {"value": value, "low": interval["low"], "high": interval["high"]}


def main_rows(collected: dict[str, Any]) -> list[dict[str, Any]]:
    """Mỗi dòng một (tác vụ, mô hình), theo thứ tự chuẩn; chỉ số chính kèm khoảng tin cậy 95%."""
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for key in sorted(collected["models"], key=key_order):
            metrics = collected["models"][key].get(task)
            if metrics is None:
                continue
            ci = metrics["ci95"]
            row: dict[str, Any] = {
                "task": task,
                "model": key,
                "display": model_display(key),
                "n": metrics["n"],
                **{m: _with_interval(metrics[m], ci[m]) for m in SCALARS},
                "ece": metrics["ece"],
                "nll": metrics["nll"],
                "per_class_f1": [
                    _with_interval(cls["f1"], interval)
                    for cls, interval in zip(metrics["per_class"], ci["per_class_f1"], strict=True)
                ],
            }
            rows.append(row)
    return rows


def _cell(stat: dict[str, float], *, interval: bool, digits: int = 1) -> str:
    text = vn(stat["value"] * 100, digits)
    if interval:
        text += f" [{vn(stat['low'] * 100, digits)}–{vn(stat['high'] * 100, digits)}]"
    return text


def main_markdown(rows: list[dict[str, Any]], *, n: int, split_label: str, note: str = "") -> str:
    lines = [
        f"Kết quả trên tập {split_label} (n = {vn(n, 0)}); phần trăm. Trong ngoặc vuông là khoảng "
        "tin cậy 95% theo bootstrap (2.000 vòng chọn lại câu, hạt giống cố định)."
    ]
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        labels = [LABELS_VI[task][label] for label in LABELS[task]]
        lines += [
            "",
            f"### {TASK_VI[task]}",
            "",
            "| Mô hình | Acc | mF1 | wF1 | " + " | ".join(f"F1 {x}" for x in labels) + " | ECE |",
            "|---|---:|---:|---:|" + "---:|" * len(labels) + "---:|",
        ]
        for row in task_rows:
            f1s = " | ".join(_cell(stat, interval=False) for stat in row["per_class_f1"])
            lines.append(
                f"| {row['display']} | {_cell(row['accuracy'], interval=True)} "
                f"| {_cell(row['macro_f1'], interval=True)} "
                f"| {_cell(row['weighted_f1'], interval=True)} | {f1s} | {vn(row['ece'], 3)} |"
            )
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def write_main_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Bảng chính dạng dài: mỗi (tác vụ, mô hình) một dòng, chỉ số kèm cận dưới/cận trên 95%."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task", "model", "display", "n"]
    for metric in SCALARS:
        columns += [metric, f"{metric}_low", f"{metric}_high"]
    columns += ["ece", "nll"]
    labels = [*LABELS["sentiment"], *LABELS["topic"]]
    for label in labels:
        columns += [f"f1_{label}", f"f1_{label}_low", f"f1_{label}_high"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            record: dict[str, Any] = {k: row[k] for k in ("task", "model", "display", "n")}
            for metric in SCALARS:
                stat = row[metric]
                record |= {metric: stat["value"], f"{metric}_low": stat["low"]}
                record[f"{metric}_high"] = stat["high"]
            record |= {"ece": row["ece"], "nll": row["nll"]}
            for label, stat in zip(LABELS[row["task"]], row["per_class_f1"], strict=True):
                record |= {f"f1_{label}": stat["value"], f"f1_{label}_low": stat["low"]}
                record[f"f1_{label}_high"] = stat["high"]
            writer.writerow(record)


def write_per_class_csv(models: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "task", "label", "precision", "recall", "f1", "f1_low", "f1_high", "support"]
        )
        for key in sorted(models, key=key_order):
            for task, metrics in models[key].items():
                intervals = metrics["ci95"]["per_class_f1"]
                for cls, interval in zip(metrics["per_class"], intervals, strict=True):
                    writer.writerow(
                        [
                            key,
                            task,
                            cls["label"],
                            cls["precision"],
                            cls["recall"],
                            cls["f1"],
                            interval["low"],
                            interval["high"],
                            cls["support"],
                        ]
                    )


# --- hình -------------------------------------------------------------------------------------


def plot_confusion(path: Path, matrix: list[list[float]], task: str, title: str) -> Path:
    setup_style()
    labels = [LABELS_VI[task][label] for label in LABELS[task]]
    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(array, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if array[i, j] > 0.55 else INK
            ax.text(
                j, i, vn(array[i, j] * 100, 1), ha="center", va="center", color=colour, fontsize=9
            )
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save_figure_to(fig, path)


def plot_reliability(path: Path, series: dict[str, dict[str, Any]], task: str, title: str) -> Path:
    """Độ tin cậy trung bình so với tỉ lệ đúng theo bin, kèm số câu mỗi bin ở khung dưới."""
    setup_style()
    fig, (ax, counts) = plt.subplots(
        2,
        1,
        figsize=(4.8, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    colours = [ACCENT, COLORS["others"], COLORS["training_program"]]
    for index, (colour, (name, item)) in enumerate(zip(colours, series.items(), strict=False)):
        bins = [b for b in item["reliability"] if b["n"]]
        xs, ys = [b["confidence"] for b in bins], [b["accuracy"] for b in bins]
        ax.plot(xs, ys, color=colour, linewidth=1, alpha=0.6)
        # Chấm to theo căn bậc hai số câu trong bin: bin ít câu (nhiễu) hiện nhỏ.
        ax.scatter(
            xs,
            ys,
            s=[6 + 2.2 * b["n"] ** 0.5 for b in bins],
            color=colour,
            label=f"{name} (ECE {vn(item['ece'], 3)})",
        )
        centres = [(b["lower"] + b["upper"]) / 2 for b in item["reliability"]]
        width = 0.9 / len(item["reliability"]) / len(series)
        offset = (index - (len(series) - 1) / 2) * width
        counts.bar(
            [c + offset for c in centres],
            [b["n"] for b in item["reliability"]],
            width=width,
            color=colour,
            alpha=0.85,
        )
    ax.set_ylabel("Tỉ lệ đúng")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{title} · {TASK_VI[task]}", loc="left", fontsize=10, fontweight="bold")
    counts.set_yscale("log")
    counts.set_ylabel("Số câu", color=MUTED)
    counts.set_xlabel("Độ tin cậy (xác suất lớn nhất)")
    counts.set_xlim(0, 1)
    vietnamese_axes(ax, "y")
    vietnamese_axes(counts, "x")  # trục x dùng chung với khung trên; trục y của khung dưới là log
    return save_figure_to(fig, path)
