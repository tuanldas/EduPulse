"""Chỉ số, bảng và hình của kết quả đánh giá, tính từ dự đoán đã lưu (`ml.preds`).

Không đọc dữ liệu hay mô hình, nên chạy lại bao nhiêu lần cũng không chạm test: `evaluate --final`
chỉ chấm một lần rồi ghi dự đoán, còn mọi bảng và hình sinh lại từ đó.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from edupulse.labels import LABELS, LABELS_VI, TASKS
from edupulse.scoring import softmax_t
from ml.common import (
    expected_calibration_error,
    negative_log_likelihood,
    prediction_metrics,
    reliability_bins,
)
from ml.plotting import ACCENT, COLORS, GRID, INK, MUTED, plt, save_figure_to, setup_style, vn
from ml.preds import BASELINES, DEPLOY_KEY, Preds, arch_of, available_keys, load_preds, seed_of
from ml.train_baseline import DISPLAY as BASELINE_DISPLAY

TASK_VI = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}
ARCH_DISPLAY = {
    "stl": "PhoBERT đơn nhiệm (STL)",
    "mtl": "PhoBERT đa nhiệm (MTL)",
    DEPLOY_KEY: "PhoBERT đa nhiệm · int8 (bản triển khai)",
}
SCALAR_METRICS = ("accuracy", "macro_f1", "weighted_f1", "ece", "nll")
RELIABILITY_KEYS = (DEPLOY_KEY, "linearsvc")  # chỉ hai mô hình này cần biểu đồ độ tin cậy


def seed_stat(values: list[float]) -> dict[str, Any]:
    """Trung bình và độ lệch chuẩn mẫu (ddof = 1; 0 khi chỉ có một giá trị)."""
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "n": len(array), "values": array.tolist()}


def preds_metrics(
    preds: Preds, task: str, temperature: float = 1.0, *, reliability: bool = False
) -> dict[str, Any]:
    """Chỉ số của một mô hình trên một tác vụ: nhãn cứng cho F1/CM, xác suất cho ECE/NLL."""
    y = preds.y[task]
    probs = softmax_t(preds.logits[task].astype(np.float64), temperature)
    metrics = prediction_metrics(y, preds.pred[task], LABELS[task])
    metrics["ece"] = expected_calibration_error(probs, y)
    metrics["nll"] = negative_log_likelihood(probs, y)
    metrics["n"] = len(y)
    if reliability:
        metrics["reliability"] = reliability_bins(probs, y)
    return metrics


def seeded_models(models: dict[str, Any], arch: str) -> list[str]:
    keys = [key for key in models if arch_of(key) == arch]
    return sorted(keys, key=lambda key: seed_of(key) or 0)


def architecture_stats(models: dict[str, Any]) -> dict[str, Any]:
    """Trung bình ± độ lệch chuẩn theo seed của MTL (cả hai tác vụ) và STL (mỗi tác vụ một họ)."""
    sources = {
        "mtl": {task: seeded_models(models, "mtl") for task in TASKS},
        "stl": {task: seeded_models(models, f"stl-{task}") for task in TASKS},
    }
    result: dict[str, Any] = {}
    for arch, by_task in sources.items():
        for task, keys in by_task.items():
            if not keys:
                continue
            per_model = [models[key][task] for key in keys]
            entry: dict[str, Any] = {"seeds": [seed_of(key) for key in keys]}
            for metric in SCALAR_METRICS:
                entry[metric] = seed_stat([m[metric] for m in per_model])
            entry["per_class_f1"] = [
                seed_stat([m["per_class"][i]["f1"] for m in per_model])
                for i in range(len(LABELS[task]))
            ]
            result.setdefault(arch, {})[task] = entry
    return result


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
            )
            for task in preds.tasks
        }
        n = len(next(iter(preds.y.values())))
    return {
        "n": n,
        "temperatures": temperatures,
        "models": models,
        "architectures": architecture_stats(models),
    }


# --- bảng chính -------------------------------------------------------------------------------


def _single(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_seeds": 1,
        **{m: {"mean": metrics[m], "std": None} for m in SCALAR_METRICS},
        "per_class_f1": [{"mean": c["f1"], "std": None} for c in metrics["per_class"]],
    }


def _aggregated(entry: dict[str, Any]) -> dict[str, Any]:
    n = entry["accuracy"]["n"]

    def keep(stat: dict[str, Any]) -> dict[str, Any]:
        return {"mean": stat["mean"], "std": stat["std"] if n > 1 else None}

    return {
        "n_seeds": n,
        **{m: keep(entry[m]) for m in SCALAR_METRICS},
        "per_class_f1": [keep(stat) for stat in entry["per_class_f1"]],
    }


def model_display(key: str) -> str:
    """Tên hiển thị của một khoá mô hình (dùng cho tiêu đề hình)."""
    if key in BASELINES:
        return BASELINE_DISPLAY[key]
    if key == DEPLOY_KEY:
        return ARCH_DISPLAY[DEPLOY_KEY]
    arch = arch_of(key)
    return f"{ARCH_DISPLAY['mtl' if arch == 'mtl' else 'stl']} · seed {seed_of(key)}"


def main_rows(collected: dict[str, Any]) -> list[dict[str, Any]]:
    """Mỗi dòng một (tác vụ, mô hình): baseline, STL và MTL (mean ± std theo seed), bản int8."""
    models, arch = collected["models"], collected["architectures"]
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for name in BASELINES:
            if name in models:
                row = _single(models[name][task])
                rows.append({"task": task, "model": name, "display": BASELINE_DISPLAY[name], **row})
        for group in ("stl", "mtl"):
            if task in arch.get(group, {}):
                row = _aggregated(arch[group][task])
                rows.append({"task": task, "model": group, "display": ARCH_DISPLAY[group], **row})
        if DEPLOY_KEY in models:
            row = _single(models[DEPLOY_KEY][task])
            rows.append(
                {"task": task, "model": DEPLOY_KEY, "display": ARCH_DISPLAY[DEPLOY_KEY], **row}
            )
    return rows


def _cell(stat: dict[str, Any], digits: int = 1, scale: float = 100.0) -> str:
    text = vn(stat["mean"] * scale, digits)
    if stat["std"] is not None:
        text += f" ± {vn(stat['std'] * scale, digits)}"
    return text


def main_markdown(rows: list[dict[str, Any]], *, n: int, split_label: str, note: str = "") -> str:
    lines = [
        f"Kết quả trên tập {split_label} (n = {vn(n, 0)}); phần trăm, ± là độ lệch chuẩn theo seed."
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
            "| Mô hình | Số run | Acc | mF1 | wF1 | "
            + " | ".join(f"F1 {x}" for x in labels)
            + " | ECE |",
            "|---|---:|---:|---:|---:|" + "---:|" * len(labels) + "---:|",
        ]
        for row in task_rows:
            f1s = " | ".join(_cell(stat) for stat in row["per_class_f1"])
            lines.append(
                f"| {row['display']} | {row['n_seeds']} | {_cell(row['accuracy'])} "
                f"| {_cell(row['macro_f1'])} | {_cell(row['weighted_f1'])} | {f1s} "
                f"| {_cell(row['ece'], 3, 1.0)} |"
            )
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def write_main_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task", "model", "display", "n_seeds"]
    for metric in SCALAR_METRICS:
        columns += [f"{metric}_mean", f"{metric}_std"]
    label_columns = [
        f"f1_{label}_{kind}"
        for label in (*LABELS["sentiment"], *LABELS["topic"])
        for kind in ("mean", "std")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*columns, *label_columns], restval="")
        writer.writeheader()
        for row in rows:
            record = {k: row[k] for k in ("task", "model", "display", "n_seeds")}
            for metric in SCALAR_METRICS:
                record[f"{metric}_mean"] = row[metric]["mean"]
                record[f"{metric}_std"] = "" if row[metric]["std"] is None else row[metric]["std"]
            for label, stat in zip(LABELS[row["task"]], row["per_class_f1"], strict=True):
                record[f"f1_{label}_mean"] = stat["mean"]
                record[f"f1_{label}_std"] = "" if stat["std"] is None else stat["std"]
            writer.writerow(record)


def write_per_class_csv(models: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task", "label", "precision", "recall", "f1", "support"])
        for key, by_task in models.items():
            for task, metrics in by_task.items():
                for cls in metrics["per_class"]:
                    writer.writerow(
                        [
                            key,
                            task,
                            cls["label"],
                            cls["precision"],
                            cls["recall"],
                            cls["f1"],
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
    for colour, (name, item) in zip(colours, series.items(), strict=False):
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
        offset = (list(series).index(name) - (len(series) - 1) / 2) * width
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
    return save_figure_to(fig, path)
