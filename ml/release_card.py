"""Thẻ mô hình của bản phát hành: số liệu trên TEST đọc từ kết quả đã chấm một lần, không tính lại.

Đầu vào (dưới `results/`): `metrics/test_all.json` (chỉ số mọi mô hình trên test, do
`ml.evaluate --final` ghi), `metrics/review_budget_test.json` (hiệu quả ngân sách duyệt trên test),
`metrics/checklist.json` (CheckList dựng từ câu validation) và `metrics/onnx_gate_v1.json` (số
validation của bản int8, để nêu khoảng cách validation - test). Thẻ chỉ được tạo khi test_all.json
do đúng lần chấm cuối của run này sinh ra: không phải diễn tập, cùng run, cùng `freeze.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edupulse.bundle_schema import (
    ArchitectureRow,
    BaselineRow,
    BudgetEfficiency,
    BudgetRow,
    ChecklistRow,
    ModelCard,
    ModelCardNotes,
    SeedStat,
    SplitMetrics,
    TaskMetrics,
)
from edupulse.config import BUDGETS, DEFAULT_BUDGET
from edupulse.labels import LABELS, TASKS
from ml.common import read_json
from ml.eval_tables import model_display
from ml.plotting import vn
from ml.preds import BASELINES, DEPLOY_KEY

TASK_FIELDS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "ece",
    "nll",
    "per_class",
    "confusion",
    "reliability",
)
# Thí nghiệm sơ bộ của nghiên cứu (validation, seed 42, 4 epoch, cùng siêu tham số), mục 5 của
# plans/reports/researcher-261014-1913-phobert-vsfc-modeling.md. KHÔNG phải run được triển khai.
PRELIMINARY_ARCHITECTURES = (
    ("Đơn nhiệm (STL), sơ bộ: validation, 1 seed", 0.8701, 0.8154),
    ("Đa nhiệm (MTL), sơ bộ: validation, 1 seed", 0.8555, 0.8272),
)
PRELIMINARY_MCNEMAR_P = {"sentiment": 0.31, "topic": 0.46}


class ReleaseError(RuntimeError):
    """Kết quả đầu vào không đủ điều kiện để làm thẻ mô hình phát hành."""


def read_results(results_dir: Path) -> dict[str, Any]:
    metrics = results_dir / "metrics"
    needed = {
        "test_all": metrics / "test_all.json",
        "review": metrics / "review_budget_test.json",
        "checklist": metrics / "checklist.json",
    }
    missing = [str(path) for path in needed.values() if not path.is_file()]
    if missing:
        raise ReleaseError(f"thiếu kết quả cho thẻ phát hành: {missing} — chạy `make report`")
    data = {name: read_json(path) for name, path in needed.items()}
    gate = metrics / "onnx_gate_v1.json"
    data["gate"] = read_json(gate) if gate.is_file() else None
    return data


def check_provenance(test_all: dict[str, Any], *, run: str, freeze_sha256: str) -> None:
    """test_all.json phải do lần chấm cuối thật của đúng run này sinh ra."""
    problems = []
    if test_all.get("split") != "test" or test_all.get("rehearsal"):
        problems.append("không phải kết quả chấm test thật (split/rehearsal)")
    if test_all.get("deploy_run") != run:
        problems.append(f"là của run {test_all.get('deploy_run')!r}, không phải {run!r}")
    if test_all.get("freeze_sha256") != freeze_sha256:
        problems.append("freeze_sha256 không khớp results/freeze.json hiện tại")
    if problems:
        raise ReleaseError("test_all.json không đủ điều kiện: " + "; ".join(problems))


def task_metrics(models: dict[str, Any], key: str) -> dict[str, TaskMetrics]:
    return {
        task: TaskMetrics(**{field: models[key][task][field] for field in TASK_FIELDS})
        for task in TASKS
    }


def baseline_rows(models: dict[str, Any]) -> list[BaselineRow]:
    """Ba baseline, PhoBERT FP32 cùng run, rồi bản int8 đang triển khai (cuối cùng, `deployed`)."""
    rows = []
    for key in (*BASELINES, *(k for k in models if k not in BASELINES and k != DEPLOY_KEY)):
        rows.append(_row(models, key, deployed=False))
    rows.append(_row(models, DEPLOY_KEY, deployed=True))
    return rows


def _row(models: dict[str, Any], key: str, *, deployed: bool) -> BaselineRow:
    sentiment, topic = models[key]["sentiment"], models[key]["topic"]
    return BaselineRow(
        name=model_display(key),
        deployed=deployed,
        sentiment_accuracy=sentiment["accuracy"],
        sentiment_macro_f1=sentiment["macro_f1"],
        topic_accuracy=topic["accuracy"],
        topic_macro_f1=topic["macro_f1"],
        sentiment_ece=sentiment["ece"],
    )


def architecture_rows() -> list[ArchitectureRow]:
    return [
        ArchitectureRow(
            name=name,
            seeds=[42],
            sentiment_macro_f1=SeedStat(mean=sentiment, std=0.0, n=1),
            topic_macro_f1=SeedStat(mean=topic, std=0.0, n=1),
        )
        for name, sentiment, topic in PRELIMINARY_ARCHITECTURES
    ]


def budget_rows(review: dict[str, Any]) -> list[BudgetRow]:
    budgets = review["budgets"]
    return [
        BudgetRow(
            budget=budget,
            default=budget == DEFAULT_BUDGET,
            efficiency=BudgetEfficiency(**budgets[str(budget)]["rank_batch"]),
        )
        for budget in BUDGETS
    ]


def checklist_rows(checklist: dict[str, Any]) -> list[ChecklistRow]:
    return [ChecklistRow.model_validate(row) for row in checklist["rows"]]


def _percent(value: float, digits: int = 1) -> str:
    return vn(value * 100, digits)


def limitations(
    test_all: dict[str, Any],
    checklist: list[ChecklistRow],
    gate: dict[str, Any] | None,
    static: list[str],
) -> list[str]:
    """Giới hạn đã đo trên test (kèm số) cộng các giới hạn không gắn với một con số."""
    deployed = test_all["models"][DEPLOY_KEY]
    neutral = deployed["sentiment"]["per_class"][LABELS["sentiment"].index("neutral")]
    others = deployed["topic"]["per_class"][LABELS["topic"].index("others")]
    items = [
        "Lớp Trung tính và chủ đề Khác yếu: trên test, recall Trung tính "
        f"{_percent(neutral['recall'])}% (F1 {_percent(neutral['f1'])}), recall Khác "
        f"{_percent(others['recall'])}% (F1 {_percent(others['f1'])}).",
    ]
    robustness = next((row for row in checklist if row.kind == "robustness"), None)
    if robustness is not None:
        items.append(
            f"Câu gõ không dấu: độ chính xác giảm ({robustness.note}); hệ thống cảnh báo và gắn cờ."
        )
    bootstrap = test_all["bootstrap"]
    intervals = []
    for task, label in (("sentiment", "cảm xúc"), ("topic", "chủ đề")):
        metrics = deployed[task]
        ci = metrics["ci95"]["macro_f1"]
        low, high = _percent(ci["low"]), _percent(ci["high"])
        intervals.append(f"{label} {_percent(metrics['macro_f1'])} [{low}–{high}]")
    rounds = vn(bootstrap["rounds"], 0)
    items.append(
        f"Khoảng tin cậy 95% của macro-F1 trên test (bootstrap {rounds} vòng): "
        f"{'; '.join(intervals)}. Chênh nhỏ hơn bề rộng khoảng này giữa các mô hình là "
        "không đáng tin."
    )
    if gate is not None:
        sentiment_val = gate["int8"]["sentiment"]["macro_f1"]
        topic_val = gate["int8"]["topic"]["macro_f1"]
        items.append(
            "Macro-F1 trên test thấp hơn validation (validation đã dùng để chọn epoch nên lạc "
            f"quan): cảm xúc {_percent(deployed['sentiment']['macro_f1'])} so với "
            f"{_percent(sentiment_val)}, chủ đề {_percent(deployed['topic']['macro_f1'])} so với "
            f"{_percent(topic_val)}."
        )
    items.append(
        "Chỉ có một run huấn luyện (seed 42): chưa đo độ lệch giữa các seed. So sánh đa nhiệm với "
        "đơn nhiệm chỉ là thí nghiệm sơ bộ của nghiên cứu (validation, 1 seed, bản huấn luyện "
        f"khác bản triển khai): McNemar p = {vn(PRELIMINARY_MCNEMAR_P['sentiment'], 2)} (cảm xúc), "
        f"{vn(PRELIMINARY_MCNEMAR_P['topic'], 2)} (chủ đề), chưa thấy khác biệt có ý nghĩa."
    )
    return [*items, *static]


def build_release_card(
    *,
    bundle_version: str,
    model_version: str,
    run: str,
    freeze_sha256: str,
    results_dir: Path,
    intended_use: str,
    out_of_scope: str,
    static_limitations: list[str],
    notes: ModelCardNotes,
) -> ModelCard:
    data = read_results(results_dir)
    test_all = data["test_all"]
    check_provenance(test_all, run=run, freeze_sha256=freeze_sha256)
    models = test_all["models"]
    metrics = task_metrics(models, DEPLOY_KEY)
    checklist = checklist_rows(data["checklist"])
    return ModelCard(
        model_version=model_version,
        bundle_version=bundle_version,
        stage="release",
        metrics=SplitMetrics(
            split="test", n=test_all["n"], sentiment=metrics["sentiment"], topic=metrics["topic"]
        ),
        baselines=baseline_rows(models),
        architectures=architecture_rows(),
        review_budget=budget_rows(data["review"]),
        checklist=checklist,
        intended_use=intended_use,
        out_of_scope=out_of_scope,
        limitations=limitations(test_all, checklist, data["gate"], static_limitations),
        notes=notes,
    )
