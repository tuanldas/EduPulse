"""Hợp đồng dự đoán của một lần đánh giá, và ngữ cảnh chạy (test thật hay diễn tập trên validation).

Mỗi mô hình một tệp `<preds_dir>/<key>.npz`; hàng i ứng với hàng i của split. Khoá trong tệp:

    y_<task>       nhãn thật
    pred_<task>    nhãn dự đoán cứng, dùng cho acc/F1/CM/kiểm định (baseline: bản thô)
    logits_<task>  logit (PhoBERT) hoặc log xác suất đã hiệu chuẩn (baseline), hình [n, K]

Khoá mô hình: `linearsvc` `logreg` `complementnb` (baseline), `mtl-s<seed>` (PhoBERT đa nhiệm FP32
bằng torch trên CPU; tên run), `mtl-int8` (bản triển khai chạy bằng ONNX Runtime từng câu một);
đều dự đoán cả 2 tác vụ. `META.json` ghi split, số dòng và mô tả từng khoá. Thư mục chỉ được đổi
tên thành `test_preds` khi đã đủ tệp và có `.done`, nên thấy `.done` là thấy đủ dự đoán.

Mọi phân tích sau khi chấm (kiểm định, lỗi, ngân sách, thẻ mô hình) đọc từ đây, không suy luận lại.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from edupulse.config import REPO_ROOT
from edupulse.labels import TASKS
from ml.common import load_processed, read_json, settings

if TYPE_CHECKING:
    import pandas as pd

BASELINES = ("linearsvc", "logreg", "complementnb")
DEPLOY_KEY = "mtl-int8"
DONE_FILE = ".done"
META_FILE = "META.json"

_FP32_KEY = re.compile(r"mtl-s(?P<seed>\d+)")


def is_fp32_key(key: str) -> bool:
    return _FP32_KEY.fullmatch(key) is not None


def seed_of(key: str) -> int | None:
    match = _FP32_KEY.fullmatch(key)
    return int(match["seed"]) if match else None


def check_key(key: str) -> str:
    if key not in BASELINES and key != DEPLOY_KEY and not is_fp32_key(key):
        raise ValueError(f"khoá mô hình lạ: {key!r}")
    return key


def final_keys(deploy_run: str) -> list[str]:
    """Các mô hình được chấm ở lần cuối: baseline, run PhoBERT FP32 được triển khai, bản int8."""
    return [*BASELINES, check_key(deploy_run), DEPLOY_KEY]


def key_order(key: str) -> tuple[int, int]:
    """Thứ tự chuẩn: baseline (theo BASELINES), rồi PhoBERT FP32 (theo seed), rồi bản int8."""
    if key in BASELINES:
        return 0, BASELINES.index(key)
    if key == DEPLOY_KEY:
        return 2, 0
    return 1, seed_of(check_key(key)) or 0


@dataclass
class Preds:
    key: str
    y: dict[str, np.ndarray]
    pred: dict[str, np.ndarray]
    logits: dict[str, np.ndarray]
    tasks: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.tasks = tuple(self.pred)
        if set(self.y) != set(self.pred) or set(self.logits) != set(self.pred):
            raise ValueError(f"{self.key}: y, pred, logits phải có cùng tập tác vụ")
        for task in self.tasks:
            n = len(self.y[task])
            if len(self.pred[task]) != n or len(self.logits[task]) != n:
                raise ValueError(f"{self.key}/{task}: y, pred, logits lệch số dòng")


def save_preds(directory: Path, preds: Preds) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for task in preds.tasks:
        arrays[f"y_{task}"] = np.asarray(preds.y[task], dtype=np.int64)
        arrays[f"pred_{task}"] = np.asarray(preds.pred[task], dtype=np.int64)
        arrays[f"logits_{task}"] = np.asarray(preds.logits[task], dtype=np.float32)
    np.savez(directory / f"{preds.key}.npz", **arrays)


def load_preds(directory: Path, key: str) -> Preds:
    path = directory / f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"chưa có dự đoán {path}")
    data = np.load(path)
    tasks = tuple(task for task in TASKS if f"pred_{task}" in data.files)
    return Preds(
        key=key,
        y={task: data[f"y_{task}"] for task in tasks},
        pred={task: data[f"pred_{task}"] for task in tasks},
        logits={task: data[f"logits_{task}"] for task in tasks},
    )


def available_keys(directory: Path) -> list[str]:
    """Các khoá đã có tệp, theo thứ tự chuẩn (baseline, PhoBERT FP32, bản int8)."""
    return sorted((path.stem for path in directory.glob("*.npz")), key=key_order)


def read_meta(directory: Path) -> dict[str, Any]:
    return read_json(directory / META_FILE)


def is_done(directory: Path) -> bool:
    return (directory / DONE_FILE).is_file()


@dataclass(frozen=True)
class EvalContext:
    """Nơi đọc/ghi của một lần đánh giá: thật trên test, hoặc diễn tập trên validation.

    Diễn tập chạy đúng các bước của đánh giá cuối (dự đoán, bảng, hình, kiểm định, lỗi, ngân sách,
    thẻ mô hình) trên tập validation, ghi vào `outputs/rehearsal/` chứ không vào `results/`, để bắt
    lỗi mã trước khi chạm test; tập test thật chỉ mở bởi `--final` sau khi `freeze.json` đã commit.
    """

    split: str  # 'test' | 'validation'
    preds_dir: Path
    results_dir: Path  # gốc chứa metrics/, tables/, figures/
    rehearsal: bool

    @property
    def split_label(self) -> str:
        return "test" if self.split == "test" else "val"

    def load_split(self) -> pd.DataFrame:
        return load_processed(self.split, allow_test=self.split == "test")

    def metrics_path(self, name: str) -> Path:
        return self.results_dir / "metrics" / name

    def table_path(self, name: str) -> Path:
        return self.results_dir / "tables" / name

    def figure_path(self, name: str) -> Path:
        return self.results_dir / "figures" / name


def context(rehearsal: bool = False) -> EvalContext:
    outputs = settings().outputs_dir
    if rehearsal:
        base = outputs / "rehearsal"
        return EvalContext("validation", base / "preds", base / "results", rehearsal=True)
    return EvalContext("test", outputs / "test_preds", REPO_ROOT / "results", rehearsal=False)
