"""Hợp đồng dự đoán của một lần đánh giá, và ngữ cảnh chạy (test thật hay diễn tập trên validation).

Mỗi mô hình một tệp `<preds_dir>/<key>.npz`; hàng i ứng với hàng i của split. Khoá trong tệp:

    y_<task>       nhãn thật
    pred_<task>    nhãn dự đoán cứng, dùng cho acc/F1/CM/kiểm định (baseline: bản thô)
    logits_<task>  logit (PhoBERT) hoặc log xác suất đã hiệu chuẩn (baseline), hình [n, K]

Khoá mô hình: `linearsvc` `logreg` `complementnb` (cả 2 tác vụ), `mtl-s<seed>` (cả 2, FP32 torch),
`stl-sentiment-s<seed>` và `stl-topic-s<seed>` (1 tác vụ), `mtl-int8` (cả 2; bản triển khai chạy
bằng ONNX Runtime từng câu một). `META.json` ghi split, số dòng và mô tả từng khoá. Thư mục chỉ
được đổi tên thành `test_preds` khi đã đủ tệp và có `.done`, nên thấy `.done` là thấy đủ dự đoán.

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
SEEDS = (42, 43, 44)

_SEEDED = re.compile(r"(?P<arch>mtl|stl-sentiment|stl-topic)-s(?P<seed>\d+)")


def tasks_of(key: str) -> tuple[str, ...]:
    """Các tác vụ mà mô hình `key` dự đoán."""
    if key in BASELINES or key == DEPLOY_KEY:
        return TASKS
    match = _SEEDED.fullmatch(key)
    if match is None:
        raise ValueError(f"khoá mô hình lạ: {key!r}")
    return {"mtl": TASKS, "stl-sentiment": ("sentiment",), "stl-topic": ("topic",)}[match["arch"]]


def arch_of(key: str) -> str:
    """'baseline' | 'mtl' | 'stl-sentiment' | 'stl-topic' | 'mtl-int8'."""
    if key in BASELINES:
        return "baseline"
    if key == DEPLOY_KEY:
        return DEPLOY_KEY
    match = _SEEDED.fullmatch(key)
    if match is None:
        raise ValueError(f"khoá mô hình lạ: {key!r}")
    return match["arch"]


def seed_of(key: str) -> int | None:
    match = _SEEDED.fullmatch(key)
    return int(match["seed"]) if match else None


def seeded_keys(arch: str, seeds: tuple[int, ...] = SEEDS) -> list[str]:
    return [f"{arch}-s{seed}" for seed in seeds]


def all_keys() -> list[str]:
    """Thứ tự chuẩn của mọi khoá: baseline, MTL, STL cảm xúc, STL chủ đề, bản int8 triển khai."""
    stl = [*seeded_keys("stl-sentiment"), *seeded_keys("stl-topic")]
    return [*BASELINES, *seeded_keys("mtl"), *stl, DEPLOY_KEY]


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
    """Các khoá đã có tệp, theo thứ tự chuẩn của `all_keys`."""
    present = {path.stem for path in directory.glob("*.npz")}
    return [key for key in all_keys() if key in present]


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
