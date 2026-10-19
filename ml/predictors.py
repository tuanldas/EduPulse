"""Dự đoán của từng loại mô hình trên một split, theo hợp đồng của `ml.preds`.

- Baseline TF-IDF: nhãn thô của pipeline; logit = log xác suất đã hiệu chuẩn (nhiệt độ trên val).
- PhoBERT FP32: torch trên CPU (tất định, không cần khoá MPS), id token như lúc huấn luyện.
- Bản triển khai int8: ONNX Runtime từng câu một (batch 1), đúng điều kiện phục vụ và hiệu chuẩn.

Hàm ở đây không biết split nào là test: ai gọi chịu trách nhiệm về quyền đọc test (`evaluate.py`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from edupulse.labels import LABELS, TASKS
from ml.common import phobert_runs_dir, settings
from ml.preds import BASELINES, DEPLOY_KEY, Preds

if TYPE_CHECKING:
    import pandas as pd


def baseline_path(task: str, name: str) -> Path:
    return settings().models_dir / "baseline" / f"{task}_{name}.skops"


def baseline_preds(name: str, frame: pd.DataFrame) -> Preds:
    from ml.train_baseline import load_model

    texts = frame["text_rdr"].tolist()
    y, pred, logits = {}, {}, {}
    for task in TASKS:
        calibrated = load_model(baseline_path(task, name))
        expected = np.arange(len(LABELS[task]))
        if not np.array_equal(calibrated.classes_, expected):
            raise RuntimeError(f"{name}/{task}: lớp {calibrated.classes_} khác {expected}")
        pipeline = calibrated.calibrated_classifiers_[0].estimator.estimator
        y[task] = frame[task].to_numpy()
        pred[task] = np.asarray(pipeline.predict(texts))
        logits[task] = np.log(np.clip(calibrated.predict_proba(texts), 1e-12, 1.0))
    return Preds(name, y, pred, logits)


def phobert_run_dir(key: str) -> Path:
    return phobert_runs_dir() / key


def phobert_preds(key: str, frame: pd.DataFrame) -> Preds:
    import torch

    from ml.phobert_model import MAX_LEN, MultiHeadPhoBERT, encode_ids, load_tokenizer
    from ml.train_phobert import predict

    model = MultiHeadPhoBERT.from_pretrained(phobert_run_dir(key)).eval()
    tasks = tuple(model.config.task_heads)
    sequences = encode_ids(load_tokenizer(MAX_LEN), frame["text_rdr"].tolist())
    logits = predict(model, sequences, torch.device("cpu"), tasks)
    return Preds(
        key,
        y={task: frame[task].to_numpy() for task in tasks},
        pred={task: logits[task].argmax(axis=1) for task in tasks},
        logits={task: logits[task] for task in tasks},
    )


def deploy_onnx_path(deploy_run: str) -> Path:
    return settings().outputs_dir / "onnx" / deploy_run / "model.int8.onnx"


def int8_preds(deploy_run: str, frame: pd.DataFrame, threads: int) -> Preds:
    from ml.export_onnx import encode, make_session, run_logits

    session = make_session(deploy_onnx_path(deploy_run), threads)
    # Từng câu một (batch 1): int8 động phụ thuộc lô, còn calibration.json cũng tính ở batch 1.
    logits = run_logits(session, encode(frame["text_rdr"].tolist()), batch_size=1)
    return Preds(
        DEPLOY_KEY,
        y={task: frame[task].to_numpy() for task in TASKS},
        pred={task: logits[task].argmax(axis=1) for task in TASKS},
        logits={task: logits[task] for task in TASKS},
    )


def describe(key: str, deploy_run: str) -> dict[str, Any]:
    if key in BASELINES:
        return {"kind": "tf-idf, nhãn thô; logit = log xác suất đã hiệu chuẩn nhiệt độ"}
    if key == DEPLOY_KEY:
        return {"kind": "onnx int8, batch 1, CPU", "run": deploy_run}
    return {"kind": "torch fp32, CPU"}


def compute_preds(
    keys: list[str],
    frame: pd.DataFrame,
    deploy_run: str,
    *,
    threads: int,
    log: Callable[[str], None] = print,
) -> tuple[list[Preds], dict[str, Any]]:
    """Dự đoán của `keys` trên `frame`, cùng mô tả (thời gian, nguồn) để ghi vào META."""
    results: list[Preds] = []
    meta: dict[str, Any] = {}
    for key in keys:
        started = time.time()
        if key in BASELINES:
            preds = baseline_preds(key, frame)
        elif key == DEPLOY_KEY:
            preds = int8_preds(deploy_run, frame, threads)
        else:
            preds = phobert_preds(key, frame)
        results.append(preds)
        meta[key] = {**describe(key, deploy_run), "seconds": round(time.time() - started, 1)}
        log(f"  {key:<20} {meta[key]['seconds']:>6.1f}s")
    return results, meta
