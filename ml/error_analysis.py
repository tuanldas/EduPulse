"""Phân tích lỗi và thiên lệch của bản triển khai, tính từ dự đoán đã lưu (`ml.preds`).

    python -m ml.error_analysis                # trên test (cần outputs/test_preds/.done)
    python -m ml.error_analysis --rehearsal    # diễn tập trên validation (outputs/rehearsal/)

Chỉ dùng hai mô hình: bản triển khai `mtl-int8` (mô hình chính) và `linearsvc` (baseline mạnh
nhất, nhãn thô). Độ tin cậy là xác suất softmax lớn nhất của logits `mtl-int8` ở T = 1, chưa áp
nhiệt độ hiệu chuẩn, nên chỉ dùng để xếp hạng lỗi chứ không đọc như xác suất đã hiệu chuẩn.

Đầu ra (dưới `ctx.results_dir`):

    metrics/error_analysis.json   mọi con số, không có câu gốc
    tables/error_analysis.md      báo cáo tiếng Việt, không có câu gốc
    tables/error_samples.csv      TỐI ĐA 30 lỗi tự tin nhất kèm câu gốc: nơi duy nhất ghi câu gốc
    tables/bias_cells.csv         recall cảm xúc theo ô chủ đề x cảm xúc của bản triển khai

Các hàm lõi (`analyze`, `confident_errors`, ...) thuần: nhận khung dữ liệu cùng `Preds` và không
đọc đĩa; chỉ `run` và `main` chạm hệ thống tệp. Khung dữ liệu lấy qua `EvalContext.load_split()`,
nên mã chạy giống hệt ở hai chế độ (test thật và diễn tập).

Mã thoát: 0 xong; 2 chưa có dự đoán (chưa chấm xong) hoặc dự đoán không khớp khung dữ liệu.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import f1_score

from edupulse.labels import LABELS, LABELS_VI, SENTIMENT, TASKS, TOPIC
from edupulse.scoring import confidence, softmax_t
from ml.common import write_json
from ml.eval_tables import TASK_VI, model_display
from ml.plotting import vn
from ml.preds import (
    DEPLOY_KEY,
    DONE_FILE,
    META_FILE,
    EvalContext,
    Preds,
    context,
    is_done,
    load_preds,
    read_meta,
)
from ml.preprocess import _SPECIAL as SPECIAL_TOKEN

if TYPE_CHECKING:
    import pandas as pd

MAIN_KEY = DEPLOY_KEY  # mô hình chính: bản triển khai
REFERENCE_KEY = "linearsvc"  # baseline mạnh nhất, nhãn thô
SHORT_NAME = {MAIN_KEY: "PhoBERT int8", REFERENCE_KEY: "LinearSVC"}

SAMPLES_PER_TASK = 15
MAX_SAMPLE_ROWS = SAMPLES_PER_TASK * len(TASKS)  # error_samples.csv không bao giờ dài hơn
LONG_WORDS = 40  # câu dài: n_words > 40
SMALL_SAMPLE = 10  # ô hoặc lát có n dưới ngưỡng này là "mẫu quá ít để kết luận"
CONFIDENCE_TEMPERATURE = 1.0
EXIT_MISSING = 2

METRICS_FILE, REPORT_FILE = "error_analysis.json", "error_analysis.md"
SAMPLES_FILE, BIAS_FILE = "error_samples.csv", "bias_cells.csv"
SAMPLE_COLUMNS = ("split", "row", "task", "true", "pred", "confidence", "sentence")
BIAS_COLUMNS = ("task", "topic", "sentiment", "n", "recall", "small_sample")
FRAME_COLUMNS = ("sentence", "sentiment", "topic", "text_rdr", "n_words", "no_diacritics")

NEGATIVE, POSITIVE = SENTIMENT.index("negative"), SENTIMENT.index("positive")
LECTURER = TOPIC.index("lecturer")
# Token "thầy" và "cô" so khớp nguyên token (sau NFC), không khớp chuỗi con ("cơ sở", "côngg").
THAY, CO = (unicodedata.normalize("NFC", word) for word in ("thầy", "cô"))

NOTES = {
    "confidence": (
        "Độ tin cậy là xác suất softmax lớn nhất của logits bản triển khai ở T = 1 (chưa áp nhiệt "
        "độ hiệu chuẩn); chỉ dùng để xếp hạng lỗi."
    ),
    "macro_f1": (
        "macro-F1 của một lát tính trên các lớp có mặt trong nhãn thật hoặc dự đoán của lát; "
        "toàn tập có đủ các lớp nên trùng macro-F1 của bảng chính."
    ),
    "prior_cells": (
        "Ô theo prior: chủ đề giảng viên + cảm xúc tích cực, hoặc chủ đề khác + cảm xúc tiêu cực "
        "(chỗ quy tắc chỉ-dùng-chủ-đề đoán đúng). Ô ngược prior: mọi câu còn lại, gồm cả "
        "trung tính."
    ),
    "teacher_proxy": (
        "Lát thầy / cô chỉ là proxy vì dữ liệu không có thuộc tính nhân khẩu nào; chênh lệch chưa "
        "qua kiểm định thống kê."
    ),
    "small_sample": f'Ô hoặc lát có n < {SMALL_SAMPLE} gắn cờ "mẫu quá ít để kết luận".',
}
SMALL_SAMPLE_FLAG = "mẫu quá ít để kết luận"
EMPTY_SLICE_NOTE = "không có câu nào thuộc lát này (n = 0), nên không tính được tỉ lệ lỗi"


class InputError(RuntimeError):
    """Khung dữ liệu và dự đoán không ứng với cùng một split (thiếu cột, lệch dòng hay nhãn)."""


# --- kiểm tra đầu vào -------------------------------------------------------------------------


def check_inputs(frame: pd.DataFrame, models: dict[str, Preds]) -> None:
    """Hàng i của khung phải là hàng i của mọi dự đoán: kiểm cột, số dòng, nhãn thật và dải nhãn."""
    missing = [column for column in FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise InputError(f"khung dữ liệu thiếu cột {missing}")
    if len(frame) == 0:
        raise InputError("khung dữ liệu rỗng")
    for task in TASKS:
        truth = frame[task].to_numpy()
        top = len(LABELS[task]) - 1
        if truth.min() < 0 or truth.max() > top:
            raise InputError(f"nhãn {task} trong khung dữ liệu nằm ngoài khoảng 0..{top}")
        for key, preds in models.items():
            if task not in preds.tasks:
                raise InputError(f"{key} không có dự đoán cho tác vụ {task}")
            if len(preds.y[task]) != len(frame):
                raise InputError(
                    f"{key}/{task}: {len(preds.y[task])} dòng dự đoán nhưng khung dữ liệu có "
                    f"{len(frame)} dòng"
                )
            if not np.array_equal(preds.y[task], truth):
                raise InputError(
                    f"{key}/{task}: nhãn thật khác khung dữ liệu (sai split hoặc lệch thứ tự hàng)"
                )
            if preds.pred[task].min() < 0 or preds.pred[task].max() > top:
                raise InputError(f"{key}/{task}: nhãn dự đoán nằm ngoài khoảng 0..{top}")


# --- chỉ số nhỏ dùng chung ---------------------------------------------------------------------


def accuracy_of(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Số câu, số lỗi và accuracy; accuracy là None khi không có câu nào."""
    n = len(y)
    errors = int(np.count_nonzero(y != pred))
    return {"n": n, "errors": errors, "accuracy": (n - errors) / n if n else None}


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """`accuracy_of` cộng macro-F1 trên các lớp có mặt trong nhãn thật hoặc dự đoán.

    Lớp có trong nhãn thật nhưng không bao giờ được đoán vẫn tính F1 = 0 (như bảng chính); chỉ lớp
    vắng cả hai phía bị bỏ, để một lát mỏng không bị kéo xuống chỉ vì thiếu một lớp.
    """
    macro = float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else None
    return {**accuracy_of(y, pred), "macro_f1": macro}


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


# --- 1. cặp nhầm lẫn ---------------------------------------------------------------------------


def confusion_counts(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Ma trận đếm [nhãn thật, nhãn dự đoán]."""
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(counts, (y, pred), 1)
    return counts


def confusion_pairs(y: np.ndarray, pred: np.ndarray, labels: tuple[str, ...]) -> list[dict]:
    """Các cặp (nhãn thật → nhãn đoán) bị sai, theo số lỗi giảm dần; hoà thì theo thứ tự nhãn.

    Mỗi cặp ghi tỉ lệ trên tổng lỗi của tác vụ, số câu và recall của lớp thật.
    """
    counts = confusion_counts(np.asarray(y), np.asarray(pred), len(labels))
    support = counts.sum(axis=1)
    errors = int(counts.sum() - np.trace(counts))
    pairs: list[dict] = []
    for true_id, row in enumerate(counts):
        for pred_id, count in enumerate(row):
            if true_id == pred_id or count == 0:
                continue
            pairs.append(
                {
                    "true": labels[true_id],
                    "pred": labels[pred_id],
                    "count": int(count),
                    "share_of_errors": float(count / errors),
                    "true_support": int(support[true_id]),
                    "true_recall": float(counts[true_id, true_id] / support[true_id]),
                }
            )
    pairs.sort(key=lambda pair: -pair["count"])  # sắp ổn định: hoà giữ thứ tự (thật, đoán)
    return pairs


# --- 2. lỗi tự tin nhất ------------------------------------------------------------------------


def confident_errors(
    frame: pd.DataFrame, preds: Preds, split: str, per_task: int = SAMPLES_PER_TASK
) -> list[dict[str, Any]]:
    """`per_task` lỗi có độ tin cậy cao nhất của mỗi tác vụ, kèm câu gốc để kiểm nhãn.

    Hoà độ tin cậy thì lấy hàng nhỏ hơn. `row` là chỉ số hàng (từ 0) trong split.
    """
    sentences = frame["sentence"].tolist()
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        conf = confidence(softmax_t(preds.logits[task], CONFIDENCE_TEMPERATURE))
        wrong = np.flatnonzero(preds.pred[task] != preds.y[task])
        for row in wrong[np.argsort(-conf[wrong], kind="stable")][:per_task]:
            rows.append(
                {
                    "split": split,
                    "row": int(row),
                    "task": task,
                    "true": LABELS[task][preds.y[task][row]],
                    "pred": LABELS[task][preds.pred[task][row]],
                    "confidence": round(float(conf[row]), 6),
                    "sentence": sentences[row],
                }
            )
    return rows


def sample_index(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Chỉ mục của các mẫu lỗi để đưa vào JSON: mọi cột trừ câu gốc."""
    per_task: dict[str, Any] = {}
    for task in TASKS:
        values = [row["confidence"] for row in samples if row["task"] == task]
        per_task[task] = {
            "n": len(values),
            "max_confidence": max(values) if values else None,
            "min_confidence": min(values) if values else None,
        }
    return {
        "file": SAMPLES_FILE,
        "limit_per_task": SAMPLES_PER_TASK,
        "max_rows": MAX_SAMPLE_ROWS,
        "n_rows": len(samples),
        "per_task": per_task,
        "rows": [{key: row[key] for key in SAMPLE_COLUMNS if key != "sentence"} for row in samples],
    }


# --- 3. lỗi theo lát ---------------------------------------------------------------------------


def error_slice(mask: np.ndarray, models: dict[str, Preds]) -> dict[str, Any]:
    """Số lỗi và tỉ lệ lỗi của từng mô hình, từng tác vụ, trong các hàng được chọn bởi `mask`."""
    n = int(mask.sum())
    entry: dict[str, Any] = {"n": n}
    for key, preds in models.items():
        entry[key] = {}
        for task in TASKS:
            errors = int(((preds.pred[task] != preds.y[task]) & mask).sum())
            entry[key][task] = {"errors": errors, "error_rate": _rate(errors, n)}
    if n == 0:
        entry["note"] = EMPTY_SLICE_NOTE
    return entry


def error_slices(frame: pd.DataFrame, models: dict[str, Preds]) -> dict[str, Any]:
    """Lỗi theo chủ đề thật, độ dài, token đặc biệt và gõ không dấu (kèm dòng toàn tập)."""
    topic = frame["topic"].to_numpy()
    long = frame["n_words"].to_numpy() > LONG_WORDS
    special = frame["text_rdr"].str.contains(SPECIAL_TOKEN).to_numpy(dtype=bool)
    unmarked = frame["no_diacritics"].to_numpy(dtype=bool)
    groups = {
        "by_true_topic": {name: topic == i for i, name in enumerate(TOPIC)},
        "by_length": {"short": ~long, "long": long},
        "by_special_token": {"without": ~special, "with": special},
        "by_diacritics": {"with_diacritics": ~unmarked, "no_diacritics": unmarked},
    }
    slices: dict[str, Any] = {"overall": error_slice(np.ones(len(frame), dtype=bool), models)}
    for group, masks in groups.items():
        slices[group] = {name: error_slice(mask, models) for name, mask in masks.items()}
    return slices


# --- 4. giao lỗi -------------------------------------------------------------------------------


def error_overlap(deployed: Preds, reference: Preds) -> dict[str, Any]:
    """Mỗi tác vụ: số câu cả hai sai, chỉ bản triển khai sai, chỉ baseline sai, cả hai đúng."""
    result: dict[str, Any] = {}
    for task in TASKS:
        main_wrong = deployed.pred[task] != deployed.y[task]
        reference_wrong = reference.pred[task] != reference.y[task]
        both = int((main_wrong & reference_wrong).sum())
        only_main = int((main_wrong & ~reference_wrong).sum())
        only_reference = int((~main_wrong & reference_wrong).sum())
        result[task] = {
            "n": len(main_wrong),
            "both_wrong": both,
            "only_main_wrong": only_main,
            "only_reference_wrong": only_reference,
            "both_right": int(len(main_wrong) - both - only_main - only_reference),
            "main_errors": both + only_main,
            "reference_errors": both + only_reference,
            "main_errors_also_reference_wrong": _rate(both, both + only_main),
        }
    return result


# --- 5. thiên lệch -----------------------------------------------------------------------------


def topic_sentiment_cells(frame: pd.DataFrame, preds: Preds) -> list[dict[str, Any]]:
    """Recall cảm xúc trong từng ô (chủ đề thật x cảm xúc thật); ô n < SMALL_SAMPLE gắn cờ."""
    topic, truth = frame["topic"].to_numpy(), frame["sentiment"].to_numpy()
    right = preds.pred["sentiment"] == truth
    cells: list[dict[str, Any]] = []
    for topic_id, topic_name in enumerate(TOPIC):
        for sentiment_id, sentiment_name in enumerate(SENTIMENT):
            mask = (topic == topic_id) & (truth == sentiment_id)
            n = int(mask.sum())
            cells.append(
                {
                    "task": "sentiment",
                    "topic": topic_name,
                    "sentiment": sentiment_name,
                    "n": n,
                    "recall": float(right[mask].mean()) if n else None,
                    "small_sample": n < SMALL_SAMPLE,
                }
            )
    return cells


def topic_only_baseline(frame: pd.DataFrame, models: dict[str, Preds]) -> dict[str, Any]:
    """Đoán cảm xúc chỉ từ NHÃN CHỦ ĐỀ THẬT, đặt cạnh các mô hình thật.

    Quy tắc: chủ đề giảng viên → tích cực, còn lại → tiêu cực. Ô "theo prior" là các câu mà quy
    tắc ấy đoán đúng, ô "ngược prior" là mọi câu còn lại; mỗi mô hình báo accuracy ở hai ô.
    """
    topic, truth = frame["topic"].to_numpy(), frame["sentiment"].to_numpy()
    guess = np.where(topic == LECTURER, POSITIVE, NEGATIVE)
    on_prior = guess == truth
    result: dict[str, Any] = {
        "rule": "chủ đề thật là giảng viên → tích cực, còn lại → tiêu cực",
        "topic_only": scores(truth, guess),
        "cells": {"on_prior": int(on_prior.sum()), "against_prior": int((~on_prior).sum())},
        "models": {},
    }
    for key, preds in models.items():
        pred = preds.pred["sentiment"]
        on = accuracy_of(truth[on_prior], pred[on_prior])
        against = accuracy_of(truth[~on_prior], pred[~on_prior])
        gap = None
        if on["accuracy"] is not None and against["accuracy"] is not None:
            gap = on["accuracy"] - against["accuracy"]
        result["models"][key] = {
            "overall": scores(truth, pred),
            "on_prior": on,
            "against_prior": against,
            "gap": gap,
        }
    return result


def _tokens(sentence: str) -> set[str]:
    return set(unicodedata.normalize("NFC", sentence).lower().split())


def teacher_slices(frame: pd.DataFrame, preds: Preds) -> dict[str, Any]:
    """Cảm xúc của bản triển khai trên câu có token "thầy" so với câu có token "cô".

    Token tách theo khoảng trắng của cột `sentence`. Câu có cả hai token bị bỏ khỏi cả hai lát.
    `difference` là thầy trừ cô. Đây chỉ là proxy: dữ liệu không có thuộc tính nhân khẩu.
    """
    tokens = [_tokens(sentence) for sentence in frame["sentence"]]
    has_thay = np.fromiter((THAY in found for found in tokens), dtype=bool, count=len(tokens))
    has_co = np.fromiter((CO in found for found in tokens), dtype=bool, count=len(tokens))
    truth, pred = frame["sentiment"].to_numpy(), preds.pred["sentiment"]

    def one(mask: np.ndarray) -> dict[str, Any]:
        part_truth, part_pred = truth[mask], pred[mask]
        found = scores(part_truth, part_pred)
        n = found["n"]
        return {
            "n": n,
            "negative_rate_true": _rate(int((part_truth == NEGATIVE).sum()), n),
            "negative_rate_pred": _rate(int((part_pred == NEGATIVE).sum()), n),
            "accuracy": found["accuracy"],
            "macro_f1": found["macro_f1"],
            "small_sample": n < SMALL_SAMPLE,
        }

    thay, co = one(has_thay & ~has_co), one(has_co & ~has_thay)
    metrics = ("negative_rate_true", "negative_rate_pred", "accuracy", "macro_f1")
    return {
        "n_both": int((has_thay & has_co).sum()),
        "thay": thay,
        "co": co,
        "difference": {
            metric: None
            if thay[metric] is None or co[metric] is None
            else thay[metric] - co[metric]
            for metric in metrics
        },
    }


# --- tổng hợp ----------------------------------------------------------------------------------


def analyze(
    frame: pd.DataFrame, deployed: Preds, reference: Preds, *, split: str
) -> dict[str, Any]:
    """Mọi con số của phân tích (không có câu gốc) từ khung dữ liệu và hai bộ dự đoán.

    Nhận `InputError` khi khung không khớp dự đoán, thay vì tính ra số sai trên hàng lệch.
    """
    models = {MAIN_KEY: deployed, REFERENCE_KEY: reference}
    check_inputs(frame, models)
    return {
        "split": split,
        "n": len(frame),
        "models": {"main": MAIN_KEY, "reference": REFERENCE_KEY},
        "params": {
            "samples_per_task": SAMPLES_PER_TASK,
            "long_words": LONG_WORDS,
            "small_sample": SMALL_SAMPLE,
            "confidence_temperature": CONFIDENCE_TEMPERATURE,
        },
        "overall": {
            key: {task: scores(preds.y[task], preds.pred[task]) for task in TASKS}
            for key, preds in models.items()
        },
        "confusion_pairs": {
            key: {
                task: confusion_pairs(preds.y[task], preds.pred[task], LABELS[task])
                for task in TASKS
            }
            for key, preds in models.items()
        },
        "error_samples": sample_index(confident_errors(frame, deployed, split)),
        "slices": error_slices(frame, models),
        "overlap": error_overlap(deployed, reference),
        "bias": {
            "topic_sentiment_cells": topic_sentiment_cells(frame, deployed),
            "topic_only_baseline": topic_only_baseline(frame, models),
            "teacher_slices": teacher_slices(frame, deployed),
        },
        "notes": dict(NOTES),
    }


# --- báo cáo markdown --------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{vn(value * 100, digits)}%"


def _points(value: float | None, digits: int = 1) -> str:
    """Hiệu hai tỉ lệ, in bằng điểm phần trăm có dấu."""
    if value is None:
        return "–"
    points = round(value * 100, digits)
    return f"{'+' if points >= 0 else '−'}{vn(abs(points), digits)}"


def _count(value: int) -> str:
    return vn(value, 0)


def _table(columns: list[str], rows: list[list[str]], align: str) -> list[str]:
    """Bảng markdown; `align` có một ký tự mỗi cột: 'l' căn trái, 'r' căn phải."""
    marks = {"l": "---", "r": "---:"}
    return [
        f"| {' | '.join(columns)} |",
        f"|{'|'.join(marks[a] for a in align)}|",
        *(f"| {' | '.join(row)} |" for row in rows),
    ]


def _split_label(split: str) -> str:
    return "test" if split == "test" else f"{split} (diễn tập, KHÔNG phải test)"


def _label_vi(task: str, label: str) -> str:
    return LABELS_VI[task][label]


def _confusion_markdown(result: dict[str, Any]) -> list[str]:
    lines = [
        "## 1. Cặp nhầm lẫn",
        "",
        'Mỗi dòng là một cặp (nhãn thật → nhãn dự đoán), xếp theo số lỗi giảm dần. "% tổng lỗi" '
        'là phần của cặp trong mọi lỗi của tác vụ; "Recall lớp thật" là tỉ lệ câu của lớp thật '
        "được đoán đúng.",
    ]
    columns = ["Nhãn thật → dự đoán", "Số lỗi", "% tổng lỗi", "Số câu lớp thật", "Recall lớp thật"]
    for task in TASKS:
        lines += ["", f"### {TASK_VI[task]}"]
        for key in (MAIN_KEY, REFERENCE_KEY):
            overall = result["overall"][key][task]
            lines += [
                "",
                f"**{model_display(key)}**: {_count(overall['errors'])} lỗi trên "
                f"{_count(overall['n'])} câu (đúng {_pct(overall['accuracy'])}).",
                "",
            ]
            pairs = result["confusion_pairs"][key][task]
            if not pairs:
                lines.append("Không có lỗi nào.")
                continue
            rows = [
                [
                    f"{_label_vi(task, p['true'])} → {_label_vi(task, p['pred'])}",
                    _count(p["count"]),
                    _pct(p["share_of_errors"]),
                    _count(p["true_support"]),
                    _pct(p["true_recall"]),
                ]
                for p in pairs
            ]
            lines += _table(columns, rows, "lrrrr")
    return lines


def _samples_markdown(result: dict[str, Any]) -> list[str]:
    info = result["error_samples"]
    lines = [
        "## 2. Lỗi tự tin nhất (kiểm nhãn)",
        "",
        f"Mỗi tác vụ lấy {info['limit_per_task']} lỗi có độ tin cậy cao nhất của "
        f"{model_display(MAIN_KEY)}, tối đa {info['max_rows']} dòng, ghi cùng câu gốc ở "
        f"`{info['file']}` để người xem đối chiếu lại nhãn. Đó là tệp duy nhất chứa câu gốc. "
        f"{result['notes']['confidence']}",
        "",
    ]
    rows = []
    for task in TASKS:
        item = info["per_task"][task]
        rows.append(
            [
                TASK_VI[task],
                _count(item["n"]),
                "–" if item["max_confidence"] is None else vn(item["max_confidence"], 4),
                "–" if item["min_confidence"] is None else vn(item["min_confidence"], 4),
            ]
        )
    columns = ["Tác vụ", "Số dòng", "Độ tin cậy cao nhất", "Độ tin cậy thấp nhất trong nhóm"]
    return lines + _table(columns, rows, "lrrr")


SLICE_GROUPS = {
    "by_true_topic": (
        "3.1 Theo chủ đề thật",
        {name: LABELS_VI["topic"][name] for name in TOPIC},
    ),
    "by_length": (
        f"3.2 Theo độ dài (số từ sau tách, ngưỡng {LONG_WORDS})",
        {"short": f"≤ {LONG_WORDS} từ", "long": f"> {LONG_WORDS} từ"},
    ),
    "by_special_token": (
        "3.3 Theo token đặc biệt (wzjwz, colon..., doubledot, ...)",
        {"with": "Có token đặc biệt", "without": "Không có"},
    ),
    "by_diacritics": (
        "3.4 Theo gõ không dấu",
        {"with_diacritics": "Có dấu", "no_diacritics": "Gõ không dấu"},
    ),
}


def _slice_cell(entry: dict[str, Any]) -> str:
    if entry["error_rate"] is None:
        return "–"
    return f"{_count(entry['errors'])} ({_pct(entry['error_rate'])})"


def _slices_markdown(result: dict[str, Any]) -> list[str]:
    lines = [
        "## 3. Lỗi theo lát",
        "",
        "Mỗi ô là số lỗi (tỉ lệ lỗi trong lát). Ở lát chủ đề thật, tỉ lệ lỗi chủ đề của một chủ đề "
        "bằng 1 trừ recall của chủ đề ấy. Lát rỗng (n = 0) không có tỉ lệ.",
    ]
    columns = ["Lát", "n"] + [
        f"{TASK_VI[task]} · {SHORT_NAME[key]}"
        for task in TASKS
        for key in (MAIN_KEY, REFERENCE_KEY)
    ]
    for group, (title, names) in SLICE_GROUPS.items():
        entries = [("Toàn tập", result["slices"]["overall"])]
        entries += [(label, result["slices"][group][name]) for name, label in names.items()]
        rows = [
            [
                label,
                _count(entry["n"]),
                *(
                    _slice_cell(entry[key][task])
                    for task in TASKS
                    for key in (MAIN_KEY, REFERENCE_KEY)
                ),
            ]
            for label, entry in entries
        ]
        lines += ["", f"### {title}", "", *_table(columns, rows, "l" + "r" * (len(columns) - 1))]
        notes = [
            f'- Lát "{label}": {entry["note"]}.' for label, entry in entries if "note" in entry
        ]
        lines += ["", *notes] if notes else []
    return lines


def _overlap_markdown(result: dict[str, Any]) -> list[str]:
    main, reference = SHORT_NAME[MAIN_KEY], SHORT_NAME[REFERENCE_KEY]
    lines = [
        "## 4. Giao lỗi TF-IDF ∩ PhoBERT",
        "",
        f"Mỗi câu rơi vào đúng một trong bốn ô, tuỳ {main} và {reference} (nhãn thô) có đoán đúng "
        "hay không.",
        "",
    ]
    columns = [
        "Tác vụ",
        "Cả hai sai",
        f"Chỉ {main} sai",
        f"Chỉ {reference} sai",
        "Cả hai đúng",
        f"Lỗi {main} cũng bị {reference} sai",
    ]
    rows = []
    for task in TASKS:
        item = result["overlap"][task]
        shared = item["main_errors_also_reference_wrong"]
        rows.append(
            [
                TASK_VI[task],
                _count(item["both_wrong"]),
                _count(item["only_main_wrong"]),
                _count(item["only_reference_wrong"]),
                _count(item["both_right"]),
                "–"
                if shared is None
                else f"{_count(item['both_wrong'])} / {_count(item['main_errors'])} "
                f"({_pct(shared)})",
            ]
        )
    return lines + _table(columns, rows, "lrrrrr")


def _bias_markdown(result: dict[str, Any]) -> list[str]:
    bias = result["bias"]
    lines = [
        "## 5. Thiên lệch",
        "",
        "### 5.1 Recall cảm xúc theo ô chủ đề × cảm xúc",
        "",
        f"Mô hình: {model_display(MAIN_KEY)}. Chủ đề và cảm xúc đều là nhãn thật; recall là tỉ "
        f"lệ câu được đoán đúng cảm xúc trong ô. {result['notes']['small_sample']}",
        "",
    ]
    rows = [
        [
            _label_vi("topic", cell["topic"]),
            _label_vi("sentiment", cell["sentiment"]),
            _count(cell["n"]),
            _pct(cell["recall"]),
            SMALL_SAMPLE_FLAG if cell["small_sample"] else "",
        ]
        for cell in bias["topic_sentiment_cells"]
    ]
    columns = ["Chủ đề thật", "Cảm xúc thật", "n", "Recall", "Ghi chú"]
    lines += _table(columns, rows, "llrrl")

    base = bias["topic_only_baseline"]
    lines += [
        "",
        "### 5.2 Baseline chỉ-dùng-chủ-đề",
        "",
        f"Quy tắc: {base['rule']}, không đọc câu. {result['notes']['prior_cells']}",
        "",
    ]
    rows = [
        [
            "Chỉ dùng chủ đề thật",
            _pct(base["topic_only"]["accuracy"]),
            _pct(base["topic_only"]["macro_f1"]),
        ]
    ]
    rows += [
        [model_display(key), _pct(item["overall"]["accuracy"]), _pct(item["overall"]["macro_f1"])]
        for key, item in base["models"].items()
    ]
    lines += _table(["Cách đoán cảm xúc", "Accuracy", "macro-F1"], rows, "lrr")
    n_on, n_against = base["cells"]["on_prior"], base["cells"]["against_prior"]
    lines += [""]
    rows = [
        [
            model_display(key),
            f"{_pct(item['on_prior']['accuracy'])} (n = {_count(n_on)})",
            f"{_pct(item['against_prior']['accuracy'])} (n = {_count(n_against)})",
            _points(item["gap"]),
        ]
        for key, item in base["models"].items()
    ]
    columns = ["Accuracy cảm xúc của", "Ô theo prior", "Ô ngược prior", "Chênh lệch (điểm)"]
    lines += _table(columns, rows, "lrrr")

    teacher = bias["teacher_slices"]
    lines += [
        "",
        '### 5.3 Lát "thầy" so với "cô"',
        "",
        f"Cảm xúc của {model_display(MAIN_KEY)}. Token tách theo khoảng trắng và so khớp nguyên "
        f'token. Câu có cả "thầy" lẫn "cô" (n = {_count(teacher["n_both"])}) bị bỏ khỏi cả hai '
        f"lát. {result['notes']['teacher_proxy']} {result['notes']['macro_f1']}",
        "",
    ]
    rows = []
    for label, item in (('Có token "thầy"', teacher["thay"]), ('Có token "cô"', teacher["co"])):
        flag = f" ({SMALL_SAMPLE_FLAG})" if item["small_sample"] else ""
        rows.append(
            [
                label + flag,
                _count(item["n"]),
                _pct(item["negative_rate_true"]),
                _pct(item["negative_rate_pred"]),
                _pct(item["accuracy"]),
                _pct(item["macro_f1"]),
            ]
        )
    diff = teacher["difference"]
    rows.append(
        [
            "Hiệu số (thầy − cô), điểm",
            "",
            _points(diff["negative_rate_true"]),
            _points(diff["negative_rate_pred"]),
            _points(diff["accuracy"]),
            _points(diff["macro_f1"]),
        ]
    )
    columns = ["Lát", "n", "Tiêu cực thật", "Tiêu cực dự đoán", "Accuracy", "macro-F1"]
    return lines + _table(columns, rows, "lrrrrr")


def render_markdown(result: dict[str, Any]) -> str:
    """Báo cáo tiếng Việt các mục 1, 3, 4, 5; không chứa câu gốc (mục 2 chỉ trỏ tới CSV)."""
    deploy_run = (result.get("source") or {}).get("deploy_run")
    origin = f", lấy từ run `{deploy_run}`" if deploy_run else ""
    intro = (
        f"Tập {_split_label(result['split'])}, n = {_count(result['n'])}. Mô hình chính: "
        f"{model_display(MAIN_KEY)}{origin}; đối chứng: {model_display(REFERENCE_KEY)} (nhãn "
        "thô). Các bảng chỉ chứa số đếm, tỉ lệ và nhãn."
    )
    sections = [
        ["# Phân tích lỗi và thiên lệch", "", intro],
        _confusion_markdown(result),
        _samples_markdown(result),
        _slices_markdown(result),
        _overlap_markdown(result),
        _bias_markdown(result),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


# --- ghi tệp và dòng lệnh ----------------------------------------------------------------------


def write_error_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    """Ghi `error_samples.csv`, nơi duy nhất được chứa câu gốc, với tối đa MAX_SAMPLE_ROWS dòng."""
    if len(rows) > MAX_SAMPLE_ROWS:
        raise ValueError(
            f"{SAMPLES_FILE} chỉ được chứa tối đa {MAX_SAMPLE_ROWS} dòng, nhận {len(rows)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_bias_cells(path: Path, cells: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BIAS_COLUMNS)
        writer.writeheader()
        for cell in cells:
            recall = "" if cell["recall"] is None else round(cell["recall"], 4)
            writer.writerow({**cell, "recall": recall})


def run(ctx: EvalContext) -> tuple[list[Path], dict[str, Any]]:
    """Đọc dự đoán của `ctx.preds_dir` và khung dữ liệu của `ctx.split`, ghi bốn tệp đầu ra.

    Trả về đường dẫn các tệp đã ghi và kết quả phân tích.
    """
    meta_path = ctx.preds_dir / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"chưa có {meta_path}")
    meta = read_meta(ctx.preds_dir)
    deployed = load_preds(ctx.preds_dir, MAIN_KEY)
    reference = load_preds(ctx.preds_dir, REFERENCE_KEY)
    frame = ctx.load_split()
    result = analyze(frame, deployed, reference, split=ctx.split)
    result["source"] = {
        "rehearsal": ctx.rehearsal,
        "deploy_run": meta.get("deploy_run"),
        "evaluated_at": meta.get("created_at"),
        "git_commit": meta.get("git_commit"),
    }

    samples_path = ctx.table_path(SAMPLES_FILE)
    write_error_samples(samples_path, confident_errors(frame, deployed, ctx.split))
    report_path = ctx.table_path(REPORT_FILE)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(result), encoding="utf-8")
    bias_path = ctx.table_path(BIAS_FILE)
    write_bias_cells(bias_path, result["bias"]["topic_sentiment_cells"])
    metrics_path = ctx.metrics_path(METRICS_FILE)
    write_json(metrics_path, result)
    return [metrics_path, report_path, samples_path, bias_path], result


def headline(result: dict[str, Any]) -> list[str]:
    """Vài dòng số chính để in ra màn hình."""
    lines = [f"Tập {_split_label(result['split'])}, n = {_count(result['n'])}:"]
    for task in TASKS:
        ours, theirs = (result["overall"][key][task] for key in (MAIN_KEY, REFERENCE_KEY))
        both = result["overlap"][task]["both_wrong"]
        lines.append(
            f"  {TASK_VI[task]}: {SHORT_NAME[MAIN_KEY]} sai {_count(ours['errors'])} "
            f"({_pct(ours['accuracy'])} đúng), {SHORT_NAME[REFERENCE_KEY]} sai "
            f"{_count(theirs['errors'])}, cả hai sai {_count(both)}"
        )
    base = result["bias"]["topic_only_baseline"]
    deployed = base["models"][MAIN_KEY]
    lines.append(
        f"  Chỉ-dùng-chủ-đề: acc {_pct(base['topic_only']['accuracy'])}; bản triển khai ô theo "
        f"prior {_pct(deployed['on_prior']['accuracy'])} so với ngược prior "
        f"{_pct(deployed['against_prior']['accuracy'])}"
    )
    teacher = result["bias"]["teacher_slices"]
    lines.append(
        f"  Thầy (n = {_count(teacher['thay']['n'])}) so với cô "
        f"(n = {_count(teacher['co']['n'])}): chênh accuracy "
        f"{_points(teacher['difference']['accuracy'])} điểm"
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="diễn tập trên validation (đọc và ghi trong outputs/rehearsal/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context(args.rehearsal)
    if not is_done(ctx.preds_dir):
        hint = "--rehearsal" if ctx.rehearsal else "--split test --final"
        print(
            f"error_analysis: chưa có {ctx.preds_dir / DONE_FILE}: chưa chấm xong "
            f"(chạy `python -m ml.evaluate {hint}`)",
            file=sys.stderr,
        )
        return EXIT_MISSING
    try:
        written, result = run(ctx)
    except (FileNotFoundError, InputError) as error:
        print(f"error_analysis: {error}", file=sys.stderr)
        return EXIT_MISSING
    for path in written:
        print(f"-> {path}")
    print("\n".join(headline(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
