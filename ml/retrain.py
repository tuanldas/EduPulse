"""Huấn luyện lại từ nhãn do người duyệt sửa, và chỉ nhận mô hình mới khi nó không tệ đi.

    python -m ml.retrain --feedback fb1.csv [fb2.csv ...] [--model baseline|phobert]
                         [--max-steps 20] [--feedback-weight 1] [--out DIR]

Đầu vào là file "nhãn đã sửa" (CSV utf-8-sig) mà API xuất sau khi duyệt, cột theo
`edupulse.api.export.CORRECTED_HEADER`: id_hash, text_masked, text_model, model_version,
pred_sentiment, conf_sentiment, pred_topic, conf_topic, label_sentiment, label_topic, source
(flag|audit|none), reviewed. `text_model` là câu đã chuẩn hoá và tách từ (cùng dạng cột `text_rdr`
của dữ liệu); label_sentiment và label_topic là nhãn đã duyệt: số theo thứ tự `edupulse.labels`
(cảm xúc 0..2, chủ đề 0..3) hoặc tên tiếng Anh (negative, lecturer, ...). Sau khi đọc, bảng nội bộ
gọi hai cột nhãn là label_s và label_t.

Các bước: đọc và kiểm (chỉ giữ dòng đã duyệt) -> làm sạch (bỏ câu trùng nhau và câu đã có trong
dữ liệu gốc) -> huấn luyện lại -> so macro-F1 trên VALIDATION trước/sau -> chỉ ghi mô hình mới khi
không tác vụ nào tụt. Test không bao giờ dùng để chấm; chỗ duy nhất chạm tới nó là
`known_text_hashes` (băm văn bản để chống rò rỉ).

- baseline: TF-IDF + LinearSVC với siêu tham số đã đóng băng của bản triển khai, fit lại trên
  train + phản hồi; bản được nhận lưu bằng skops, bọc hiệu chuẩn nhiệt độ fit trên validation như
  bản gốc.
- phobert: tiếp tục huấn luyện PhoBERT đa nhiệm đã triển khai `--max-steps` bước, CHỈ trên CPU; mỗi
  batch nửa là phản hồi (lấy có hoàn lại), nửa là mẫu phát lại từ train để mô hình không trôi.
  Cổng chấm bản FP32 (torch) trên validation, không phải int8: mô hình mới muốn thành bundle phải
  đi tiếp qua export ONNX int8 và cổng int8 (`make export calibrate artifacts`).

Không bao giờ ghi vào models/ hay bundle: mô hình mới nằm ở --out (mặc định
outputs/retrain/<loại>-<giờ>/). Mã thoát: 0 xong, 2 lỗi đầu vào, 3 mô hình mới bị từ chối, 4 khoá
máy (dùng chung với huấn luyện) đang do job khác giữ.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from edupulse.config import REPO_ROOT
from edupulse.labels import LABELS, TASKS
from edupulse.text import NORMALIZER_VERSION
from ml.common import (
    MpsBusyError,
    load_processed,
    mps_lock,
    normalizer_compatible,
    read_json,
    resolve_run,
    set_seed,
    settings,
    write_json,
)
from ml.plotting import vn
from ml.stats import macro_f1
from ml.train_baseline import drop_rare_terms, make_classifier, make_features, save_model

if TYPE_CHECKING:
    import pandas as pd

# Mô hình mới chỉ được nhận khi KHÔNG tác vụ nào tụt macro-F1 (trên validation) quá mức này.
ACCEPT_TOLERANCE = 0.0
_FLOAT_NOISE = 1e-12  # sai số dấu phẩy động khi so sánh hai macro-F1 bằng nhau

DEFAULT_STEPS = 20
BATCH_SIZE = 32
REPLAY_SHARE = 0.5  # phần batch lấy từ train để phát lại
# Cùng cấu hình huấn luyện gốc của `ml.train_phobert`.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
WARMUP_RATIO = 0.1

REQUIRED_COLUMNS = ("text_model", "label_s", "label_t")
LABEL_COLUMN = {"sentiment": "label_s", "topic": "label_t"}
# Tên cột nhãn trong file do API xuất (`edupulse.api.export.CORRECTED_HEADER`).
FILE_LABEL_COLUMN = {"sentiment": "label_sentiment", "topic": "label_topic"}
FILE_REQUIRED_COLUMNS = ("text_model", *FILE_LABEL_COLUMN.values())
FEEDBACK_COLUMNS = ["text_model", "label_s", "label_t", "source", "file", "line"]
REVIEWED_VALUES = frozenset({"true", "1", "yes"})
# Ký tự đầu ô mà bản xuất CSV chặn formula injection bằng cách chèn dấu nháy đơn phía trước:
# = + - @, dạng full-width của chúng (U+FF1D, U+FF0B, U+FF0D, U+FF20), TAB và CR.
GUARD_CHARS = frozenset("=+-@\t\r") | {chr(code) for code in (0xFF1D, 0xFF0B, 0xFF0D, 0xFF20)}

TASK_VI = {"sentiment": "cảm xúc", "topic": "chủ đề"}
READ_ROWS = "dòng đọc"
SKIP_UNREVIEWED = "chưa duyệt"
SKIP_MISSING = "thiếu câu hoặc nhãn"
REASON_EMPTY = "câu rỗng"
REASON_BAD_LABEL = "nhãn ngoài miền"
REASON_DUPLICATE = "trùng trong tệp"
REASON_KNOWN = "trùng dữ liệu gốc"


class InputError(ValueError):
    """Đầu vào của lần huấn luyện lại không dùng được: tham số, thư mục ghi hoặc dữ liệu."""


class FeedbackError(InputError):
    """Tệp phản hồi sai định dạng hoặc có nhãn ngoài miền; thông điệp nêu tệp và số dòng."""


# --- Đọc và làm sạch phản hồi -----------------------------------------------------------------


def unguard(value: str) -> str:
    """Bỏ dấu nháy đơn đầu ô mà bản xuất CSV thêm vào để chống formula injection.

    Chỉ bỏ khi ký tự đứng sau nháy là một ký tự nguy hiểm; nháy đơn có ở đầu một câu bình thường
    thì giữ nguyên.
    """
    if len(value) >= 2 and value[0] == "'" and value[1] in GUARD_CHARS:
        return value[1:]
    return value


def parse_label(value: str, task: str) -> int | None:
    """Nhãn dạng số 0..K-1 hoặc tên tiếng Anh (negative, lecturer, ...); None khi ô để trống."""
    labels = LABELS[task]
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        name = text.lower().replace(" ", "_").replace("-", "_")
        if name in labels:
            return labels.index(name)
    else:
        if number.is_integer() and 0 <= number < len(labels):
            return int(number)
    raise ValueError(
        f"{text!r} ngoài miền nhãn {TASK_VI[task]} (số 0..{len(labels) - 1} hoặc một trong "
        f"{', '.join(labels)})"
    )


def text_hash(text: str) -> str:
    """sha256 của câu đã strip: khoá để loại câu trùng nhau và trùng dữ liệu gốc."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _read_one(path: Path, tally: Counter[str]) -> pd.DataFrame:
    import pandas as pd

    if not path.is_file():
        raise FeedbackError(f"không thấy tệp phản hồi {path}")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise FeedbackError(f"{path.name}: không đọc được như CSV UTF-8 ({error})") from error
    raw.columns = [str(name).strip() for name in raw.columns]
    missing = [name for name in FILE_REQUIRED_COLUMNS if name not in raw.columns]
    if missing:
        raise FeedbackError(f"{path.name}: thiếu cột bắt buộc {', '.join(missing)}")

    has_reviewed = "reviewed" in raw.columns
    rows = []
    for position, record in enumerate(raw.to_dict("records")):
        line = position + 2  # dòng 1 là tiêu đề, khớp số dòng người dùng thấy trong Excel
        tally[READ_ROWS] += 1
        if has_reviewed and unguard(record["reviewed"]).strip().lower() not in REVIEWED_VALUES:
            tally[SKIP_UNREVIEWED] += 1
            continue
        text = unguard(record["text_model"]).strip()
        cells = {}
        for task, column in FILE_LABEL_COLUMN.items():
            try:
                cells[task] = parse_label(unguard(record[column]), task)
            except ValueError as error:
                raise FeedbackError(f"{path.name}, dòng {line}: cột {column}: {error}") from None
        if not text or cells["sentiment"] is None or cells["topic"] is None:
            tally[SKIP_MISSING] += 1
            continue
        source = unguard(record["source"]).strip() if "source" in raw.columns else ""
        rows.append((text, cells["sentiment"], cells["topic"], source, path.name, line))
    frame = pd.DataFrame(rows, columns=FEEDBACK_COLUMNS)
    return frame.astype({"label_s": "int64", "label_t": "int64"})


def read_feedback(
    paths: Iterable[str | Path] | str | Path, tally: Counter[str] | None = None
) -> pd.DataFrame:
    """Đọc các tệp phản hồi thành một bảng: text_model, label_s, label_t, source, file, line.

    Chỉ giữ dòng đã duyệt (reviewed là true/1/yes; không có cột reviewed thì coi mọi dòng là đã
    duyệt), bỏ dấu nháy chống injection ở đầu ô, bỏ dòng thiếu câu hoặc thiếu nhãn. Nhãn ngoài
    miền là lỗi: báo tệp và số dòng. `tally` (nếu có) được cộng số dòng đọc/bỏ theo lý do.
    """
    import pandas as pd

    if isinstance(paths, str | Path):
        paths = [paths]
    tally = Counter() if tally is None else tally
    frames = [_read_one(Path(path), tally) for path in paths]
    if not frames:
        raise FeedbackError("chưa chỉ định tệp phản hồi nào")
    return pd.concat(frames, ignore_index=True)


def _valid_label(value: Any, task: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer() and 0 <= number < len(LABELS[task])


def clean_feedback(
    frame: pd.DataFrame, known_hashes: Collection[str]
) -> tuple[pd.DataFrame, Counter[str]]:
    """Bỏ câu rỗng, nhãn ngoài miền, câu có băm nằm trong `known_hashes` và câu trùng nhau.

    Thứ tự xét từng dòng: rỗng, nhãn lỗi, trùng dữ liệu gốc, trùng trong tệp. Câu trùng nhau giữ
    dòng đầu tiên. Trả bảng đã làm sạch (câu đã strip, chỉ số 0..n-1) và số dòng bị loại theo lý do.
    """
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    keep: list[int] = []
    texts: list[str] = []
    rows = zip(frame["text_model"], frame["label_s"], frame["label_t"], strict=True)
    for position, (raw_text, label_s, label_t) in enumerate(rows):
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            reasons[REASON_EMPTY] += 1
        elif not (_valid_label(label_s, "sentiment") and _valid_label(label_t, "topic")):
            reasons[REASON_BAD_LABEL] += 1
        elif (digest := text_hash(text)) in known_hashes:
            reasons[REASON_KNOWN] += 1
        elif digest in seen:
            reasons[REASON_DUPLICATE] += 1
        else:
            seen.add(digest)
            keep.append(position)
            texts.append(text)
    cleaned = frame.iloc[keep].copy()
    cleaned["text_model"] = texts
    cleaned = cleaned.astype({"label_s": "int64", "label_t": "int64"})
    return cleaned.reset_index(drop=True), reasons


def known_text_hashes(
    splits: Sequence[str] = ("train", "validation", "test"),
) -> set[str]:
    """Băm sha256 của cột `text_rdr` ở từng split, để loại câu phản hồi trùng dữ liệu gốc.

    CHỖ DUY NHẤT của module này mở tập test (`allow_test=True`), và chỉ để BĂM văn bản chống rò
    rỉ: một câu đã nằm trong test thì không được đưa vào huấn luyện lại. Không tính chỉ số, không
    đọc nhãn test, không giữ lại nội dung; bỏ "test" khỏi `splits` thì nó không bị mở.
    """
    hashes: set[str] = set()
    for split in splits:
        texts = load_processed(split, allow_test=split == "test")["text_rdr"].tolist()
        hashes.update(text_hash(text) for text in texts)
    return hashes


# --- Baseline ---------------------------------------------------------------------------------


def load_deployed_baseline_hp(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Siêu tham số LinearSVC đã đóng băng của bản triển khai (chọn trên validation)."""
    path = path or REPO_ROOT / "results" / "metrics" / "val_baselines.json"
    if not path.is_file():
        raise FileNotFoundError(f"chưa có {path} — chạy `make baseline`")
    document = read_json(path)
    return {task: dict(document["tasks"][task]["linearsvc"]["hp"]) for task in TASKS}


def hp_text(hp: Mapping[str, Any]) -> str:
    """Siêu tham số LinearSVC dạng ngắn, số kiểu Việt: «C = 0,3» (kèm class_weight nếu có)."""
    parts = [f"C = {hp['C']:g}".replace(".", ",")]
    if hp.get("class_weight"):
        parts.append(f"class_weight = {hp['class_weight']}")
    return ", ".join(parts)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise InputError(f"bảng {name} thiếu cột {', '.join(missing)}")


def retrain_baseline(
    train: pd.DataFrame,
    feedback: pd.DataFrame,
    hp_by_task: Mapping[str, Mapping[str, Any]],
    weight: int = 1,
) -> dict[str, Pipeline]:
    """Fit lại TF-IDF rồi LinearSVC từng tác vụ trên train + phản hồi lặp `weight` lần.

    `train` cần cột text_rdr, sentiment, topic; `feedback` cần text_model, label_s, label_t và có
    thể rỗng (khi đó kết quả đúng là mô hình huấn luyện chỉ trên train). Siêu tham số truyền vào,
    không chọn lại. Các dòng phản hồi được lặp thật trước khi fit TF-IDF, nên n-gram chỉ có trong
    một câu phản hồi cũng qua ngưỡng min_df khi weight >= 2. Mọi tác vụ dùng chung một bộ đặc trưng
    như lúc huấn luyện gốc.
    """
    if weight < 1:
        raise InputError(f"feedback-weight phải >= 1, nhận {weight}")
    _require_columns(train, ["text_rdr", *TASKS], "train")
    _require_columns(feedback, REQUIRED_COLUMNS, "phản hồi")
    absent = [task for task in TASKS if task not in hp_by_task]
    if absent:
        raise InputError(f"thiếu siêu tham số cho {', '.join(absent)}")

    texts = [*train["text_rdr"].tolist(), *feedback["text_model"].tolist() * weight]
    features = make_features().fit(texts)
    drop_rare_terms(features)
    matrix = features.transform(texts)
    models: dict[str, Pipeline] = {}
    for task in TASKS:
        labels = np.concatenate(
            [
                train[task].to_numpy(dtype=np.int64),
                np.tile(feedback[LABEL_COLUMN[task]].to_numpy(dtype=np.int64), weight),
            ]
        )
        classifier = make_classifier("linearsvc", dict(hp_by_task[task])).fit(matrix, labels)
        models[task] = Pipeline([("features", features), ("clf", classifier)])
    return models


def accept(before: Mapping[str, float], after: Mapping[str, float]) -> tuple[bool, str]:
    """Nhận mô hình mới khi không tác vụ nào tụt macro-F1 (validation) quá ACCEPT_TOLERANCE.

    `before` và `after` là macro-F1 theo tác vụ. Bằng nhau thì nhận. Trả (nhận?, lý do).
    """
    if set(before) != set(after):
        raise ValueError(f"trước và sau phải cùng tập tác vụ: {sorted(before)} và {sorted(after)}")
    worse = []
    for task in before:
        drop = before[task] - after[task]
        if drop > ACCEPT_TOLERANCE + _FLOAT_NOISE:
            worse.append(
                f"{TASK_VI.get(task, task)} tụt {vn(drop, 4)} "
                f"({vn(before[task], 4)} -> {vn(after[task], 4)})"
            )
    limit = f"dung sai {vn(ACCEPT_TOLERANCE, 4)}"
    if worse:
        return False, f"{'; '.join(worse)}, vượt {limit}"
    moves = "; ".join(
        f"{TASK_VI.get(task, task)} {vn(before[task], 4)} -> {vn(after[task], 4)}"
        for task in before
    )
    return True, f"không tác vụ nào tụt macro-F1 ({limit}): {moves}"


# --- PhoBERT ----------------------------------------------------------------------------------


def default_out(kind: str) -> Path:
    """outputs/retrain/<loại>-<giờ>: nơi ghi mô hình mới khi không chỉ định --out."""
    return settings().outputs_dir / "retrain" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"


def assert_safe_output(out: Path, *protected: Path) -> Path:
    """Thư mục ghi mô hình mới không được trùng, nằm trong hay chứa nơi đang triển khai.

    Cấm models/ (PhoBERT và baseline), bundle đang phục vụ và `protected` (run gốc), và cấm ghi
    vào thư mục đã có nội dung, để lần huấn luyện lại không bao giờ ghi đè thứ gì.
    """
    target = out.expanduser().resolve()
    guarded = [settings().models_dir, settings().bundle_dir, *protected]
    for place in guarded:
        root = place.resolve()
        if target == root or root in target.parents or target in root.parents:
            raise InputError(
                f"không được ghi mô hình mới vào {root} (đang triển khai); chọn --out khác"
            )
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise InputError(f"{target} đã có nội dung; chọn --out khác, không ghi đè")
    return target


def predict_phobert(run: str | Path, texts: Sequence[str]) -> dict[str, np.ndarray]:
    """Logits FP32 trên CPU của một checkpoint PhoBERT, theo đúng thứ tự `texts`."""
    import torch

    from ml.phobert_model import MAX_LEN, MultiHeadPhoBERT, encode_ids, load_tokenizer
    from ml.train_phobert import predict

    model = MultiHeadPhoBERT.from_pretrained(run).eval()
    sequences = encode_ids(load_tokenizer(MAX_LEN), list(texts))
    return predict(model, sequences, torch.device("cpu"), tuple(model.config.task_heads))


def retrain_phobert(
    base_run: str | Path,
    feedback: pd.DataFrame,
    train: pd.DataFrame,
    steps: int = DEFAULT_STEPS,
    device: str = "cpu",
    replay: float = REPLAY_SHARE,
    seed: int = 0,
    out: Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Tiếp tục huấn luyện PhoBERT đa nhiệm `steps` bước trên CPU rồi lưu checkpoint mới ở `out`.

    Mỗi batch gồm `batch_size` câu: phần `replay` lấy ngẫu nhiên (không lặp trong batch) từ train
    để mô hình không trôi, phần còn lại lấy có hoàn lại từ phản hồi. Loss là tổng cross-entropy
    của các head; AdamW lr 2e-5, weight decay 0,01, clip 1,0, warmup 10% rồi giảm tuyến tính, như
    cấu hình gốc. `base_run` là 'best', tên run dưới models/phobert hoặc đường dẫn thư mục run; nó
    và models/ không bao giờ bị ghi. Trả bản ghi (số bước, thời gian, cỡ phản hồi/phát lại, đường
    dẫn, loss từng bước), cũng được lưu thành `retrain.json` cạnh checkpoint.
    """
    if device != "cpu":
        raise InputError(
            f"retrain chỉ chạy trên CPU (không dùng MPS hay CUDA), nhận --device {device}"
        )
    if steps < 1 or batch_size < 2:
        raise InputError(f"steps phải >= 1 và batch_size >= 2, nhận {steps} và {batch_size}")
    if not 0 <= replay < 1:
        raise InputError(f"replay phải thuộc [0, 1), nhận {replay}")
    if len(feedback) == 0:
        raise InputError("không có câu phản hồi nào để huấn luyện")
    _require_columns(feedback, REQUIRED_COLUMNS, "phản hồi")
    _require_columns(train, ["text_rdr", *TASKS], "train")

    run = resolve_run(str(base_run))
    history_path = run / "history.json"
    if history_path.is_file():
        found = read_json(history_path).get("normalizer_version")
        if not normalizer_compatible(found):
            raise InputError(
                f"{run.name} huấn luyện với bản chuẩn hoá {found!r} nhưng mã đang là "
                f"{NORMALIZER_VERSION!r} và không có bằng chứng bất biến; không trộn phản hồi "
                "mới vào mô hình cũ"
            )
    target = assert_safe_output(out or default_out("phobert"), run)

    import torch
    from torch import nn
    from transformers import get_linear_schedule_with_warmup

    from ml.phobert_model import MAX_LEN, MultiHeadPhoBERT, encode_ids, load_tokenizer
    from ml.train_phobert import optimizer_for, pad_batch

    n_replay = min(round(batch_size * replay), len(train))
    n_feedback = max(1, batch_size - n_replay)
    cpu = torch.device("cpu")
    tokenizer = load_tokenizer(MAX_LEN)
    feedback_ids = encode_ids(tokenizer, feedback["text_model"].tolist())
    train_ids = encode_ids(tokenizer, train["text_rdr"].tolist())

    set_seed(seed)
    model = MultiHeadPhoBERT.from_pretrained(run)
    tasks = tuple(model.config.task_heads)
    feedback_labels = {t: feedback[LABEL_COLUMN[t]].to_numpy(dtype=np.int64) for t in tasks}
    train_labels = {t: train[t].to_numpy(dtype=np.int64) for t in tasks}

    rng = np.random.default_rng(seed)
    feedback_pick = rng.integers(0, len(feedback), size=(steps, n_feedback))
    replay_pick = [rng.choice(len(train), size=n_replay, replace=False) for _ in range(steps)]

    model.train()
    optimizer = optimizer_for(model, LEARNING_RATE, WEIGHT_DECAY, fused=False)
    warmup_steps = int(WARMUP_RATIO * steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    started = time.time()
    for step in range(steps):
        rows, replayed = feedback_pick[step], replay_pick[step]
        sequences = [feedback_ids[i] for i in rows] + [train_ids[i] for i in replayed]
        ids, mask = pad_batch(sequences, cpu)
        logits = model(input_ids=ids, attention_mask=mask)
        loss = sum(
            loss_fn(
                logits[task],
                torch.from_numpy(
                    np.concatenate([feedback_labels[task][rows], train_labels[task][replayed]])
                ),
            )
            for task in tasks
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    seconds = time.time() - started

    target.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(target)
    record = {
        "kind": "phobert",
        "base_run": run.name,
        "base_run_path": str(run),
        "out": str(target),
        "device": "cpu",
        "tasks": list(tasks),
        "steps": steps,
        "batch_size": n_feedback + n_replay,
        "feedback_size": len(feedback),
        "feedback_per_batch": n_feedback,
        "replay_per_batch": n_replay,
        "replay_pool": len(train),
        "replay_share": replay,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "seconds": round(seconds, 1),
        "losses": losses,
        "normalizer_version": NORMALIZER_VERSION,
    }
    write_json(target / "retrain.json", record)
    return record


# --- Dòng lệnh --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--feedback", nargs="+", type=Path, required=True, help="CSV nhãn đã duyệt do API xuất"
    )
    parser.add_argument("--model", choices=["baseline", "phobert"], default="baseline")
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_STEPS, help="phobert: số bước huấn luyện tiếp"
    )
    parser.add_argument(
        "--feedback-weight",
        type=int,
        default=1,
        help="baseline: lặp mỗi câu phản hồi từng này lần khi fit",
    )
    parser.add_argument(
        "--out", type=Path, help="thư mục ghi mô hình mới (mặc định outputs/retrain)"
    )
    parser.add_argument("--device", default="cpu", help="chỉ hỗ trợ cpu")
    parser.add_argument("--base-run", default="best", help="phobert: best | tên run | đường dẫn")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _ratio(value: float) -> str:
    return vn(value * 100, 1)


def _print_comparison(before: Mapping[str, float], after: Mapping[str, float], n_val: int) -> None:
    print(f"Macro-F1 trên validation (n = {vn(n_val, 0)}), phần trăm:")
    for task in before:
        delta = (after[task] - before[task]) * 100
        sign = "+" if delta > 0 else ""
        print(
            f"  {TASK_VI[task]:<7}: trước {_ratio(before[task])} -> sau {_ratio(after[task])} "
            f"({sign}{vn(delta, 1)})"
        )


def _print_feedback_summary(
    tally: Counter[str], rejected: Counter[str], cleaned: pd.DataFrame
) -> None:
    reviewed = tally[READ_ROWS] - tally[SKIP_UNREVIEWED] - tally[SKIP_MISSING]
    print(
        f"Phản hồi: đọc {tally[READ_ROWS]} dòng · chưa duyệt {tally[SKIP_UNREVIEWED]} · "
        f"thiếu câu hoặc nhãn {tally[SKIP_MISSING]} · còn {reviewed} dòng đã duyệt"
    )
    detail = ", ".join(f"{reason} {count}" for reason, count in rejected.most_common())
    print(
        f"Làm sạch: giữ {len(cleaned)} câu · loại {sum(rejected.values())}"
        + (f" ({detail})" if detail else "")
    )
    sources = Counter(cleaned["source"].replace("", "không rõ"))
    if sources:
        print("Nguồn: " + " · ".join(f"{name} {count}" for name, count in sources.most_common()))


def _macro_by_task(y: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {task: macro_f1(y[task], pred[task], len(LABELS[task])) for task in TASKS}


def _run_baseline(
    args: argparse.Namespace,
    cleaned: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    out: Path,
) -> int:
    hp = load_deployed_baseline_hp()
    print(
        "Mô hình: TF-IDF + LinearSVC, siêu tham số đã đóng băng: "
        + " · ".join(f"{TASK_VI[t]} {hp_text(hp[t])}" for t in TASKS)
    )
    print("«Trước» = huấn luyện lại chỉ trên train bằng đúng mã và siêu tham số này.")
    val_texts = val["text_rdr"].tolist()
    y = {task: val[task].to_numpy(dtype=np.int64) for task in TASKS}
    before_models = retrain_baseline(train, cleaned.iloc[0:0], hp)
    after_models = retrain_baseline(train, cleaned, hp, weight=args.feedback_weight)
    before = _macro_by_task(y, {t: before_models[t].predict(val_texts) for t in TASKS})
    after = _macro_by_task(y, {t: after_models[t].predict(val_texts) for t in TASKS})
    _print_comparison(before, after, len(val))
    accepted, reason = accept(before, after)
    if not accepted:
        print(f"Quyết định: TỪ CHỐI mô hình mới, không ghi gì. {reason}")
        return 3

    out.mkdir(parents=True, exist_ok=True)
    files = []
    for task in TASKS:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(after_models[task]), method="temperature"
        )
        calibrated.fit(val_texts, y[task])
        path = out / f"{task}_linearsvc.skops"
        save_model(calibrated, path)
        files.append(path.name)
    write_json(
        out / "retrain.json",
        {
            "kind": "baseline",
            "out": str(out),
            "feedback_size": len(cleaned),
            "feedback_weight": args.feedback_weight,
            "hp": hp,
            "val_macro_f1": {"before": before, "after": after},
            "decision": {"accepted": True, "reason": reason},
            "files": files,
            "normalizer_version": NORMALIZER_VERSION,
        },
    )
    print(f"Quyết định: NHẬN mô hình mới. {reason}")
    print(f"Đã ghi {', '.join(files)} vào {out}")
    return 0


def _run_phobert(
    args: argparse.Namespace,
    cleaned: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    out: Path,
) -> int:
    base = resolve_run(args.base_run)
    val_texts = val["text_rdr"].tolist()
    y = {task: val[task].to_numpy(dtype=np.int64) for task in TASKS}

    def val_macro(checkpoint: Path) -> dict[str, float]:
        logits = predict_phobert(checkpoint, val_texts)
        return _macro_by_task(y, {t: logits[t].argmax(axis=1) for t in TASKS})

    print(f"Mô hình: PhoBERT đa nhiệm {base.name}, tiếp tục {args.max_steps} bước trên CPU")
    before = val_macro(base)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        record = retrain_phobert(
            base,
            cleaned,
            train,
            steps=args.max_steps,
            device=args.device,
            seed=args.seed,
            out=candidate / "model",
        )
        after = val_macro(candidate / "model")
        _print_comparison(before, after, len(val))
        accepted, reason = accept(before, after)
        if not accepted:
            print(f"Quyết định: TỪ CHỐI mô hình mới, không ghi gì. {reason}")
            return 3
        os.replace(candidate / "model", out)
        record |= {"out": str(out), "val_macro_f1": {"before": before, "after": after}}
        record["decision"] = {"accepted": True, "reason": reason}
        write_json(out / "retrain.json", record)
    finally:
        shutil.rmtree(candidate, ignore_errors=True)
    print(f"Quyết định: NHẬN mô hình mới. {reason}")
    print(f"Đã ghi checkpoint vào {out} ({record['steps']} bước, {record['seconds']} giây)")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.max_steps < 1 or args.feedback_weight < 1:
        raise InputError("--max-steps và --feedback-weight phải >= 1")
    if args.device != "cpu":
        raise InputError(f"chỉ hỗ trợ --device cpu (không dùng MPS hay CUDA), nhận {args.device}")
    # Báo sớm nếu thư mục ghi không hợp lệ, trước khi tốn thời gian huấn luyện.
    out = assert_safe_output(args.out or default_out(args.model))

    tally: Counter[str] = Counter()
    frame = read_feedback(args.feedback, tally)
    cleaned, rejected = clean_feedback(frame, known_text_hashes())
    _print_feedback_summary(tally, rejected, cleaned)
    if cleaned.empty:
        raise InputError("không còn câu phản hồi hợp lệ nào sau khi làm sạch")

    train = load_processed("train")
    val = load_processed("validation")
    if args.model == "baseline":
        return _run_baseline(args, cleaned, train, val, out)
    # PhoBERT chiếm hết lõi CPU: giữ khoá toàn máy như mọi lần huấn luyện thật, không chờ.
    with mps_lock(note="retrain phobert"):
        return _run_phobert(args, cleaned, train, val, out)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (InputError, FileNotFoundError) as error:
        print(f"Lỗi đầu vào: {error}", file=sys.stderr)
        return 2
    except MpsBusyError as error:
        print(error, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
