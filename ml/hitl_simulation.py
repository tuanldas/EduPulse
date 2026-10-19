"""Mô phỏng vòng người duyệt (HITL) bằng LinearSVC trên CPU, chỉ dùng train và validation.

    python -m ml.hitl_simulation [--skip-phobert] [--seed 0]

Ghi `results/metrics/hitl_retrain.json` (đủ số liệu, tham số, kích thước, seed, phiên bản mã, thời
gian; không chứa câu gốc) và `results/tables/hitl_retrain.md` (báo cáo tiếng Việt). Các bước:

1. Chia validation cố định (seed, phân tầng theo nhãn cảm xúc) thành nửa DUYỆT R và nửa ĐÁNH GIÁ E.
2. Mô hình «trước» = TF-IDF + LinearSVC với siêu tham số đã đóng băng của bản triển khai, fit chỉ
   trên train bằng đúng `ml.retrain.retrain_baseline` với phản hồi rỗng; được kiểm là cho nhãn dự
   đoán trùng với bản đã lưu trong models/baseline.
3. Gắn cờ trên R theo ngân sách 10% bằng mô hình «trước» (luật của `edupulse.scoring`: phân vị độ
   tin cậy trong R, điểm cờ là min của hai tác vụ, cờ ⌈10%·|R|⌉ câu điểm thấp nhất).
4. Người duyệt MÔ PHỎNG: câu cờ nhận NHÃN GỐC của cả hai tác vụ làm nhãn đã sửa (nhãn gốc coi như
   đúng, không có người duyệt thật).
5. Ba nhánh «sau», mỗi nhánh retrain bằng `retrain_baseline` cùng siêu tham số: câu cờ; đối chứng
   ngẫu nhiên (cùng số câu, lặp 20 lần); trần trên (toàn bộ R).
6. Đánh giá mọi nhánh trên E; nhánh cờ so với «trước» bằng McNemar chính xác và bootstrap ghép cặp.
7. Smoke PhoBERT: tiếp tục huấn luyện 20 bước trên CPU từ run đã triển khai với chính các câu cờ.

Tập test không bao giờ được mở ở đây.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import train_test_split

from edupulse.config import DEFAULT_BUDGET, REPO_ROOT
from edupulse.labels import LABELS, TASKS
from edupulse.scoring import (
    confidence,
    efficiency,
    flag_score,
    make_cdf,
    quantile_rank,
    rank_batch,
    softmax_t,
)
from edupulse.text import NORMALIZER_VERSION
from ml import retrain
from ml.common import (
    MpsBusyError,
    load_processed,
    mps_lock,
    prediction_metrics,
    processed_manifest,
    resolve_run,
    settings,
    sha256_file,
    write_json,
)
from ml.freeze import git_dirty, git_head
from ml.plotting import vn
from ml.predictors import baseline_path, baseline_preds
from ml.retrain import (
    DEFAULT_STEPS,
    hp_text,
    load_deployed_baseline_hp,
    predict_phobert,
    retrain_baseline,
    retrain_phobert,
)
from ml.stats import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED, paired_difference

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_VERSION = 1
SEED = 0
BUDGET_PCT = DEFAULT_BUDGET  # 10% câu của R
RANDOM_REPEATS = 20  # đối chứng ngẫu nhiên: seed 0..19
PHOBERT_STEPS = DEFAULT_STEPS
SIGNIFICANCE = 0.05
FOCUS_CLASS = {"sentiment": "neutral", "topic": "others"}  # lớp ít câu nhất, khó nhất
TASK_NAME = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}
FOCUS_NAME = {"sentiment": "trung tính", "topic": "«khác»"}
METRICS_PATH = Path("metrics") / "hitl_retrain.json"
TABLE_PATH = Path("tables") / "hitl_retrain.md"
CODE_FILES = ("ml/hitl_simulation.py", "ml/retrain.py", "ml/train_baseline.py", "ml/stats.py")

Logger = Callable[[str], None]


def _silent(message: str) -> None:
    del message


def log_line(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- Chia dữ liệu, gắn cờ, chỉ số -------------------------------------------------------------


def split_validation(val: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Vị trí dòng của nửa duyệt R và nửa đánh giá E: rời nhau, phân tầng theo nhãn cảm xúc.

    Cố định theo `seed`. Số dòng lẻ thì E hơn R một dòng (1.583 dòng: R 791, E 792).
    """
    positions = np.arange(len(val))
    review, evaluation = train_test_split(
        positions,
        test_size=0.5,
        stratify=val["sentiment"].to_numpy(),
        random_state=seed,
        shuffle=True,
    )
    return np.sort(review), np.sort(evaluation)


def select_flags(
    conf_sentiment: np.ndarray, conf_topic: np.ndarray, pct: float = BUDGET_PCT
) -> tuple[np.ndarray, np.ndarray]:
    """Cờ ⌈pct·n/100⌉ câu điểm thấp nhất của một lô; trả (mặt nạ cờ, điểm cờ).

    Phân vị của độ tin cậy lấy trong chính lô này (bảng 101 điểm rồi nội suy), điểm cờ là phân vị
    thấp hơn của hai tác vụ; điểm bằng nhau xử lý theo thứ tự dòng.
    """
    q_sentiment = quantile_rank(conf_sentiment, make_cdf(conf_sentiment))
    q_topic = quantile_rank(conf_topic, make_cdf(conf_topic))
    scores = flag_score(q_sentiment, q_topic)
    return rank_batch(scores, pct), scores


def decision_confidence(pipeline: Any, texts: list[str]) -> np.ndarray:
    """Độ tin cậy = xác suất lớn nhất của softmax trên điểm quyết định của LinearSVC (T = 1)."""
    return confidence(softmax_t(pipeline.decision_function(texts)))


def labels_of(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {task: frame[task].to_numpy(dtype=np.int64) for task in TASKS}


def feedback_from(rows: pd.DataFrame) -> pd.DataFrame:
    """Nhãn gốc của các dòng validation đóng vai nhãn người duyệt đã sửa (bảng phản hồi)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "text_model": rows["text_rdr"].tolist(),
            "label_s": rows["sentiment"].to_numpy(dtype=np.int64),
            "label_t": rows["topic"].to_numpy(dtype=np.int64),
        }
    )


def predict_all(models: Mapping[str, Any], texts: list[str]) -> dict[str, np.ndarray]:
    return {task: np.asarray(models[task].predict(texts), dtype=np.int64) for task in TASKS}


def summarize(
    y: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], tasks: tuple[str, ...] = TASKS
) -> dict[str, dict[str, Any]]:
    """Accuracy, macro-F1, weighted-F1 và F1 từng lớp của mỗi tác vụ."""
    summary = {}
    for task in tasks:
        metrics = prediction_metrics(y[task], pred[task], LABELS[task])
        summary[task] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class_f1": [item["f1"] for item in metrics["per_class"]],
        }
    return summary


def aggregate_runs(runs: list[dict[str, dict[str, Any]]]) -> tuple[dict, dict]:
    """Trung bình và độ lệch chuẩn mẫu (ddof = 1) của các lần chạy, theo tác vụ và chỉ số."""
    mean: dict[str, dict[str, Any]] = {}
    std: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        mean[task], std[task] = {}, {}
        for key in ("accuracy", "macro_f1", "weighted_f1", "per_class_f1"):
            values = np.array([run[task][key] for run in runs], dtype=np.float64)
            spread = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(values.shape[1:])
            mean[task][key] = values.mean(axis=0).tolist()
            std[task][key] = spread.tolist()
    return mean, std


def flagging_report(flags: np.ndarray, correct: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Mô hình «trước» làm gì trên R: độ chính xác phần cờ so với phần còn lại, lỗi bị bắt."""
    ok_sentiment, ok_topic = correct["sentiment"], correct["topic"]
    both_ok = ok_sentiment & ok_topic
    n_flagged, n_review = int(flags.sum()), int(flags.size)
    result = efficiency(flags, ok_sentiment, ok_topic)

    def share(mask: np.ndarray, where: np.ndarray) -> float | None:
        return float(mask[where].mean()) if where.any() else None

    return {
        "n_review": n_review,
        "n_flagged": n_flagged,
        "flagged_share": n_flagged / n_review,
        "wrong_any_share_flagged": share(~both_ok, flags),
        "wrong_any_share_unflagged": share(~both_ok, ~flags),
        "accuracy_flagged": {task: share(correct[task], flags) for task in TASKS},
        "accuracy_unflagged": {task: share(correct[task], ~flags) for task in TASKS},
        "accuracy_both_flagged": share(both_ok, flags),
        "accuracy_both_unflagged": result["accept_accuracy_both"],
        "accuracy_both_all": result["all_accuracy_both"],
        "errors": {
            task: {
                "total": int((~correct[task]).sum()),
                "in_flagged": int((~correct[task] & flags).sum()),
                "caught_share": result[f"caught_{task}_errors"],
            }
            for task in TASKS
        },
    }


def gate_decision(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Quy tắc nhận mô hình mới của `ml.retrain` áp lên macro-F1 trước/sau của từng tác vụ."""
    accepted, reason = retrain.accept(
        {task: before[task]["macro_f1"] for task in before},
        {task: after[task]["macro_f1"] for task in before},
    )
    return {"accepted": accepted, "reason": reason}


def compare_with_deployed(
    pred: Mapping[str, np.ndarray], deployed: Mapping[str, np.ndarray] | None
) -> dict[str, Any]:
    """Nhãn dự đoán của mô hình «trước» trùng bao nhiêu với bản baseline đã lưu (cả validation)."""
    if deployed is None:
        return {"checked": False, "identical": None, "match_rate": None, "n": None}
    rates = {task: float((pred[task] == deployed[task]).mean()) for task in TASKS}
    return {
        "checked": True,
        "identical": all(rate == 1.0 for rate in rates.values()),
        "match_rate": rates,
        "n": int(len(pred[TASKS[0]])),
    }


# --- Mô phỏng ---------------------------------------------------------------------------------


def run_simulation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    hp_by_task: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = SEED,
    budget_pct: float = BUDGET_PCT,
    random_repeats: int = RANDOM_REPEATS,
    bootstrap_rounds: int = BOOTSTRAP_ROUNDS,
    deployed_pred: Mapping[str, np.ndarray] | None = None,
    phobert_run: str | Path | None = None,
    phobert_steps: int = PHOBERT_STEPS,
    log: Logger = _silent,
) -> dict[str, Any]:
    """Toàn bộ vòng mô phỏng trên `train` và `val` (cột text_rdr, sentiment, topic); trả kết quả.

    `deployed_pred` là nhãn dự đoán trên TOÀN `val` của baseline đã lưu (None thì không kiểm).
    `phobert_run` (thư mục run) bật bước smoke PhoBERT; None thì bỏ qua. Kết quả không chứa câu.
    """
    started = time.time()
    timing: dict[str, float] = {}

    review_pos, eval_pos = split_validation(val, seed)
    review = val.iloc[review_pos].reset_index(drop=True)
    evaluation = val.iloc[eval_pos].reset_index(drop=True)
    y_review, y_eval = labels_of(review), labels_of(evaluation)
    log(f"chia validation: R {len(review)} câu · E {len(evaluation)} câu")

    def fit(rows: pd.DataFrame) -> dict[str, Any]:
        return retrain_baseline(train, feedback_from(rows), hp_by_task)

    def timed(name: str, begin: float) -> None:
        timing[name] = round(time.time() - begin, 2)

    step = time.time()
    before_models = fit(review.iloc[0:0])
    pred_val = predict_all(before_models, val["text_rdr"].tolist())
    pred_review = {task: pred_val[task][review_pos] for task in TASKS}
    pred_eval = {task: pred_val[task][eval_pos] for task in TASKS}
    deployed_check = compare_with_deployed(pred_val, deployed_pred)
    timed("before", step)
    log(f"mô hình trước: trùng bản đã lưu = {deployed_check['identical']}")

    step = time.time()
    review_texts = review["text_rdr"].tolist()
    confidences = {task: decision_confidence(before_models[task], review_texts) for task in TASKS}
    flags, _ = select_flags(confidences["sentiment"], confidences["topic"], budget_pct)
    correct = {task: pred_review[task] == y_review[task] for task in TASKS}
    flagging = flagging_report(flags, correct)
    flagging["flagged_val_positions"] = review_pos[flags].tolist()
    n_flagged = flagging["n_flagged"]
    timed("flagging", step)
    log(f"gắn cờ {n_flagged}/{len(review)} câu của R")

    eval_texts = evaluation["text_rdr"].tolist()

    def branch(rows: pd.DataFrame) -> dict[str, np.ndarray]:
        return predict_all(fit(rows), eval_texts)

    step = time.time()
    pred_flagged = branch(review[flags])
    timed("flagged", step)

    step = time.time()
    random_runs = []
    for repeat_seed in range(random_repeats):
        pick = np.random.default_rng(repeat_seed).choice(len(review), size=n_flagged, replace=False)
        pred = branch(review.iloc[np.sort(pick)])
        random_runs.append({"seed": repeat_seed, "metrics": summarize(y_eval, pred)})
    timed("random", step)
    log(f"đối chứng ngẫu nhiên: {random_repeats} lần")

    step = time.time()
    pred_all = branch(review)
    timed("all_review", step)

    step = time.time()
    branches = {
        "before": summarize(y_eval, pred_eval),
        "flagged": summarize(y_eval, pred_flagged),
        "all_review": summarize(y_eval, pred_all),
    }
    mean, std = aggregate_runs([run["metrics"] for run in random_runs])
    branches["random"] = {
        "n_sentences": n_flagged,
        "seeds": [run["seed"] for run in random_runs],
        "runs": random_runs,
        "mean": mean,
        "std": std,
    }
    tests: dict[str, dict[str, Any]] = {}
    for name, pred in (("flagged_vs_before", pred_flagged), ("all_review_vs_before", pred_all)):
        tests[name] = {
            task: paired_difference(
                y_eval[task],
                pred[task],
                pred_eval[task],
                len(LABELS[task]),
                rounds=bootstrap_rounds,
            )
            for task in TASKS
        }
    tests["flagged_vs_random"] = {}
    for task in TASKS:
        random_f1 = np.array([run["metrics"][task]["macro_f1"] for run in random_runs])
        flagged_f1 = branches["flagged"][task]["macro_f1"]
        tests["flagged_vs_random"][task] = {
            "macro_f1_flagged": flagged_f1,
            "macro_f1_random_mean": float(random_f1.mean()),
            "macro_f1_random_std": float(random_f1.std(ddof=1)) if len(random_f1) > 1 else 0.0,
            "macro_f1_random_min": float(random_f1.min()),
            "macro_f1_random_max": float(random_f1.max()),
            "delta_macro_f1": float(flagged_f1 - random_f1.mean()),
            "runs_flagged_at_least": int((flagged_f1 >= random_f1).sum()),
            "n_runs": int(random_f1.size),
            # p thực nghiệm một phía kiểu hoán vị: nếu cờ không hơn ngẫu nhiên thì nhánh cờ chỉ là
            # thêm một lần rút nữa, nên thứ hạng của nó trong n + 1 lần rút là đều.
            "p_empirical": float((1 + (random_f1 >= flagged_f1).sum()) / (random_f1.size + 1)),
        }
    timed("evaluation", step)

    if phobert_run is None:
        smoke: dict[str, Any] = {"skipped": True, "reason": "bỏ qua theo yêu cầu (--skip-phobert)"}
    else:
        step = time.time()
        smoke = phobert_smoke(
            phobert_run,
            train,
            feedback_from(review[flags]),
            evaluation,
            steps=phobert_steps,
            seed=seed,
            log=log,
        )
        timed("phobert_smoke", step)

    timing["total"] = round(time.time() - started, 2)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "hitl_simulation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "simulated_reviewer": (
            "MÔ PHỎNG: câu cờ nhận nhãn gốc của cả hai tác vụ làm nhãn đã sửa; "
            "không có người duyệt thật và nhãn gốc được coi là đúng"
        ),
        "parameters": {
            "seed": seed,
            "budget_pct": budget_pct,
            "random_repeats": random_repeats,
            "random_seeds": list(range(random_repeats)),
            "bootstrap_rounds": bootstrap_rounds,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "model": "TF-IDF + LinearSVC",
            "hp": {task: dict(hp_by_task[task]) for task in TASKS},
            "confidence": (
                "xác suất lớn nhất của softmax(decision_function), T = 1, không hiệu chuẩn"
            ),
            "flag_score": "min(phân vị cảm xúc, phân vị chủ đề), phân vị tính trong chính R",
            "split": (
                "train_test_split(test_size=0.5, stratify=cảm xúc, random_state=seed): "
                "R là phần «train» (⌊n/2⌋ câu), E là phần «test» (⌈n/2⌉ câu); "
                "cả hai sắp theo vị trí dòng trong validation"
            ),
            "labels": {task: list(LABELS[task]) for task in TASKS},
            "significance_level": SIGNIFICANCE,
        },
        "data": {
            "normalizer_version": NORMALIZER_VERSION,
            "n_train": len(train),
            "n_val": len(val),
            "n_review": len(review),
            "n_eval": len(evaluation),
            "review_support": {
                t: np.bincount(y_review[t], minlength=len(LABELS[t])).tolist() for t in TASKS
            },
            "eval_support": {
                t: np.bincount(y_eval[t], minlength=len(LABELS[t])).tolist() for t in TASKS
            },
        },
        "before_matches_deployed_baseline": deployed_check["identical"],
        "deployed_check": deployed_check,
        "flagging": flagging,
        "branches": branches,
        "tests": tests,
        "accept_rule": {
            "rule": (
                "không tác vụ nào tụt macro-F1 quá dung sai; áp lên E (nửa validation chưa dùng để "
                "huấn luyện), không phải toàn bộ validation như `ml.retrain`"
            ),
            "tolerance": retrain.ACCEPT_TOLERANCE,
            "flagged": gate_decision(branches["before"], branches["flagged"]),
            "all_review": gate_decision(branches["before"], branches["all_review"]),
        },
        "phobert_smoke": smoke,
        "timing_seconds": timing,
    }


# --- Smoke PhoBERT ----------------------------------------------------------------------------


def phobert_smoke(
    base_run: str | Path,
    train: pd.DataFrame,
    feedback: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    steps: int = PHOBERT_STEPS,
    seed: int = SEED,
    log: Logger = _silent,
) -> dict[str, Any]:
    """Tiếp tục huấn luyện PhoBERT `steps` bước (CPU) bằng `feedback` rồi đo macro-F1 trên E.

    Dự đoán FP32 bằng torch trên CPU cho cả trước và sau (không phải bản int8 triển khai).
    Checkpoint mới nằm ở thư mục tạm và bị xoá ngay sau khi đo. Chỉ chứng minh đường chạy: vài
    chục bước không kỳ vọng đổi đáng kể.
    """
    started = time.time()
    run = resolve_run(str(base_run))
    texts, y = evaluation["text_rdr"].tolist(), labels_of(evaluation)

    log(f"PhoBERT {run.name}: dự đoán trên E trước khi huấn luyện tiếp")
    before_logits = predict_phobert(run, texts)
    scratch = settings().outputs_dir / "retrain"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hitl-", dir=scratch) as tmp:
        log(f"PhoBERT: huấn luyện tiếp {steps} bước trên CPU với {len(feedback)} câu cờ")
        record = retrain_phobert(
            run, feedback, train, steps=steps, device="cpu", seed=seed, out=Path(tmp) / "model"
        )
        log(f"PhoBERT: xong {steps} bước trong {record['seconds']} giây, dự đoán lại trên E")
        after_logits = predict_phobert(record["out"], texts)

    tasks = tuple(before_logits)
    before_pred = {task: before_logits[task].argmax(axis=1) for task in tasks}
    after_pred = {task: after_logits[task].argmax(axis=1) for task in tasks}
    before, after = summarize(y, before_pred, tasks), summarize(y, after_pred, tasks)
    return {
        "skipped": False,
        "base_run": record["base_run"],
        "device": "cpu",
        "precision": "fp32 (torch), không phải bản int8 ONNX đang triển khai",
        "steps": record["steps"],
        "batch_size": record["batch_size"],
        "feedback_size": record["feedback_size"],
        "feedback_per_batch": record["feedback_per_batch"],
        "replay_per_batch": record["replay_per_batch"],
        "replay_pool": record["replay_pool"],
        "learning_rate": record["learning_rate"],
        "weight_decay": record["weight_decay"],
        "clip_norm": record["clip_norm"],
        "warmup_steps": record["warmup_steps"],
        "seed": record["seed"],
        "n_eval": len(evaluation),
        "train_seconds": record["seconds"],
        "total_seconds": round(time.time() - started, 1),
        "losses": record["losses"],
        "before": before,
        "after": after,
        "delta_macro_f1": {t: after[t]["macro_f1"] - before[t]["macro_f1"] for t in tasks},
        "label_changes": {t: int((before_pred[t] != after_pred[t]).sum()) for t in tasks},
        "accept_rule": gate_decision(before, after),
        "note": (
            "Chỉ chứng minh đường chạy của `ml.retrain --model phobert`; vài chục bước trên vài "
            "chục câu không kỳ vọng thay đổi đáng kể."
        ),
    }


# --- Báo cáo ----------------------------------------------------------------------------------


def pct(value: float, digits: int = 1) -> str:
    return vn(value * 100, digits)


def percent_or_dash(value: float | None) -> str:
    return "-" if value is None else pct(value)


def signed(value: float, digits: int = 1) -> str:
    """Số có dấu +/-, không dấu khi làm tròn về 0."""
    text = vn(abs(value), digits)
    if round(abs(value), digits) == 0:
        return text
    return ("+" if value > 0 else "-") + text


def points(delta: float, digits: int = 1) -> str:
    """Hiệu của hai tỉ lệ, tính bằng điểm phần trăm, có dấu."""
    return signed(delta * 100, digits)


def p_text(p_value: float) -> str:
    return "< 0,001" if p_value < 0.001 else vn(p_value, 3)


def interval_text(paired: Mapping[str, Any]) -> str:
    return f"[{points(paired['ci95_low'])}; {points(paired['ci95_high'])}]"


def significance(paired: Mapping[str, Any]) -> tuple[bool, bool]:
    """(khoảng bootstrap 95% của ΔmF1 loại trừ 0?, McNemar p < 0,05?)."""
    excludes_zero = paired["ci95_low"] > 0 or paired["ci95_high"] < 0
    return excludes_zero, paired["mcnemar_p"] < SIGNIFICANCE


def verdict(paired: Mapping[str, Any]) -> str:
    """Một câu mô tả hướng thay đổi và mức ý nghĩa của một phép so sánh ghép cặp."""
    delta = paired["delta_macro_f1"]
    if round(abs(delta) * 100, 1) == 0:
        direction = "gần như không đổi (dưới 0,1 điểm)"
    else:
        direction = f"{'tăng' if delta > 0 else 'giảm'} {vn(abs(delta) * 100, 1)} điểm"
    boot, mcnemar = significance(paired)
    evidence = (
        f"KTC 95% {interval_text(paired)}, p bootstrap {p_text(paired['bootstrap_p'])}; "
        f"McNemar p {p_text(paired['mcnemar_p'])}"
    )
    if boot and mcnemar:
        meaning = "có ý nghĩa thống kê ở cả hai kiểm định"
    elif boot or mcnemar:
        which = "bootstrap" if boot else "McNemar"
        meaning = f"chỉ kiểm định {which} có ý nghĩa, bằng chứng yếu"
    else:
        meaning = "không có ý nghĩa thống kê"
    return f"macro-F1 {direction} ({evidence}): {meaning}"


def _metric_cells(metrics: Mapping[str, Any], task: str) -> list[str]:
    focus = LABELS[task].index(FOCUS_CLASS[task])
    return [
        pct(metrics["accuracy"]),
        pct(metrics["macro_f1"]),
        pct(metrics["weighted_f1"]),
        pct(metrics["per_class_f1"][focus]),
    ]


def _spread_cells(mean: Mapping[str, Any], std: Mapping[str, Any], task: str) -> list[str]:
    focus = LABELS[task].index(FOCUS_CLASS[task])
    values = [
        (mean["accuracy"], std["accuracy"]),
        (mean["macro_f1"], std["macro_f1"]),
        (mean["weighted_f1"], std["weighted_f1"]),
        (mean["per_class_f1"][focus], std["per_class_f1"][focus]),
    ]
    return [f"{pct(m)} ± {pct(s)}" for m, s in values]


def results_table(result: Mapping[str, Any]) -> str:
    branches, flagging = result["branches"], result["flagging"]
    n_flagged, n_review = vn(flagging["n_flagged"], 0), vn(flagging["n_review"], 0)
    random = branches["random"]
    rows = [
        "| Nhánh | Cảm xúc · Acc | Cảm xúc · mF1 | Cảm xúc · wF1 | F1 trung tính "
        "| Chủ đề · Acc | Chủ đề · mF1 | Chủ đề · wF1 | F1 «khác» |",
        "|---|" + "---:|" * 8,
    ]

    def line(name: str, sentiment: list[str], topic: list[str]) -> str:
        return f"| {name} | " + " | ".join([*sentiment, *topic]) + " |"

    def plain(name: str, key: str) -> str:
        return line(
            name,
            _metric_cells(branches[key]["sentiment"], "sentiment"),
            _metric_cells(branches[key]["topic"], "topic"),
        )

    rows += [
        plain("Trước (chỉ train)", "before"),
        plain(f"Sau · câu cờ ({n_flagged} câu)", "flagged"),
        line(
            f"Sau · ngẫu nhiên ({n_flagged} câu, TB ± ĐLC, {len(random['runs'])} lần)",
            _spread_cells(random["mean"]["sentiment"], random["std"]["sentiment"], "sentiment"),
            _spread_cells(random["mean"]["topic"], random["std"]["topic"], "topic"),
        ),
        plain(f"Trần trên · duyệt hết R ({n_review} câu)", "all_review"),
    ]
    return "\n".join(rows)


def tests_table(result: Mapping[str, Any]) -> str:
    rows = [
        "| Tác vụ | So sánh (A so với B) | Δ macro-F1 (điểm) | KTC 95% của Δ | p bootstrap "
        "| A đúng, B sai | A sai, B đúng | p McNemar |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for key, name in (
            ("flagged_vs_before", "Cờ so với trước"),
            ("all_review_vs_before", "Trần trên so với trước"),
        ):
            paired = result["tests"][key][task]
            rows.append(
                f"| {TASK_NAME[task]} | {name} | {points(paired['delta_macro_f1'])} "
                f"| {interval_text(paired)} | {p_text(paired['bootstrap_p'])} "
                f"| {paired['a_only_correct']} | {paired['b_only_correct']} "
                f"| {p_text(paired['mcnemar_p'])} |"
            )
    return "\n".join(rows)


def random_table(result: Mapping[str, Any]) -> str:
    rows = [
        "| Tác vụ | mF1 nhánh cờ | mF1 ngẫu nhiên (TB ± ĐLC) | Nhỏ nhất – lớn nhất "
        "| Hiệu cờ − ngẫu nhiên (điểm) | Số lần cờ ≥ ngẫu nhiên | p thực nghiệm (một phía) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = result["tests"]["flagged_vs_random"][task]
        rows.append(
            f"| {TASK_NAME[task]} | {pct(item['macro_f1_flagged'])} "
            f"| {pct(item['macro_f1_random_mean'])} ± {pct(item['macro_f1_random_std'])} "
            f"| {pct(item['macro_f1_random_min'])} – {pct(item['macro_f1_random_max'])} "
            f"| {points(item['delta_macro_f1'])} "
            f"| {item['runs_flagged_at_least']}/{item['n_runs']} | {p_text(item['p_empirical'])} |"
        )
    return "\n".join(rows)


def flagging_table(result: Mapping[str, Any]) -> str:
    flagging = result["flagging"]
    sentiment, topic = flagging["errors"]["sentiment"], flagging["errors"]["topic"]
    rows = [
        "| Chỉ số trên R (mô hình «trước») | Cảm xúc | Chủ đề | Cả hai tác vụ |",
        "|---|---:|---:|---:|",
    ]

    def accuracy_row(name: str, key: str, both: float | None) -> str:
        cells = [percent_or_dash(flagging[key][task]) for task in TASKS]
        return f"| {name} | {cells[0]} | {cells[1]} | {percent_or_dash(both)} |"

    wrong_flagged = percent_or_dash(flagging["wrong_any_share_flagged"])
    wrong_rest = percent_or_dash(flagging["wrong_any_share_unflagged"])
    rows += [
        accuracy_row(
            "Accuracy phần được cờ", "accuracy_flagged", flagging["accuracy_both_flagged"]
        ),
        accuracy_row(
            "Accuracy phần còn lại", "accuracy_unflagged", flagging["accuracy_both_unflagged"]
        ),
        f"| Số câu sai trong R | {sentiment['total']} | {topic['total']} | - |",
        f"| Số câu sai nằm trong phần cờ | {sentiment['in_flagged']} | {topic['in_flagged']} | - |",
        "| Lỗi bị bắt (tỉ lệ lỗi của R nằm trong phần cờ) "
        f"| {percent_or_dash(sentiment['caught_share'])} "
        f"| {percent_or_dash(topic['caught_share'])} | - |",
        "| Tỉ lệ câu sai ít nhất một tác vụ: phần cờ / phần còn lại | - | - "
        f"| {wrong_flagged} / {wrong_rest} |",
    ]
    return "\n".join(rows)


def _exponent(value: float) -> str:
    """2e-05 -> «2e-5»."""
    return f"{value:.0e}".replace("e-0", "e-")


def _deployed_text(check: Mapping[str, Any]) -> str:
    if not check["checked"]:
        return "chưa kiểm được với bản đã lưu (không thấy `models/baseline/*_linearsvc.skops`)"
    where = "bản đã lưu `models/baseline/*_linearsvc.skops`"
    if check["identical"]:
        return f"nhãn dự đoán trên toàn validation ({vn(check['n'], 0)} câu) trùng 100% với {where}"
    rates = ", ".join(f"{TASK_NAME[t].lower()} {pct(check['match_rate'][t], 2)}%" for t in TASKS)
    return f"nhãn dự đoán trùng {where} {rates} (KHÔNG trùng hoàn toàn; vẫn chạy tiếp)"


def _design_section(result: Mapping[str, Any]) -> list[str]:
    params, data, flagging = result["parameters"], result["data"], result["flagging"]
    n = {key: vn(data[key], 0) for key in ("n_train", "n_val", "n_review", "n_eval")}
    n_flagged, budget = vn(flagging["n_flagged"], 0), vn(params["budget_pct"], 0)
    repeats = params["random_repeats"]
    hp = " · ".join(f"{TASK_NAME[t].lower()} {hp_text(params['hp'][t])}" for t in TASKS)
    return [
        "## Thiết kế",
        "",
        f"- Dữ liệu: train {n['n_train']} câu, validation {n['n_val']} câu "
        f"(chuẩn hoá bản {data['normalizer_version']}). Không dùng tập test.",
        f"- Chia validation cố định (seed {params['seed']}, phân tầng theo nhãn cảm xúc): nửa "
        f"duyệt **R** = {n['n_review']} câu, nửa đánh giá **E** = {n['n_eval']} câu, rời nhau.",
        "- Mô hình «trước»: TF-IDF + LinearSVC với siêu tham số đã đóng băng của bản triển khai "
        f"({hp}), fit chỉ trên train bằng `ml.retrain.retrain_baseline` với phản hồi rỗng; "
        f"{_deployed_text(result['deployed_check'])}.",
        f"- Gắn cờ trên R theo ngân sách {budget}%: ⌈{budget}% · {n['n_review']}⌉ = {n_flagged} "
        "câu điểm thấp nhất. Độ tin cậy là xác suất lớn nhất của softmax(decision_function); "
        "điểm cờ là phân vị thấp hơn của hai tác vụ, phân vị tính ngay trong R "
        "(`edupulse.scoring`).",
        f"- Người duyệt mô phỏng: {n_flagged} câu cờ nhận nhãn gốc của cả hai tác vụ.",
        "- Ba nhánh «sau», mỗi nhánh retrain bằng đúng `retrain_baseline` và cùng siêu tham số: "
        "**cờ** (train + câu cờ); **ngẫu nhiên** (đối chứng: train + cùng số câu chọn ngẫu "
        f"nhiên trong R, lặp {repeats} lần với seed 0..{repeats - 1}); **trần trên** (train + "
        "toàn bộ R).",
        f"- Đánh giá mọi nhánh trên E (n = {n['n_eval']}); kiểm định ghép cặp: McNemar chính xác "
        f"(độ chính xác) và bootstrap {vn(params['bootstrap_rounds'], 0)} vòng (hiệu macro-F1).",
        "",
    ]


def _gate_text(gate: Mapping[str, Any]) -> str:
    return f"{'NHẬN' if gate['accepted'] else 'TỪ CHỐI'} ({gate['reason']})"


def _ceiling_note(paired: Mapping[str, Any]) -> str:
    if paired["delta_macro_f1"] > 0:
        return "Trần trên dùng toàn bộ nhãn của R nên là mốc tham chiếu cho nhánh cờ."
    return (
        "Trần trên không cải thiện ở tác vụ này, nên mô phỏng không cho cơ sở kỳ vọng nhánh "
        "cờ cải thiện."
    )


def _reading_section(result: Mapping[str, Any]) -> list[str]:
    data, flagging, tests = result["data"], result["flagging"], result["tests"]
    errors = flagging["errors"]
    train_share = flagging["n_flagged"] / data["n_train"] * 100
    lines = [
        "## Cách đọc",
        "",
        f"Đây là mô phỏng nhỏ: R chỉ có {vn(data['n_review'], 0)} câu, ngân sách "
        f"{vn(result['parameters']['budget_pct'], 0)}% là {vn(flagging['n_flagged'], 0)} câu "
        f"(bằng {vn(train_share, 1)}% số câu train), nên hiệu ứng kỳ vọng rất nhỏ và có thể không "
        "có ý nghĩa thống kê. Kết quả được nêu đúng như đo được, kể cả khi không có cải thiện.",
        "",
    ]
    for task in TASKS:
        paired = tests["flagged_vs_before"][task]
        lines.append(f"- **{TASK_NAME[task]}**, nhánh cờ so với «trước»: {verdict(paired)}.")
    for task in TASKS:
        item = tests["flagged_vs_random"][task]
        delta = item["delta_macro_f1"]
        if round(abs(delta) * 100, 1) == 0:
            relation = "xấp xỉ trung bình ngẫu nhiên (chênh dưới 0,1 điểm)"
        else:
            side = "cao hơn" if delta > 0 else "thấp hơn"
            relation = f"{side} trung bình ngẫu nhiên {vn(abs(delta) * 100, 1)} điểm"
        lines.append(
            f"- **{TASK_NAME[task]}**, cờ so với ngẫu nhiên cùng cỡ: macro-F1 nhánh cờ {relation}; "
            f"nhánh cờ không thấp hơn ngẫu nhiên ở {item['runs_flagged_at_least']}/{item['n_runs']}"
            f" lần, p thực nghiệm một phía {p_text(item['p_empirical'])}. Độ lệch chuẩn giữa các "
            f"lần ngẫu nhiên là {pct(item['macro_f1_random_std'])} điểm, cho biết nhiễu của chính "
            "phép đo."
        )
    for task in TASKS:
        paired = tests["all_review_vs_before"][task]
        lines.append(
            f"- **{TASK_NAME[task]}**, trần trên (duyệt hết R): {verdict(paired)}. "
            f"{_ceiling_note(paired)}"
        )
    lines += [
        "- Quy tắc nhận mô hình mới của `ml.retrain`, áp lên số đo trên E (nửa validation chưa "
        "dùng để huấn luyện) thay vì toàn validation, với nhánh cờ: "
        f"{_gate_text(result['accept_rule']['flagged'])}.",
        f"- Gắn cờ {pct(flagging['flagged_share'], 0)}% số câu của R bắt được "
        f"{percent_or_dash(errors['sentiment']['caught_share'])}% lỗi cảm xúc và "
        f"{percent_or_dash(errors['topic']['caught_share'])}% lỗi chủ đề của mô hình «trước» "
        f"(chọn ngẫu nhiên kỳ vọng bắt khoảng {pct(flagging['flagged_share'], 0)}%).",
        "",
    ]
    return lines


def _limits_section(result: Mapping[str, Any]) -> list[str]:
    data, params = result["data"], result["parameters"]
    neutral = data["eval_support"]["sentiment"][LABELS["sentiment"].index("neutral")]
    others = data["eval_support"]["topic"][LABELS["topic"].index("others")]
    steps = result["phobert_smoke"].get("steps")
    smoke = (
        f"chỉ smoke {steps} bước, không phải retrain đầy đủ" if steps else "không chạy (đã bỏ qua)"
    )
    return [
        "## Giới hạn",
        "",
        "- Nhãn gốc thay người duyệt: người thật có thể sửa sai hoặc không nhất quán, còn nhãn gốc "
        "của UIT-VSFC cũng có nhiễu (nhất là chủ đề «khác»); mô phỏng này vì vậy là ước lượng lạc "
        "quan về chất lượng nhãn.",
        "- Chỉ LinearSVC (baseline TF-IDF) được retrain đầy đủ; mô hình triển khai là PhoBERT và "
        f"phần PhoBERT {smoke}.",
        f"- Một lần chia R/E (seed {params['seed']}) và một vòng duyệt; chưa lặp qua nhiều cách "
        f"chia. E chỉ có {vn(data['n_eval'], 0)} câu, trong đó lớp trung tính {neutral} câu và "
        f"lớp «khác» {others} câu, nên F1 của hai lớp này nhiễu mạnh.",
        "- Siêu tham số baseline đã được chọn trên toàn bộ validation (gồm cả E), nên số của "
        "«trước» hơi lạc quan; mọi nhánh dùng chung E nên so sánh trước/sau vẫn cùng điều kiện.",
        "- Câu cờ được chọn theo độ tin cậy của LinearSVC (softmax của điểm quyết định, chưa hiệu "
        "chuẩn), không phải của PhoBERT đã hiệu chuẩn đang triển khai, nên tập câu cờ khác tập mà "
        "API sẽ cờ; vì chỉ xếp hạng theo phân vị trong R nên chỉ thứ hạng mới có nghĩa.",
        "- Mô hình mới chỉ được chấm trên validation; tập test không được dùng ở đây.",
        "",
    ]


def _smoke_section(result: Mapping[str, Any]) -> list[str]:
    smoke = result["phobert_smoke"]
    if smoke.get("skipped"):
        return ["## Smoke retrain PhoBERT", "", f"Không chạy: {smoke['reason']}.", ""]
    lines = [
        f"## Smoke retrain PhoBERT (CPU, {smoke['steps']} bước)",
        "",
        f"Chạy `retrain_phobert` từ run đã triển khai `{smoke['base_run']}` với "
        f"{vn(smoke['feedback_size'], 0)} câu cờ (nhãn gốc) làm phản hồi. Mỗi batch "
        f"{smoke['batch_size']} câu gồm {smoke['feedback_per_batch']} phản hồi (lấy có hoàn lại) "
        f"và {smoke['replay_per_batch']} câu phát lại từ train; AdamW lr "
        f"{_exponent(smoke['learning_rate'])}, weight decay {vn(smoke['weight_decay'], 2)}, clip "
        f"{vn(smoke['clip_norm'], 1)}. Dự đoán là FP32 bằng torch trên CPU (cả trước lẫn sau), "
        "không phải bản int8 ONNX đang triển khai; đo trên E "
        f"(n = {vn(smoke['n_eval'], 0)}).",
        "",
        "| Tác vụ | Acc trước | Acc sau | mF1 trước | mF1 sau | Δ mF1 (điểm) "
        "| Số nhãn đổi trên E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in smoke["before"]:
        before, after = smoke["before"][task], smoke["after"][task]
        lines.append(
            f"| {TASK_NAME[task]} | {pct(before['accuracy'])} | {pct(after['accuracy'])} "
            f"| {pct(before['macro_f1'])} | {pct(after['macro_f1'])} "
            f"| {points(smoke['delta_macro_f1'][task])} | {smoke['label_changes'][task]} |"
        )
    losses = smoke["losses"]
    lines += [
        "",
        f"Thời gian: huấn luyện {vn(smoke['train_seconds'], 1)} giây; cả bước (gồm lưu "
        f"checkpoint và dự đoán trước/sau) {vn(smoke['total_seconds'], 1)} giây. Loss tổng hai "
        f"tác vụ: bước đầu {vn(losses[0], 3)}, bước cuối {vn(losses[-1], 3)}.",
        "",
        "Quy tắc nhận mô hình mới của `ml.retrain` áp lên số đo trên E (không phải toàn "
        f"validation): {_gate_text(smoke['accept_rule'])}.",
        "",
        "Đây chỉ là smoke chứng minh đường chạy `python -m ml.retrain --model phobert` chạy trọn "
        f"trên CPU; {smoke['steps']} bước không kỳ vọng thay đổi đáng kể, và phép đo này không "
        "dùng để kết luận mô hình tốt lên hay xấu đi.",
        "",
    ]
    return lines


def render_report(result: Mapping[str, Any]) -> str:
    """Báo cáo Markdown tiếng Việt từ kết quả mô phỏng (không chứa câu gốc)."""
    data, params = result["data"], result["parameters"]
    lines = [
        "# Mô phỏng vòng người duyệt (HITL) trên validation",
        "",
        "> **Đây là mô phỏng.** Nhãn gốc của UIT-VSFC đóng vai nhãn do người duyệt sửa (coi như "
        "đúng); không có người duyệt thật. Không dùng tập test.",
        "",
        *_design_section(result),
        f"## Kết quả trên nửa đánh giá E (n = {vn(data['n_eval'], 0)})",
        "",
        "Phần trăm. Acc = accuracy, mF1 = macro-F1, wF1 = weighted-F1; nhánh ngẫu nhiên là trung "
        "bình ± độ lệch chuẩn mẫu của các lần lặp.",
        "",
        results_table(result),
        "",
        "## Kiểm định trên E",
        "",
        "Δ macro-F1 = macro-F1(A) − macro-F1(B), tính bằng điểm phần trăm; khoảng tin cậy 95% "
        f"theo phân vị của {vn(params['bootstrap_rounds'], 0)} lần bootstrap ghép cặp (cùng chỉ "
        "số câu cho hai mô hình); McNemar chính xác hai phía trên đúng/sai từng câu (b = A đúng, "
        "B sai; c = A sai, B đúng).",
        "",
        tests_table(result),
        "",
        "Nhánh cờ so với đối chứng ngẫu nhiên cùng cỡ (p thực nghiệm một phía = (1 + số lần "
        "ngẫu nhiên ≥ nhánh cờ) / (số lần + 1)):",
        "",
        random_table(result),
        "",
        f"## Hiệu quả gắn cờ trên R (n = {vn(data['n_review'], 0)})",
        "",
        flagging_table(result),
        "",
        *_reading_section(result),
        *_limits_section(result),
        *_smoke_section(result),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(result: Mapping[str, Any], metrics_path: Path, table_path: Path) -> None:
    write_json(metrics_path, result)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_report(result), encoding="utf-8")


# --- Dòng lệnh --------------------------------------------------------------------------------


def deployed_predictions(val: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Nhãn dự đoán (pipeline thô) của baseline LinearSVC đã lưu trên `val`; None nếu chưa có."""
    if not all(baseline_path(task, "linearsvc").is_file() for task in TASKS):
        return None
    preds = baseline_preds("linearsvc", val)
    return {task: np.asarray(preds.pred[task], dtype=np.int64) for task in TASKS}


def code_version() -> dict[str, Any]:
    """Phiên bản mã: commit, cây làm việc có thay đổi chưa, sha256 các tệp tạo ra kết quả."""
    try:
        commit, dirty = git_head(), len(git_dirty())
    except (OSError, subprocess.SubprocessError):  # không có git hoặc không phải kho git
        commit, dirty = None, None
    hashes = {name: sha256_file(REPO_ROOT / name) for name in CODE_FILES}
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_commit": commit, "git_dirty_files": dirty, "sha256": hashes, "versions": versions}


def data_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Dấu vân tay của train và validation (không ghi gì về test)."""
    files = manifest["files"]
    return {
        "dataset_revision": manifest["dataset_revision"],
        "train_sha256": files["train"]["sha256"],
        "validation_sha256": files["validation"]["sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-phobert", action="store_true", help="bỏ bước smoke PhoBERT")
    parser.add_argument("--seed", type=int, default=SEED, help="seed chia validation thành R và E")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results", help="thư mục gốc của kết quả"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Smoke PhoBERT chiếm hết lõi CPU: giữ khoá toàn máy như mọi lần huấn luyện thật, không chờ.
    lock = contextlib.nullcontext() if args.skip_phobert else mps_lock(note="hitl_simulation")
    try:
        with lock:
            result = simulate(args)
    except MpsBusyError as error:
        print(error, file=sys.stderr)
        return 4
    metrics_path = args.results_dir / METRICS_PATH
    table_path = args.results_dir / TABLE_PATH
    write_outputs(result, metrics_path, table_path)
    log_line(f"đã ghi {metrics_path} và {table_path}")
    print(table_path.read_text(encoding="utf-8"))
    return 0


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    train, val = load_processed("train"), load_processed("validation")
    hp = load_deployed_baseline_hp()
    result = run_simulation(
        train,
        val,
        hp,
        seed=args.seed,
        deployed_pred=deployed_predictions(val),
        phobert_run=None if args.skip_phobert else resolve_run("best"),
        log=log_line,
    )
    result["data"] |= data_manifest(processed_manifest())
    result["code"] = code_version()
    return result


if __name__ == "__main__":
    sys.exit(main())
