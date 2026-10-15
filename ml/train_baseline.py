"""Ba baseline TF-IDF cho cả hai tác vụ: LinearSVC, LogisticRegression, ComplementNB.

Đặc trưng chung: TF-IDF từ 1-2 gram trên văn bản đã tách từ ⊕ TF-IDF ký tự 2-5 trong từ. Siêu
tham số (C, alpha, class_weight) chọn theo macro-F1 trên validation; train chỉ dùng train.
Xác suất lấy từ `CalibratedClassifierCV(FrozenEstimator(...), method="temperature")` fit trên
validation; bảng báo cả bản thô (dùng cho F1) lẫn bản đã hiệu chuẩn (dùng cho xác suất).

Mô hình lưu bằng skops (không dùng pickle/joblib vì nạp được chạy mã tuỳ ý). Tách từ nằm NGOÀI
pipeline: đầu vào là `text_rdr` do `edupulse.text.prepare` sinh ra.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from edupulse.bundle_schema import BASELINE_TRUSTED_TYPES
from edupulse.config import REPO_ROOT
from edupulse.labels import LABELS, TASKS
from edupulse.text import NORMALIZER_VERSION
from ml.common import load_processed, prediction_metrics, settings, task_metrics, write_json

LOG = logging.getLogger("train_baseline")
SEED = 42

# Lưới siêu tham số: C nhỏ làm sập lớp trung tính, class_weight làm đường cong phẳng hơn theo C.
GRIDS: dict[str, list[dict[str, Any]]] = {
    "linearsvc": [{"C": c, "class_weight": w} for c in (0.3, 1.0, 3.0) for w in (None, "balanced")],
    "logreg": [{"C": c, "class_weight": w} for c in (3.0, 10.0, 30.0) for w in (None, "balanced")],
    "complementnb": [{"alpha": a} for a in (0.1, 0.3, 1.0)],
}
DISPLAY = {
    "linearsvc": "TF-IDF + LinearSVC",
    "logreg": "TF-IDF + LogisticRegression",
    "complementnb": "TF-IDF + ComplementNB",
}
FEATURE_SPEC = {
    "word": "TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf, token_pattern=\\b\\w+\\b)",
    "char": "TfidfVectorizer(analyzer=char_wb, ngram_range=(2,5), min_df=3, sublinear_tf)",
}
NOTE = (
    "Acc/mF1/F1 từng lớp là bản thô (không hiệu chuẩn). ECE ở bảng là của bản hiệu chuẩn, đo trên "
    "chính tập val dùng để fit nhiệt độ nên hơi lạc quan; số cuối cùng đo trên test."
)


def make_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True
                ),
            ),
        ]
    )


def make_classifier(name: str, hp: dict[str, Any]) -> Any:
    if name == "linearsvc":
        return LinearSVC(random_state=SEED, **hp)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=SEED, **hp)
    if name == "complementnb":
        return ComplementNB(**hp)
    raise ValueError(f"không có baseline {name!r}")


def drop_rare_terms(features: FeatureUnion) -> None:
    """Xoá `stop_words_` (mọi n-gram bị loại vì min_df): gọn hơn và không giữ cụm hiếm."""
    for _, vectorizer in features.transformer_list:
        if hasattr(vectorizer, "stop_words_"):
            del vectorizer.stop_words_


def describe(metrics: dict[str, Any], *, probabilistic: bool) -> dict[str, Any]:
    """Phần chỉ số đưa vào JSON; ECE/NLL chỉ có khi mô hình cho xác suất."""
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_f1": [c["f1"] for c in metrics["per_class"]],
        "ece": metrics["ece"] if probabilistic else None,
        "nll": metrics["nll"] if probabilistic else None,
    }


def fit_best(
    name: str, task: str, x_train: Any, y_train: np.ndarray, x_val: Any, y_val: np.ndarray
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Thử cả lưới trên val, giữ cấu hình có macro-F1 cao nhất."""
    best_score, best_hp, best_clf = -1.0, {}, None
    trials = []
    for hp in GRIDS[name]:
        started = time.time()
        clf = make_classifier(name, hp).fit(x_train, y_train)
        metrics = prediction_metrics(y_val, clf.predict(x_val), LABELS[task])
        trials.append({"hp": hp, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        LOG.info(
            "%s/%s %s mF1=%.4f (%.1fs)", task, name, hp, metrics["macro_f1"], time.time() - started
        )
        if metrics["macro_f1"] > best_score:
            best_score, best_hp, best_clf = metrics["macro_f1"], hp, clf
    return best_hp, best_clf, trials


def run_task(
    task: str,
    features: FeatureUnion,
    train_texts: list[str],
    val_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kết quả từng baseline của một tác vụ và các mô hình đã hiệu chuẩn tương ứng."""
    x_train, x_val = features.transform(train_texts), features.transform(val_texts)
    labels = LABELS[task]
    results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for name in GRIDS:
        hp, clf, trials = fit_best(name, task, x_train, y_train, x_val, y_val)
        pipeline = Pipeline([("features", features), ("clf", clf)])
        if hasattr(clf, "predict_proba"):
            raw = describe(
                task_metrics(np.log(np.clip(clf.predict_proba(x_val), 1e-12, 1.0)), y_val, labels),
                probabilistic=True,
            )
        else:
            raw = describe(
                prediction_metrics(y_val, clf.predict(x_val), labels), probabilistic=False
            )
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="temperature")
        calibrated.fit(val_texts, y_val)
        log_probs = np.log(np.clip(calibrated.predict_proba(val_texts), 1e-12, 1.0))
        results[name] = {
            "display": DISPLAY[name],
            "hp": hp,
            "grid": trials,
            "raw": raw,
            "calibrated": describe(task_metrics(log_probs, y_val, labels), probabilistic=True),
        }
        models[name] = calibrated
    return results, models


def save_model(model: Any, path: Path) -> None:
    """Ghi skops và kiểm mọi kiểu cần tin đều nằm trong danh sách đã khai báo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # skops ghi schema JSON có thụt lề và mặc định không nén (~58MB); nén deflate còn vài MB.
    sio.dump(model, path, compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    unexpected = set(sio.get_untrusted_types(file=path)) - set(BASELINE_TRUSTED_TYPES)
    if unexpected:
        raise RuntimeError(
            f"{path.name} cần tin thêm kiểu chưa khai báo trong BASELINE_TRUSTED_TYPES: "
            f"{sorted(unexpected)}"
        )


def load_model(path: Path) -> Any:
    """Nạp baseline với danh sách kiểu tin cậy tường minh (không bao giờ `trusted=True`)."""
    return sio.load(path, trusted=list(BASELINE_TRUSTED_TYPES))


def table_markdown(results: dict[str, Any]) -> str:
    neutral = LABELS["sentiment"].index("neutral")
    others = LABELS["topic"].index("others")

    def ece(task: str, name: str) -> str:
        return f"{results['tasks'][task][name]['calibrated']['ece']:.3f}"

    lines = [
        "| Mô hình | Cảm xúc · Acc | Cảm xúc · mF1 | F1 trung tính | Chủ đề · Acc | Chủ đề · mF1 "
        "| F1 «khác» | ECE cảm xúc | ECE chủ đề |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in GRIDS:
        s = results["tasks"]["sentiment"][name]["raw"]
        t = results["tasks"]["topic"][name]["raw"]
        lines.append(
            f"| {DISPLAY[name]} | {s['accuracy'] * 100:.1f} | {s['macro_f1'] * 100:.1f} "
            f"| {s['per_class_f1'][neutral] * 100:.1f} | {t['accuracy'] * 100:.1f} "
            f"| {t['macro_f1'] * 100:.1f} | {t['per_class_f1'][others] * 100:.1f} "
            f"| {ece('sentiment', name)} | {ece('topic', name)} |"
        )
    return "\n".join(lines) + f"\n\n{NOTE}\n"


def run() -> dict[str, Any]:
    train = load_processed("train")
    val = load_processed("validation")
    train_texts, val_texts = train["text_rdr"].tolist(), val["text_rdr"].tolist()
    features = make_features().fit(train_texts)
    drop_rare_terms(features)
    LOG.info("đặc trưng: %d chiều", len(features.get_feature_names_out()))

    output: dict[str, Any] = {
        "split": "validation",
        "normalizer_version": NORMALIZER_VERSION,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "features": FEATURE_SPEC,
        "note": NOTE,
        "tasks": {},
    }
    for task in TASKS:
        results, models = run_task(
            task, features, train_texts, val_texts, train[task].to_numpy(), val[task].to_numpy()
        )
        output["tasks"][task] = results
        for name, model in models.items():
            save_model(model, settings().models_dir / "baseline" / f"{task}_{name}.skops")
    return output


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run()
    write_json(REPO_ROOT / "results" / "metrics" / "val_baselines.json", results)
    table = table_markdown(results)
    table_path = REPO_ROOT / "results" / "tables" / "val_baselines.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
