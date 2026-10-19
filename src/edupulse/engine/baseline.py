"""Mô hình cơ sở TF-IDF + LinearSVC để so sánh với PhoBERT (nút "So sánh với mô hình cơ sở").

Hai mô hình (cảm xúc, chủ đề) nằm trong `baseline/*.skops` của bundle và chỉ được nạp lần đầu có
người yêu cầu so sánh, nên không tốn RAM khi không dùng. Đầu vào là `Prepared.model_text` giống hệt
đầu vào của PhoBERT. Xác suất lấy từ lớp hiệu chuẩn nhiệt độ đã lưu cùng mô hình.

Giải thích bằng hệ số: với lớp được dự đoán c, đóng góp của đặc trưng j là `coef[c, j] · x[j]`
(x là vector TF-IDF đã chuẩn hoá L2); các từ/cụm có đóng góp dương lớn nhất là "từ quyết định".
Chỉ xét đặc trưng mức từ (đơn từ và cụm hai từ); đặc trưng ký tự không đọc được nên bị bỏ qua.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio

from edupulse.bundle_schema import BASELINE_TRUSTED_TYPES
from edupulse.labels import LABELS, TASKS

logger = logging.getLogger("edupulse.engine")

NAME = "TF-IDF + LinearSVC"
FILENAME = "{task}_linearsvc.skops"
TOP_TERMS = 3
WORD_FEATURES = "word"
FEATURE_PREFIX = f"{WORD_FEATURES}__"


class BaselineUnavailable(RuntimeError):
    """Bundle không có mô hình cơ sở dùng được (thiếu tệp, hỏng hoặc kiểu không được tin)."""

    code = "baseline_unavailable"


@dataclass(frozen=True, slots=True)
class BaselineHead:
    label: str
    confidence: float
    top_terms: tuple[str, ...]  # đặc trưng gốc, vd. "nhiệt_tình" hoặc "dạy rất"


@dataclass(frozen=True, slots=True)
class BaselineResult:
    sentiment: BaselineHead
    topic: BaselineHead
    elapsed_ms: float  # thời gian dự đoán, không gồm lần nạp mô hình


@dataclass(frozen=True, slots=True)
class _TaskModel:
    calibrated: Any  # CalibratedClassifierCV
    features: Any  # FeatureUnion
    classifier: Any  # LinearSVC
    word_terms: np.ndarray  # tên đặc trưng mức từ (đã bỏ tiền tố), theo cột của ma trận
    n_word: int


def _load_task(path: Path) -> _TaskModel:
    try:
        calibrated = sio.load(path, trusted=list(BASELINE_TRUSTED_TYPES))
        pipeline = calibrated.calibrated_classifiers_[0].estimator.estimator
        features, classifier = pipeline.named_steps["features"], pipeline.named_steps["clf"]
        names = features.get_feature_names_out()
    except Exception as exc:  # skops/sklearn báo lỗi bằng nhiều kiểu: thiếu tệp, zip hỏng, kiểu lạ
        raise BaselineUnavailable(
            f"Không nạp được mô hình cơ sở {path.name} ({type(exc).__name__})."
        ) from exc
    word = [i for i, name in enumerate(names) if name.startswith(FEATURE_PREFIX)]
    # FeatureUnion nối các khối theo thứ tự, nên đặc trưng mức từ phải là một đoạn liền ở đầu.
    if word != list(range(len(word))):
        raise BaselineUnavailable(f"{path.name}: thứ tự đặc trưng không như mong đợi.")
    terms = np.array([names[i][len(FEATURE_PREFIX) :] for i in word])
    return _TaskModel(calibrated, features, classifier, terms, len(word))


class Baseline:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._models: dict[str, _TaskModel] | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> dict[str, _TaskModel]:
        models = self._models
        if models is not None:
            return models
        with self._lock:
            if self._models is None:
                started = time.perf_counter()
                self._models = {
                    task: _load_task(self._directory / FILENAME.format(task=task)) for task in TASKS
                }
                logger.info("Đã nạp mô hình cơ sở trong %.1f s", time.perf_counter() - started)
            return self._models

    def _head(self, task: str, model: _TaskModel, text: str) -> BaselineHead:
        probs = model.calibrated.predict_proba([text])[0]
        index = int(np.argmax(probs))
        # Hàng TF-IDF của câu, chỉ phần đặc trưng mức từ (các cột đầu của FeatureUnion).
        row = model.features.transform([text]).tocsr()[:, : model.n_word]
        contributions = row.data * model.classifier.coef_[index, row.indices]
        order = np.argsort(-contributions, kind="stable")
        chosen = [int(row.indices[i]) for i in order if contributions[i] > 0][:TOP_TERMS]
        terms = tuple(str(model.word_terms[col]) for col in chosen)
        return BaselineHead(LABELS[task][index], float(probs[index]), terms)

    def predict(self, model_text: str) -> BaselineResult:
        """Dự đoán hai tác vụ cho một câu đã tiền xử lý (`Prepared.model_text`)."""
        models = self._ensure_loaded()
        started = time.perf_counter()
        sentiment = self._head("sentiment", models["sentiment"], model_text)
        topic = self._head("topic", models["topic"], model_text)
        return BaselineResult(sentiment, topic, (time.perf_counter() - started) * 1000)

    @property
    def loaded(self) -> bool:
        return self._models is not None
