"""Baseline TF-IDF: lưu skops với danh sách kiểu tin cậy tường minh và nạp lại cho kết quả y hệt."""

import numpy as np
import pytest
import skops.io as sio
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from edupulse.bundle_schema import BASELINE_TRUSTED_TYPES
from edupulse.config import load_settings
from edupulse.labels import LABELS
from ml.common import load_processed, prediction_metrics
from ml.train_baseline import (
    GRIDS,
    drop_rare_terms,
    load_model,
    make_classifier,
    make_features,
    save_model,
)

TEXTS = [
    "thầy dạy rất nhiệt_tình",
    "phòng máy nóng quá",
    "chương_trình học nặng lý_thuyết",
    "thầy giảng dễ hiểu",
    "máy_chiếu mờ , micro rè",
    "môn này có nhiều bài_tập",
] * 10
LABEL_IDS = np.array([2, 0, 0, 2, 0, 1] * 10)


def toy_model(name: str):
    features = make_features().fit(TEXTS)
    drop_rare_terms(features)
    classifier = make_classifier(name, GRIDS[name][0]).fit(features.transform(TEXTS), LABEL_IDS)
    pipeline = Pipeline([("features", features), ("clf", classifier)])
    return CalibratedClassifierCV(FrozenEstimator(pipeline), method="temperature").fit(
        TEXTS, LABEL_IDS
    )


@pytest.mark.parametrize("name", list(GRIDS))
def test_roundtrip_gives_identical_probabilities(name, tmp_path):
    model = toy_model(name)
    path = tmp_path / f"{name}.skops"
    save_model(model, path)
    loaded = load_model(path)
    assert np.allclose(loaded.predict_proba(TEXTS), model.predict_proba(TEXTS))
    assert (loaded.predict(TEXTS) == model.predict(TEXTS)).all()


@pytest.mark.parametrize("name", list(GRIDS))
def test_only_the_two_calibration_types_need_trust(name, tmp_path):
    path = tmp_path / f"{name}.skops"
    save_model(toy_model(name), path)
    assert set(sio.get_untrusted_types(file=path)) == set(BASELINE_TRUSTED_TYPES)


def test_loading_without_trusting_anything_is_refused(tmp_path):
    path = tmp_path / "m.skops"
    save_model(toy_model("linearsvc"), path)
    with pytest.raises(Exception, match="Untrusted"):
        sio.load(path, trusted=[])


def test_rare_terms_are_not_kept_in_the_saved_model(tmp_path):
    features = make_features().fit(TEXTS)
    drop_rare_terms(features)
    assert not any(hasattr(v, "stop_words_") for _, v in features.transformer_list)


def test_unexpected_type_is_reported_instead_of_trusted(tmp_path, monkeypatch):
    import ml.train_baseline as module

    monkeypatch.setattr(module, "BASELINE_TRUSTED_TYPES", ())
    with pytest.raises(RuntimeError, match="BASELINE_TRUSTED_TYPES"):
        module.save_model(toy_model("linearsvc"), tmp_path / "m.skops")


@pytest.mark.data
@pytest.mark.parametrize("task", ["sentiment", "topic"])
def test_saved_linearsvc_is_a_working_baseline_on_validation(task):
    path = load_settings().models_dir / "baseline" / f"{task}_linearsvc.skops"
    if not path.is_file():
        pytest.skip(f"chưa có {path.name} — chạy `make baseline`")
    model = load_model(path)
    val = load_processed("validation")
    metrics = prediction_metrics(val[task], model.predict(val["text_rdr"].tolist()), LABELS[task])
    # Khoảng của nghiên cứu: cảm xúc ~79,6 và chủ đề ~77,2 mF1 (sai số ±1,5 điểm).
    expected = {"sentiment": 0.796, "topic": 0.772}[task]
    assert abs(metrics["macro_f1"] - expected) < 0.025
    probs = model.predict_proba(val["text_rdr"].tolist())
    assert probs.shape == (len(val), len(LABELS[task]))
    assert np.allclose(probs.sum(axis=1), 1.0)
