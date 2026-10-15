"""Chỉ số và tiện ích dùng chung của `ml/`, kiểm trên mảng có đáp án biết trước."""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edupulse.config import REPO_ROOT
from edupulse.labels import SENTIMENT
from ml.common import (
    MpsBusyError,
    confusion_normalized,
    expected_calibration_error,
    load_processed,
    macro_f1,
    mps_lock,
    mps_lock_is_held,
    negative_log_likelihood,
    pick_device,
    reliability_bins,
    set_seed,
    task_metrics,
    write_json,
)


def one_hot_logits(pred, n_classes, scale=5.0):
    logits = np.zeros((len(pred), n_classes))
    logits[np.arange(len(pred)), pred] = scale
    return logits


# --- Chỉ số ---------------------------------------------------------------------------------------


def test_task_metrics_match_hand_computed_values():
    y = [0, 0, 0, 1, 1, 2]
    pred = [0, 0, 1, 1, 2, 2]
    result = task_metrics(one_hot_logits(pred, 3), y, SENTIMENT)
    assert result["accuracy"] == pytest.approx(4 / 6)
    # F1 từng lớp: 0,8 · 0,5 · 2/3
    assert [round(c["f1"], 4) for c in result["per_class"]] == [0.8, 0.5, 0.6667]
    assert result["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert result["weighted_f1"] == pytest.approx((3 * 0.8 + 2 * 0.5 + 2 / 3) / 6)
    assert [c["support"] for c in result["per_class"]] == [3, 2, 1]
    assert [c["label"] for c in result["per_class"]] == list(SENTIMENT)
    assert result["per_class"][0]["precision"] == 1.0
    assert result["per_class"][0]["recall"] == pytest.approx(2 / 3)


def test_macro_f1_helper_agrees_with_task_metrics():
    y = [0, 0, 0, 1, 1, 2]
    logits = one_hot_logits([0, 0, 1, 1, 2, 2], 3)
    assert macro_f1(logits, y, 3) == pytest.approx(task_metrics(logits, y, SENTIMENT)["macro_f1"])


def test_macro_f1_counts_a_class_that_is_never_predicted_as_zero():
    # Lớp 2 không bao giờ được đoán: F1 = 0 kéo macro xuống, không bị bỏ qua.
    logits = one_hot_logits([0, 0, 1, 1], 3)
    assert macro_f1(logits, [0, 0, 1, 2], 3) == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)


def test_confusion_matrix_rows_sum_to_one_and_empty_rows_stay_zero():
    matrix = np.array(confusion_normalized([0, 0, 1, 1], [0, 1, 1, 1], 3))
    assert matrix[0].tolist() == [0.5, 0.5, 0.0]
    assert matrix[1].tolist() == [0.0, 1.0, 0.0]
    assert matrix[2].tolist() == [0.0, 0.0, 0.0]


def test_ece_of_an_overconfident_model():
    probs = np.array([[0.9, 0.1]] * 4)
    assert expected_calibration_error(probs, [0, 0, 1, 1]) == pytest.approx(0.4)


def test_ece_of_a_calibrated_model_is_zero():
    probs = np.array([[0.8, 0.2]] * 10)
    y = [0] * 8 + [1] * 2
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_ece_uses_equal_width_bins_across_two_levels():
    probs = np.array([[0.6, 0.4]] * 5 + [[0.95, 0.05]] * 5)
    y = [0, 0, 0, 1, 1] + [0] * 5
    # bin 0,6: acc 0,6 vs conf 0,6 -> 0; bin 0,95: acc 1,0 vs conf 0,95 -> 0,05 (trọng số 0,5)
    assert expected_calibration_error(probs, y) == pytest.approx(0.025)


def test_reliability_bins_partition_the_samples():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=500)
    y = rng.integers(0, 3, size=500)
    bins = reliability_bins(probs, y)
    assert len(bins) == 15
    assert sum(b["n"] for b in bins) == 500
    for b in bins:
        assert (b["confidence"] is None) == (b["n"] == 0) == (b["accuracy"] is None)
        if b["n"]:
            assert b["lower"] <= b["confidence"] <= b["upper"] + 1e-9


def test_confidence_of_exactly_one_falls_in_the_last_bin():
    bins = reliability_bins(np.array([[1.0, 0.0]]), [0])
    assert bins[-1]["n"] == 1


def test_nll_is_log_two_for_a_coin_flip():
    assert negative_log_likelihood(np.array([[0.5, 0.5]]), [0]) == pytest.approx(np.log(2))
    assert np.isfinite(negative_log_likelihood(np.array([[0.0, 1.0]]), [0]))  # có chặn log(0)


def test_temperature_changes_ece_but_not_accuracy():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(300, 3)) * 4
    y = logits.argmax(axis=1)
    y[:60] = (y[:60] + 1) % 3  # 20% sai nhưng rất tự tin
    sharp = task_metrics(logits, y, SENTIMENT, temperature=1.0)
    softened = [task_metrics(logits, y, SENTIMENT, temperature=t) for t in (1.5, 2.0, 2.5)]
    assert all(m["accuracy"] == sharp["accuracy"] for m in softened)
    assert min(m["ece"] for m in softened) < sharp["ece"]  # làm mềm bớt tự tin thì ECE giảm


# --- Tiện ích ---------------------------------------------------------------------------------


def test_set_seed_makes_python_and_numpy_reproducible():
    set_seed(7)
    first = (random.random(), np.random.random())
    set_seed(7)
    assert (random.random(), np.random.random()) == first


def test_write_json_is_utf8_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "out" / "m.json"
    write_json(target, {"nhãn": "Tiêu cực", "n": 3})
    assert json.loads(target.read_text("utf-8")) == {"nhãn": "Tiêu cực", "n": 3}
    assert "Tiêu cực" in target.read_text("utf-8")  # không escape \\uXXXX
    assert [p.name for p in target.parent.iterdir()] == ["m.json"]


def test_write_json_failure_keeps_the_old_file(tmp_path):
    target = tmp_path / "m.json"
    write_json(target, {"v": 1})
    with pytest.raises(TypeError):
        write_json(target, {"v": object()})
    assert json.loads(target.read_text("utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_test_split_is_closed_unless_asked_for():
    with pytest.raises(PermissionError, match="test"):
        load_processed("test")
    with pytest.raises(ValueError, match="split"):
        load_processed("dev")


@pytest.mark.data
def test_processed_splits_load_with_expected_shape():
    train = load_processed("train")
    assert len(train) == 11426
    assert {"sentence", "sentiment", "topic", "text_rdr", "n_words", "no_diacritics"} <= set(
        train.columns
    )
    assert len(load_processed("validation")) == 1583
    assert len(load_processed("test", allow_test=True)) == 3166


@pytest.mark.train
def test_pick_device():
    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in {"mps", "cuda", "cpu"}
    with pytest.raises(RuntimeError, match="không khả dụng"):
        pick_device("tpu")


# --- Khoá MPS ---------------------------------------------------------------------------------

_TRY_LOCK = """
import sys
from pathlib import Path
from ml.common import MpsBusyError, mps_lock, mps_lock_is_held
path = Path(sys.argv[1])
print("HELD" if mps_lock_is_held(path) else "FREE")
try:
    with mps_lock(path):
        print("ACQUIRED")
except MpsBusyError as error:
    print("BUSY", error)
"""


def run_probe(path: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _TRY_LOCK, str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-400:]
    return result.stdout.strip().splitlines()


def test_mps_lock_excludes_other_processes_and_releases_on_exit(tmp_path):
    path = tmp_path / "home" / ".mps.lock"
    assert not mps_lock_is_held(path)
    with mps_lock(path, note="test") as held:
        assert held == path
        assert mps_lock_is_held(path)
        out = run_probe(path)
        assert out[0] == "HELD"
        assert out[1].startswith("BUSY") and "đang do job khác giữ" in out[1]
        assert '"note": "test"' in out[1]  # thông báo lỗi nêu ai đang giữ
    assert not mps_lock_is_held(path)
    assert run_probe(path) == ["FREE", "ACQUIRED"]


def test_mps_lock_is_not_reentrant_within_the_same_machine(tmp_path):
    path = tmp_path / ".mps.lock"
    with mps_lock(path):
        with pytest.raises(MpsBusyError):
            with mps_lock(path):
                pass
    with mps_lock(path):  # nhả khoá sau lỗi
        pass


def test_mps_lock_defaults_to_the_configured_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    with mps_lock():
        assert mps_lock_is_held()
        assert (tmp_path / ".mps.lock").is_file()
