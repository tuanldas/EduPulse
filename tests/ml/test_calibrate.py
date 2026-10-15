"""Hiệu chuẩn: nhiệt độ, bảng phân vị và ngưỡng ngân sách khớp mã `scoring` mà API dùng."""

import math

import numpy as np
import pytest

from edupulse.bundle_schema import Calibration
from edupulse.config import BUDGETS
from edupulse.labels import LABELS, TASKS
from edupulse.scoring import (
    confidence,
    flag_score,
    flags_by_threshold,
    quantile_rank,
    rank_batch,
    softmax_t,
)
from ml import calibrate


def sample(n, n_classes, true_temperature, seed):
    """Logits có nhãn lấy mẫu từ softmax(logits / T0): mô hình hiệu chuẩn nếu dùng T = T0."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n_classes)) * 3
    probs = softmax_t(logits, true_temperature)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    return logits, labels


def test_fit_temperature_recovers_the_generating_temperature():
    logits, y = sample(20000, 3, true_temperature=2.0, seed=0)
    assert calibrate.fit_temperature(logits, y) == pytest.approx(2.0, abs=0.15)


def test_overconfident_head_uses_the_fitted_temperature():
    logits, y = sample(1500, 3, true_temperature=2.0, seed=1)
    temperature, decision = calibrate.choose_temperature(logits, y)
    assert decision["used_fitted"] is True
    assert temperature == pytest.approx(decision["fitted"])
    assert temperature > 1.5
    assert decision["nll_after"] < decision["nll_before"]


def test_already_calibrated_head_keeps_temperature_one():
    logits, y = sample(1500, 3, true_temperature=1.0, seed=2)
    temperature, decision = calibrate.choose_temperature(logits, y)
    assert decision["fitted"] == pytest.approx(1.0, abs=0.2)
    if not decision["used_fitted"]:
        assert temperature == 1.0
    else:  # kiểm chéo thấy có lợi thật thì được dùng; khi đó vẫn phải gần 1
        assert temperature == pytest.approx(1.0, abs=0.2)


def synthetic_validation(n=1583):
    s_logits, s_y = sample(n, len(LABELS["sentiment"]), 1.5, seed=3)
    t_logits, t_y = sample(n, len(LABELS["topic"]), 1.0, seed=4)
    return {"sentiment": s_logits, "topic": t_logits}, {"sentiment": s_y, "topic": t_y}


def test_calibration_table_agrees_with_the_runtime_scoring_path():
    val_logits, targets = synthetic_validation()
    calibration, _ = calibrate.calibrate(val_logits, targets)
    assert isinstance(calibration, Calibration)
    assert calibration.n == 1583

    heads = {"sentiment": calibration.sentiment, "topic": calibration.topic}
    ranks = {}
    for task in TASKS:
        probs = softmax_t(val_logits[task].astype(np.float64), heads[task].temperature)
        ranks[task] = quantile_rank(confidence(probs), heads[task].cdf)
    scores = flag_score(ranks["sentiment"], ranks["topic"])
    for budget in BUDGETS:
        expected = math.ceil(round(budget * len(scores) / 100, 9))
        # Luật cho từng câu (ngưỡng) và luật cho lô (xếp hạng) cùng cờ đúng ⌈B·n/100⌉ câu val.
        assert flags_by_threshold(scores, calibration.tau[budget]).sum() == expected
        assert rank_batch(scores, budget).sum() == expected
        assert calibration.efficiency[budget].flagged_share == pytest.approx(expected / len(scores))


def test_larger_budgets_catch_more_errors():
    val_logits, targets = synthetic_validation()
    calibration, _ = calibrate.calibrate(val_logits, targets)
    caught = [calibration.efficiency[b].caught_sentiment_errors for b in BUDGETS]
    assert caught == sorted(caught)
    assert caught[-1] > caught[0]
    assert [calibration.tau[b] for b in BUDGETS] == sorted(calibration.tau[b] for b in BUDGETS)


def test_the_gain_of_flagging_is_real_on_synthetic_data():
    val_logits, targets = synthetic_validation()
    calibration, _ = calibrate.calibrate(val_logits, targets)
    ten = calibration.efficiency[10]
    assert ten.accept_accuracy_both > ten.all_accuracy_both


def test_main_asks_for_export_when_logits_are_missing(monkeypatch, tmp_path, capsys):
    run = tmp_path / "models" / "phobert" / "mtl-s1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    assert calibrate.main(["--run", "mtl-s1"]) == 2
    assert "make export" in capsys.readouterr().err
