"""Thống kê ghép cặp: F1 từ ma trận nhầm lẫn khớp scikit-learn, bootstrap tái lập, McNemar đúng."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from ml import stats


def random_case(n: int, k: int, seed: int, noise: float = 0.3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    flip = rng.random(n) < noise
    pred = np.where(flip, rng.integers(0, k, size=n), y)
    return y, pred


@pytest.mark.parametrize(("n", "k", "seed"), [(500, 3, 0), (800, 4, 1), (60, 4, 2)])
def test_f1_family_matches_scikit_learn(n, k, seed):
    y, pred = random_case(n, k, seed)
    counts = stats.confusion_counts(y, pred, k)
    labels = list(range(k))
    assert stats.macro_f1_from_counts(counts) == pytest.approx(
        f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    )
    assert stats.weighted_f1_from_counts(counts) == pytest.approx(
        f1_score(y, pred, labels=labels, average="weighted", zero_division=0)
    )
    assert stats.per_class_f1(counts) == pytest.approx(
        f1_score(y, pred, labels=labels, average=None, zero_division=0)
    )
    assert stats.accuracy_from_counts(counts) == pytest.approx(accuracy_score(y, pred))
    assert stats.macro_f1(y, pred, k) == pytest.approx(stats.macro_f1_from_counts(counts))


def test_a_class_that_is_never_present_or_predicted_scores_zero_like_scikit_learn():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 1, 0, 0])
    counts = stats.confusion_counts(y, pred, 3)  # lớp 2 không có trong y lẫn pred
    assert stats.per_class_f1(counts)[2] == 0.0
    assert stats.macro_f1_from_counts(counts) == pytest.approx(
        f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
    )


def test_resampled_counts_equal_the_direct_counts_row_by_row():
    y, pred = random_case(300, 4, 3)
    indices = stats.bootstrap_indices(300, rounds=25, seed=7)
    batched = stats.resampled_counts(y, pred, 4, indices)
    assert batched.shape == (25, 4, 4)
    for row in (0, 7, 24):
        assert np.array_equal(
            batched[row], stats.confusion_counts(y[indices[row]], pred[indices[row]], 4)
        )
    assert (batched.sum(axis=(1, 2)) == 300).all()


def test_batched_metric_samples_equal_scikit_learn_on_each_resample():
    y, pred = random_case(250, 3, 4)
    indices = stats.bootstrap_indices(250, rounds=10, seed=1)
    samples = stats.metric_samples(stats.resampled_counts(y, pred, 3, indices))
    for row in range(10):
        expected = f1_score(
            y[indices[row]], pred[indices[row]], labels=[0, 1, 2], average="macro", zero_division=0
        )
        assert samples["macro_f1"][row] == pytest.approx(expected)


def test_bootstrap_is_reproducible_and_depends_on_the_seed():
    y, pred = random_case(400, 3, 5)
    first = stats.bootstrap_intervals(y, pred, 3, rounds=300, seed=0)
    assert first == stats.bootstrap_intervals(y, pred, 3, rounds=300, seed=0)
    assert first != stats.bootstrap_intervals(y, pred, 3, rounds=300, seed=1)
    assert (first["rounds"], first["seed"], first["level"]) == (300, 0, 0.95)
    assert len(first["per_class_f1"]) == 3


def test_interval_contains_the_point_estimate_and_narrows_with_more_data():
    small_y, small_pred = random_case(150, 3, 6)
    large_y, large_pred = random_case(3000, 3, 6)
    small = stats.bootstrap_intervals(small_y, small_pred, 3)
    large = stats.bootstrap_intervals(large_y, large_pred, 3)
    for y, pred, interval in ((small_y, small_pred, small), (large_y, large_pred, large)):
        point = stats.macro_f1(y, pred, 3)
        assert interval["macro_f1"]["low"] <= point <= interval["macro_f1"]["high"]
        accuracy = float((y == pred).mean())
        assert interval["accuracy"]["low"] <= accuracy <= interval["accuracy"]["high"]
    width = lambda item: item["macro_f1"]["high"] - item["macro_f1"]["low"]  # noqa: E731
    assert width(large) < width(small)


def test_percentile_interval_on_a_known_sample():
    samples = np.arange(101, dtype=float)  # 0..100
    assert stats.percentile_interval(samples, 0.95) == (2.5, 97.5)
    assert stats.percentile_interval(samples, 0.5) == (25.0, 75.0)


def test_mcnemar_exact_known_values():
    correct_a = np.array([True] * 10 + [False] * 2 + [True] * 5)
    correct_b = np.array([False] * 10 + [True] * 2 + [True] * 5)
    result = stats.mcnemar_exact(correct_a, correct_b)
    assert (result["a_only_correct"], result["b_only_correct"]) == (10, 2)
    assert result["p_value"] == pytest.approx(2 * (1 + 12 + 66) / 4096)  # 2·P(X <= 2), n = 12


def test_mcnemar_equal_discordance_and_no_discordance_give_p_one():
    tie = stats.mcnemar_exact(np.array([True, False]), np.array([False, True]))
    assert tie["p_value"] == pytest.approx(1.0)
    none = stats.mcnemar_exact(np.array([True, False, True]), np.array([True, False, True]))
    assert none["p_value"] == 1.0 and none["a_only_correct"] == none["b_only_correct"] == 0


def test_paired_difference_is_antisymmetric_and_keeps_the_same_p_values():
    y, pred_a = random_case(600, 3, 8, noise=0.15)
    _, pred_b = random_case(600, 3, 9, noise=0.45)
    ab = stats.paired_difference(y, pred_a, pred_b, 3, rounds=500)
    ba = stats.paired_difference(y, pred_b, pred_a, 3, rounds=500)
    assert ab["delta_macro_f1"] > 0 and ab["delta_macro_f1"] == pytest.approx(-ba["delta_macro_f1"])
    assert ab["ci95_low"] == pytest.approx(-ba["ci95_high"])
    assert ab["ci95_high"] == pytest.approx(-ba["ci95_low"])
    assert ab["mcnemar_p"] == pytest.approx(ba["mcnemar_p"])
    assert ab["bootstrap_p"] == pytest.approx(ba["bootstrap_p"])
    assert (ab["a_only_correct"], ab["b_only_correct"]) == (
        ba["b_only_correct"],
        ba["a_only_correct"],
    )
    assert ab["mcnemar_p"] < 0.05 and ab["ci95_low"] > 0  # chênh rõ ràng thì phải có ý nghĩa


def test_identical_models_have_a_zero_difference_in_every_resample():
    """Ghép cặp thật: cùng chỉ số câu cho hai mô hình nên hai mô hình giống hệt cho hiệu = 0."""
    y, pred = random_case(400, 4, 10)
    result = stats.paired_difference(y, pred, pred.copy(), 4, rounds=200)
    assert result["delta_macro_f1"] == 0.0
    assert result["ci95_low"] == result["ci95_high"] == 0.0
    assert result["mcnemar_p"] == 1.0 and result["bootstrap_p"] == 1.0


def test_paired_difference_reports_both_models_numbers_and_the_run_settings():
    y, pred_a = random_case(300, 3, 11, noise=0.1)
    _, pred_b = random_case(300, 3, 12, noise=0.4)
    result = stats.paired_difference(y, pred_a, pred_b, 3, rounds=100, seed=3)
    assert result["accuracy_a"] == pytest.approx(accuracy_score(y, pred_a))
    assert result["macro_f1_b"] == pytest.approx(
        f1_score(y, pred_b, labels=[0, 1, 2], average="macro", zero_division=0)
    )
    assert (result["n"], result["rounds"], result["seed"]) == (300, 100, 3)
    assert result == stats.paired_difference(y, pred_a, pred_b, 3, rounds=100, seed=3)
