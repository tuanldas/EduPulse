"""Độ tin cậy, điểm cờ, ngân sách duyệt; quan trọng nhất: đường calibrate khớp đường chạy thật."""

import math

import numpy as np
import pytest

from edupulse.config import BUDGETS
from edupulse.scoring import (
    CDF_POINTS,
    confidence,
    efficiency,
    flag_score,
    flags_by_threshold,
    make_cdf,
    quantile_rank,
    rank_batch,
    softmax_t,
    threshold_for_budget,
)


def test_softmax_rows_sum_to_one_and_keep_order():
    probs = softmax_t([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0].argmax() == 0
    assert np.allclose(probs[1], 1 / 3)


def test_softmax_is_stable_for_huge_logits():
    probs = softmax_t([1000.0, 999.0, -1000.0])
    assert np.isfinite(probs).all()
    assert probs.argmax() == 0
    assert probs.sum() == pytest.approx(1.0)


def test_higher_temperature_lowers_confidence():
    logits = [[3.0, 0.5, -1.0]]
    assert confidence(softmax_t(logits, 2.0))[0] < confidence(softmax_t(logits, 1.0))[0]
    assert np.allclose(softmax_t(logits, 1.0), softmax_t(logits))


@pytest.mark.parametrize("temperature", [0, -1.0, float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        softmax_t([1.0, 2.0], temperature)


def test_make_cdf_has_101_monotone_points_spanning_the_range():
    values = np.random.default_rng(0).uniform(0.4, 1.0, size=500)
    cdf = make_cdf(values)
    assert len(cdf) == CDF_POINTS == 101
    assert cdf == sorted(cdf)
    assert cdf[0] == pytest.approx(values.min())
    assert cdf[-1] == pytest.approx(values.max())
    with pytest.raises(ValueError):
        make_cdf([])


def test_quantile_rank_is_monotone_and_bounded():
    values = np.random.default_rng(1).uniform(0.4, 1.0, size=1000)
    cdf = make_cdf(values)
    grid = np.linspace(0.0, 1.0, 400)
    ranks = quantile_rank(grid, cdf)
    assert (np.diff(ranks) >= 0).all()
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0
    assert quantile_rank(values.min() - 1, cdf) == 0.0
    assert quantile_rank(values.max() + 1, cdf) == 1.0
    # Trên chính tập tạo bảng, phân vị xấp xỉ vị trí xếp hạng thật.
    empirical = np.argsort(np.argsort(values)) / (len(values) - 1)
    assert np.abs(quantile_rank(values, cdf) - empirical).max() < 0.02


def test_quantile_rank_handles_saturated_plateaus():
    # Độ tin cậy bão hoà gần 1,0: nhiều điểm của bảng trùng nhau.
    values = np.concatenate([np.linspace(0.5, 0.9, 30), np.full(70, 0.999)])
    cdf = make_cdf(values)
    assert len(set(cdf)) < len(cdf)
    ranks = quantile_rank([0.5, 0.7, 0.999, 1.0], cdf)
    assert (np.diff(ranks) >= 0).all()
    assert ranks[0] == 0.0
    assert ranks[-1] == ranks[-2]  # trên vùng bão hoà không phân biệt được nữa


def test_flag_score_takes_the_less_certain_head():
    assert flag_score([0.9, 0.2], [0.3, 0.8]).tolist() == [0.3, 0.2]


# --- Ngưỡng ngân sách và xếp hạng lô ----------------------------------------------------------


@pytest.mark.parametrize("pct", BUDGETS)
@pytest.mark.parametrize("n", [37, 200, 1000, 1583])
def test_threshold_flags_the_budgeted_share_of_validation(n, pct):
    scores = np.random.default_rng(n).uniform(size=n)
    tau = threshold_for_budget(scores, pct)
    expected = math.ceil(round(pct * n / 100, 9))
    assert flags_by_threshold(scores, tau).sum() == expected


@pytest.mark.parametrize("pct", BUDGETS)
@pytest.mark.parametrize("n", [37, 200, 1000, 1583])
def test_calibration_path_equals_batch_path(n, pct):
    """Bảng hiệu quả tính bằng ngưỡng trên val phải trùng kết quả `rank_batch` trên chính val."""
    scores = np.random.default_rng(n + pct).uniform(size=n)
    by_threshold = flags_by_threshold(scores, threshold_for_budget(scores, pct))
    by_rank = rank_batch(scores, pct)
    assert (by_threshold == by_rank).all()
    rng = np.random.default_rng(7)
    ok_s, ok_t = rng.random(n) > 0.1, rng.random(n) > 0.2
    assert efficiency(by_threshold, ok_s, ok_t) == efficiency(by_rank, ok_s, ok_t)


def test_threshold_with_ties_never_flags_more_than_the_budget():
    scores = np.array([0.1] * 5 + [0.5] * 90 + [0.9] * 5)
    tau = threshold_for_budget(scores, 10)
    assert flags_by_threshold(scores, tau).sum() <= 10


def test_threshold_edge_cases():
    assert threshold_for_budget([0.3], 10) == math.inf  # lô 1 câu: ngân sách làm tròn lên 1 câu
    assert threshold_for_budget([0.1, 0.2, 0.3, 0.4], 100) == math.inf
    with pytest.raises(ValueError):
        threshold_for_budget([], 10)
    for bad in (0, -5, 101):
        with pytest.raises(ValueError, match="ngân sách"):
            threshold_for_budget([0.1, 0.2], bad)


def test_rank_batch_flags_ceil_of_the_budget_lowest_scores():
    scores = np.arange(30, 0, -1) / 30  # câu cuối thấp nhất
    flags = rank_batch(scores, 10)
    assert flags.sum() == 3
    assert flags[-3:].all() and not flags[:-3].any()
    assert rank_batch(scores, 5).sum() == 2  # ⌈1,5⌉
    assert rank_batch(np.array([0.5]), 5).sum() == 1  # lô 1 câu


def test_rank_batch_counts_forced_rows_against_the_budget_first():
    scores = np.linspace(0.0, 1.0, 20)  # ngân sách 10% = 2 câu
    forced = np.zeros(20, dtype=bool)
    forced[15] = True
    flags = rank_batch(scores, 10, forced)
    assert flags[15] and flags[0] and flags.sum() == 2  # 1 bắt buộc + 1 thấp nhất


def test_rank_batch_flags_every_forced_row_even_beyond_the_budget():
    scores = np.linspace(0.0, 1.0, 20)
    forced = np.zeros(20, dtype=bool)
    forced[[5, 6, 7, 8]] = True  # 4 > ngân sách 2
    flags = rank_batch(scores, 10, forced)
    assert flags.sum() == 4
    assert flags[[5, 6, 7, 8]].all()


def test_rank_batch_is_deterministic_with_tied_scores():
    scores = np.full(20, 0.5)
    first = rank_batch(scores, 20)
    assert first.sum() == 4
    assert first[:4].all()  # hoà điểm: theo thứ tự dòng
    assert (rank_batch(scores, 20) == first).all()


def test_rank_batch_validates_shapes_and_budget():
    with pytest.raises(ValueError, match="cùng kích thước"):
        rank_batch([0.1, 0.2], 10, [True])
    with pytest.raises(ValueError, match="ngân sách"):
        rank_batch([0.1, 0.2], 0)
    assert rank_batch(np.array([]), 10).size == 0


def test_rank_batch_does_not_mutate_inputs():
    scores = np.array([0.4, 0.1, 0.9])
    forced = np.array([False, False, True])
    rank_batch(scores, 30, forced)
    assert forced.tolist() == [False, False, True]
    assert scores.tolist() == [0.4, 0.1, 0.9]


# --- Hiệu quả -----------------------------------------------------------------------------------


def test_efficiency_on_a_hand_checked_example():
    flags = [True, True, False, False, False, False, False, False, False, False]
    ok_s = [False, True, True, True, True, True, True, True, True, True]  # 1 lỗi, nằm trong cờ
    ok_t = [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # 2 lỗi: 1 trong cờ, 1 ngoài
    result = efficiency(flags, ok_s, ok_t)
    assert result["flagged_share"] == 0.2
    assert result["caught_sentiment_errors"] == 1.0
    assert result["caught_topic_errors"] == 0.5
    # Phần không cờ (8 câu): câu thứ 3 sai chủ đề nên đúng cả hai là 7/8.
    assert result["accept_accuracy_both"] == 7 / 8
    assert result["all_accuracy_both"] == 0.7


def test_efficiency_degenerate_cases():
    ok = [True, True, True]
    none_flagged = efficiency([False] * 3, ok, ok)
    assert none_flagged["caught_sentiment_errors"] == 0.0  # không có lỗi nào để bắt
    assert none_flagged["accept_accuracy_both"] == 1.0
    all_flagged = efficiency([True] * 3, ok, ok)
    assert all_flagged["accept_accuracy_both"] == 1.0  # không còn phần chấp nhận
    with pytest.raises(ValueError):
        efficiency([True, False], [True], [True])
    with pytest.raises(ValueError):
        efficiency([], [], [])
