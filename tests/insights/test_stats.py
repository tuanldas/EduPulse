"""Wilson, log-odds có trọng số và đếm tài liệu."""

import math

import pytest

from edupulse.insights.stats import (
    document_frequency,
    weighted_log_odds,
    wilson_interval,
    wilson_lower,
)

# --- Wilson -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "lower"),
    [(4, 5, 0.38), (28, 40, 0.55), (70, 100, 0.60), (182, 195, 0.89)],
)
def test_wilson_lower_bounds_match_the_hand_checked_values(k: int, n: int, lower: float):
    assert wilson_lower(k, n) == pytest.approx(lower, abs=0.005)


def test_wilson_matches_the_closed_form():
    k, n, z = 70, 100, 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert wilson_interval(k, n) == pytest.approx((centre - margin, centre + margin))


def test_wilson_without_observations_knows_nothing():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(("k", "n"), [(0, 10), (10, 10), (1, 1), (0, 1)])
def test_wilson_stays_inside_the_unit_interval_at_the_extremes(k: int, n: int):
    low, high = wilson_interval(k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0
    assert low < high


def test_wilson_is_symmetric_and_shrinks_with_more_data():
    low, high = wilson_interval(30, 100)
    mirror_low, mirror_high = wilson_interval(70, 100)
    assert (low, high) == pytest.approx((1 - mirror_high, 1 - mirror_low))
    narrow = wilson_interval(300, 1000)
    assert narrow[1] - narrow[0] < high - low
    assert wilson_lower(8, 10) < wilson_lower(80, 100) < wilson_lower(800, 1000)


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(k: int, n: int):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


# --- Log-odds có trọng số -----------------------------------------------------------------------


def background(**counts: int) -> tuple[dict[str, int], int]:
    """Prior nền: tần suất của vài n-gram và tổng số lần xuất hiện của mọi n-gram."""
    return counts, sum(counts.values()) + 1000


def test_a_term_concentrated_in_the_group_scores_high_and_common_terms_near_zero():
    common = [f"từ{i}" for i in range(10)]
    group = {"máy_chiếu": 30, **{t: 40 for t in common}}
    rest = {"máy_chiếu": 2, **{t: 400 for t in common}}
    prior, total = background(máy_chiếu=40, **{t: 4000 for t in common})
    z = weighted_log_odds(group, rest, prior, prior_total=total, alpha0=1000)
    assert z["máy_chiếu"] > 3
    assert all(abs(z[t]) < 1.0 for t in common)  # tần suất tương đối gần như nhau ở hai bên
    assert z["máy_chiếu"] == max(z.values())


def test_a_term_rarer_in_the_group_than_outside_scores_negative():
    prior, total = background(tốt=500, máy=100)
    z = weighted_log_odds(
        {"tốt": 1, "máy": 30}, {"tốt": 300, "máy": 20}, prior, prior_total=total, alpha0=1000
    )
    assert z["tốt"] < 0 < z["máy"]


def test_the_prior_shrinks_scores_of_rarely_seen_terms():
    prior, total = background(a=500, b=500)
    group_a = {"a": 20, "b": 1000}
    rest = {"a": 1, "b": 1000}
    z_many = weighted_log_odds(group_a, rest, prior, prior_total=total, alpha0=1000)["a"]
    z_few = weighted_log_odds(
        {"a": 2, "b": 1000}, {"a": 0, "b": 1000}, prior, prior_total=total, alpha0=1000
    )["a"]
    assert z_many > z_few > 0


def test_stronger_prior_pulls_scores_toward_zero():
    prior, total = background(x=100, y=100)
    group, rest = {"x": 40, "y": 10}, {"x": 5, "y": 10}
    weak = weighted_log_odds(group, rest, prior, prior_total=total, alpha0=10)["x"]
    strong = weighted_log_odds(group, rest, prior, prior_total=total, alpha0=100_000)["x"]
    assert abs(strong) < abs(weak)


def test_a_new_term_missing_from_the_prior_can_still_stand_out_if_repeated():
    prior, total = background(a=100)
    group = {"a": 5, "zoom": 40}
    z = weighted_log_odds(group, {"a": 5}, prior, prior_total=total, alpha0=1000)
    assert z["zoom"] > 2
    # Chỉ vài lần thì chưa đủ để nổi bật.
    few = weighted_log_odds({"a": 5, "zoom": 1}, {"a": 5}, prior, prior_total=total, alpha0=1000)
    assert few["zoom"] < z["zoom"] and few["zoom"] < 1


def test_the_floor_only_matters_for_terms_the_prior_barely_knows():
    prior, total = background(a=100_000, b=1)
    group, rest = {"a": 30, "b": 3}, {"a": 5, "b": 0}
    low = weighted_log_odds(group, rest, prior, prior_total=total, alpha0=1000, alpha_floor=0.01)
    high = weighted_log_odds(group, rest, prior, prior_total=total, alpha0=1000, alpha_floor=0.5)
    assert low["a"] == pytest.approx(high["a"])  # prior của "a" lớn hơn mọi sàn
    assert low["b"] != pytest.approx(high["b"])


def test_the_function_scores_only_terms_that_occur_in_the_group():
    prior, total = background(a=10, b=10)
    z = weighted_log_odds({"a": 3}, {"a": 1, "b": 9}, prior, prior_total=total, alpha0=1000)
    assert set(z) == {"a"}


@pytest.mark.parametrize(
    ("group", "rest"),
    [({}, {"a": 1}), ({"a": 1}, {}), ({}, {})],
    ids=["no-group", "no-rest", "none"],
)
def test_no_data_means_no_scores(group, rest):
    assert weighted_log_odds(group, rest, {"a": 1}, prior_total=10, alpha0=1000) == {}


def test_degenerate_prior_means_no_scores():
    assert weighted_log_odds({"a": 1}, {"a": 1}, {"a": 1}, prior_total=0, alpha0=1000) == {}
    assert weighted_log_odds({"a": 1}, {"a": 1}, {"a": 1}, prior_total=10, alpha0=0) == {}


def test_scores_are_symmetric_when_the_groups_swap():
    prior, total = background(x=100, y=300)
    a, b = {"x": 20, "y": 5}, {"x": 3, "y": 40}
    forward = weighted_log_odds(a, b, prior, prior_total=total, alpha0=1000)
    backward = weighted_log_odds(b, a, prior, prior_total=total, alpha0=1000)
    assert forward["x"] == pytest.approx(-backward["x"])
    assert forward["y"] == pytest.approx(-backward["y"])


# --- Đếm tài liệu ---------------------------------------------------------------------------------


def test_document_frequency_counts_each_sentence_once():
    counts = document_frequency([["a", "a", "b"], ["a"], []])
    assert counts == {"a": 2, "b": 1}
