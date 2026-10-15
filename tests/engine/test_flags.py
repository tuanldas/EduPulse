"""Luật cờ và lý do bằng lời: dựng trên `edupulse.scoring`, không cần mô hình."""

import numpy as np
import pytest

from edupulse.engine.flags import (
    LOW_CONFIDENCE,
    NO_DIACRITICS,
    NOT_FLAGGED,
    Candidate,
    FlagReason,
    batch_needs_review,
    build_flag,
    single_needs_review,
    top_candidates,
    weakest_task,
)
from edupulse.labels import LABELS

TOPIC_PROBS = (0.58, 0.07, 0.31, 0.04)  # lecturer, training_program, facility, others


def test_weakest_task_is_the_head_with_the_lower_quantile():
    assert weakest_task(0.2, 0.7) == "sentiment"
    assert weakest_task(0.7, 0.2) == "topic"
    assert weakest_task(0.4, 0.4) == "sentiment"  # bằng nhau thì lấy cảm xúc


def test_top_candidates_are_the_two_most_likely_labels_in_order():
    candidates = top_candidates("topic", TOPIC_PROBS)
    assert candidates == (Candidate("lecturer", 0.58), Candidate("facility", 0.31))


def test_candidate_labels_belong_to_the_task():
    for task, labels in LABELS.items():
        probs = np.linspace(0.1, 0.3, len(labels))
        assert {c.label for c in top_candidates(task, probs)} <= set(labels)


def test_low_confidence_message_names_the_task_and_two_labels_with_percentages():
    reason = FlagReason(LOW_CONFIDENCE, "topic", top_candidates("topic", TOPIC_PROBS))
    assert (
        reason.message == "Chủ đề không chắc — phân vân giữa Giảng viên 58% và Cơ sở vật chất 31%"
    )

    sentiment = FlagReason(
        LOW_CONFIDENCE, "sentiment", top_candidates("sentiment", (0.41, 0.35, 0.24))
    )
    assert sentiment.message == "Cảm xúc không chắc — phân vân giữa Tiêu cực 41% và Trung tính 35%"


def test_no_diacritics_message_explains_why():
    message = FlagReason(NO_DIACRITICS).message
    assert "không dấu" in message


def test_single_sentence_is_flagged_strictly_below_tau():
    assert single_needs_review(0.049, no_diacritics=False, tau=0.05)
    assert not single_needs_review(0.05, no_diacritics=False, tau=0.05)
    assert not single_needs_review(0.9, no_diacritics=False, tau=0.05)


def test_no_diacritics_is_always_flagged_even_when_confident():
    assert single_needs_review(0.99, no_diacritics=True, tau=0.05)


def test_batch_flags_the_ceiling_of_the_budget_taking_the_lowest_scores():
    scores = np.linspace(0.05, 0.95, 20)  # 20 câu, ngân sách 10% => 2 câu thấp nhất
    flagged = batch_needs_review(scores, np.zeros(20, dtype=bool), 10)
    assert flagged.sum() == 2
    assert list(np.flatnonzero(flagged)) == [0, 1]

    shuffled = scores[::-1]
    flagged = batch_needs_review(shuffled, np.zeros(20, dtype=bool), 10)
    assert list(np.flatnonzero(flagged)) == [18, 19]


def test_batch_budget_rounds_up_so_a_small_batch_still_gets_a_flag():
    flagged = batch_needs_review([0.5, 0.2, 0.9], [False, False, False], 10)
    assert list(flagged) == [False, True, False]


def test_batch_counts_no_diacritics_sentences_inside_the_budget():
    scores = np.linspace(0.05, 0.95, 20)
    forced = np.zeros(20, dtype=bool)
    forced[15] = True  # câu chắc chắn nhưng gõ không dấu
    flagged = batch_needs_review(scores, forced, 10)
    assert flagged.sum() == 2
    assert flagged[15] and flagged[0]


def test_batch_flags_every_no_diacritics_sentence_even_beyond_the_budget():
    forced = np.zeros(20, dtype=bool)
    forced[[3, 9, 12]] = True
    flagged = batch_needs_review(np.linspace(0.05, 0.95, 20), forced, 10)
    assert flagged.sum() == 3
    assert list(np.flatnonzero(flagged)) == [3, 9, 12]


def test_batch_ties_are_resolved_by_row_order():
    flagged = batch_needs_review([0.3] * 10, [False] * 10, 30)
    assert list(np.flatnonzero(flagged)) == [0, 1, 2]


def test_unflagged_sentence_has_no_reasons():
    assert (
        build_flag(flagged=False, no_diacritics=False, task="topic", probs=TOPIC_PROBS)
        is NOT_FLAGGED
    )


def test_flagged_by_confidence_names_the_weak_head_and_its_top_labels():
    flag = build_flag(flagged=True, no_diacritics=False, task="topic", probs=TOPIC_PROBS)
    assert flag.needs_review
    (reason,) = flag.reasons
    assert (reason.code, reason.task) == (LOW_CONFIDENCE, "topic")
    assert [c.label for c in reason.candidates] == ["lecturer", "facility"]


def test_no_diacritics_reason_replaces_the_confidence_reason():
    flag = build_flag(flagged=True, no_diacritics=True, task="topic", probs=TOPIC_PROBS)
    assert [r.code for r in flag.reasons] == [NO_DIACRITICS]


@pytest.mark.parametrize("budget", [5, 10, 20, 30])
def test_batch_flag_share_follows_the_budget(budget):
    scores = np.random.default_rng(0).uniform(size=1000)
    flagged = batch_needs_review(scores, np.zeros(1000, dtype=bool), budget)
    assert flagged.sum() == budget * 10
