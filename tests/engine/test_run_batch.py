"""`Analyzer.run_batch`: đường của job nền, kiểm trên bundle thật."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from edupulse.config import AUDIT_RATE, AUTO_XAI_TOP
from edupulse.engine.service import Analyzer, BatchCancelled, NoContent

pytestmark = pytest.mark.model

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sentences.json"
BUDGET = 20


@pytest.fixture(scope="module")
def texts() -> list[str]:
    """Câu thật: 20 câu vàng tự viết lặp lại nhiều lần với hậu tố khác nhau để có ~100 câu."""
    items = json.loads(GOLDEN.read_text(encoding="utf-8"))["items"]
    base = [i["text"] for i in items if "no_diacritics" not in i["covers"]]
    return [f"{text} {'!' * (k % 3)}".strip() for k in range(6) for text in base]


@pytest.fixture(scope="module")
def outcome(analyzer: Analyzer, texts):
    return analyzer.run_batch(texts, BUDGET, seed=11)


def test_every_row_has_a_label_and_the_lists_align(outcome, texts):
    assert len(outcome.prepared) == len(outcome.analyses) == len(outcome.audit) == len(texts)
    assert all(a is not None for a in outcome.analyses)
    assert outcome.n_skipped == 0


def test_flags_follow_the_budget_over_the_analyzed_rows(outcome, texts):
    assert outcome.n_flagged == math.ceil(BUDGET * len(texts) / 100)


def test_audit_is_a_share_of_the_unflagged_rows_and_never_overlaps_the_flags(outcome, texts):
    flagged = np.array([a.flag.needs_review for a in outcome.analyses])
    assert outcome.audit.sum() == math.ceil(AUDIT_RATE * (len(texts) - flagged.sum()))
    assert not (outcome.audit & flagged).any()


def test_review_queue_is_flags_plus_audit_in_ascending_score_order(outcome):
    flagged = {i for i, a in enumerate(outcome.analyses) if a.flag.needs_review}
    audited = set(np.flatnonzero(outcome.audit))
    assert set(outcome.review_order) == flagged | audited
    assert len(outcome.review_order) == len(set(outcome.review_order))
    scores = [outcome.analyses[i].score for i in outcome.review_order]
    assert scores == sorted(scores)


def test_same_seed_same_result_other_seed_other_sample_same_flags(analyzer, texts, outcome):
    again = analyzer.run_batch(texts, BUDGET, seed=11)
    assert (again.audit == outcome.audit).all() and again.review_order == outcome.review_order
    other = analyzer.run_batch(texts, BUDGET, seed=12)
    assert not (other.audit == outcome.audit).all()
    assert [a.flag.needs_review for a in other.analyses] == [
        a.flag.needs_review for a in outcome.analyses
    ]


def test_labels_match_analyzing_each_sentence_alone(analyzer, texts, outcome):
    for text, analysis in list(zip(texts, outcome.analyses, strict=True))[:25]:
        alone = analyzer.analyze(text)
        assert (analysis.sentiment, analysis.topic) == (alone.sentiment, alone.topic)
        assert analysis.score == alone.score


def test_auto_explanations_are_the_lowest_scoring_flagged_rows_within_the_word_limit(
    analyzer, outcome
):
    eligible = [
        i
        for i, a in enumerate(outcome.analyses)
        if a.flag.needs_review and a.prepared.n_words <= analyzer.xai_max_words
    ]
    expected = sorted(eligible, key=lambda i: (outcome.analyses[i].score, i))[:AUTO_XAI_TOP]
    assert sorted(outcome.explanations) == sorted(expected)
    for index, explanation in outcome.explanations.items():
        analysis = outcome.analyses[index]
        assert explanation.sentiment.label == analysis.sentiment.label
        assert len(explanation.topic.tokens) == analysis.prepared.n_words


def test_the_number_of_auto_explanations_is_capped(analyzer, texts):
    outcome = analyzer.run_batch(texts, 30, seed=1, explain_top=3)
    assert len(outcome.explanations) == 3
    assert not analyzer.run_batch(texts[:10], 30, seed=1, explain_top=0).explanations


def test_rows_without_content_get_no_label_and_stay_out_of_the_budget(analyzer, texts):
    mixed = [texts[0], "0912345678", texts[1], "🙂🙂", texts[2], texts[3], texts[4], texts[5]]
    outcome = analyzer.run_batch(mixed, 30, seed=1)
    assert outcome.n_skipped == 2
    assert outcome.analyses[1] is None and outcome.analyses[3] is None
    assert outcome.n_flagged == math.ceil(30 * 6 / 100)  # tính trên 6 câu có nhãn
    assert set(outcome.review_order).isdisjoint({1, 3})
    assert not outcome.audit[1] and not outcome.audit[3]
    assert all(i in {1, 3} or a is not None for i, a in enumerate(outcome.analyses))


def test_a_batch_with_nothing_to_analyze_is_refused(analyzer):
    with pytest.raises(NoContent) as refused:
        analyzer.run_batch(["0912345678", "   ", "a@b.vn"], 10, seed=1)
    assert refused.value.rows == (0, 1, 2)


def test_progress_covers_the_steps_in_order_and_never_goes_backwards(analyzer, texts):
    events: list[tuple[str, int, int]] = []
    analyzer.run_batch(texts[:60], 20, seed=1, explain_top=5, progress=lambda *e: events.append(e))
    order = ["prepare", "predict", "flag", "explain"]
    steps = [event[0] for event in events]
    assert sorted(set(steps), key=order.index) == order
    assert steps == sorted(steps, key=order.index)
    for step in order:
        done = [d for s, d, _ in events if s == step]
        assert done == sorted(done), step
        totals = {t for s, _, t in events if s == step}
        assert len(totals) == 1 and done[-1] == totals.pop()
    assert [t for s, _, t in events if s == "predict"][0] == 60


def test_cancelling_stops_before_the_next_step(analyzer, texts):
    seen: list[str] = []

    def cancel_after_predict_starts() -> bool:
        return "predict" in seen

    with pytest.raises(BatchCancelled):
        analyzer.run_batch(
            texts,
            10,
            seed=1,
            progress=lambda step, *_: seen.append(step),
            cancelled=cancel_after_predict_starts,
        )
    assert "explain" not in seen


def test_steps_are_timed(outcome):
    assert set(outcome.steps_s) == {"prepare", "predict", "flags", "explain"}
    assert all(seconds >= 0 for seconds in outcome.steps_s.values())
